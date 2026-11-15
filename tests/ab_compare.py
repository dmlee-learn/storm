#!/usr/bin/env python
"""Pre/post optimization A/B comparison harness.

Compares implementations at two points in time by extracting the actual
functions from each revision and running both on identical inputs. Useful for
tracking how an optimization changes behavior across releases.

    python tests/ab_compare.py                    # last commit vs working tree
    python tests/ab_compare.py --old v1.1.0 --new HEAD
    python tests/ab_compare.py --old HEAD~5 --new HEAD --strict

Only numpy is required. No network access: dspy, sentence-transformers and the
embedding provider are stubbed.

Checks marked "known divergence" are intentional behavior changes documented in
OPTIMIZATION.md. Everything else must match exactly; `--strict` turns an
unexpected difference into a non-zero exit code.
"""

import argparse
import ast
import copy
import hashlib
import random
import re
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORKTREE = "worktree"

CITATION = "knowledge_storm/utils.py"
ENCODER = "knowledge_storm/encoder.py"
STORM_DATACLASS = "knowledge_storm/storm_wiki/modules/storm_dataclass.py"
RETRIEVER = "knowledge_storm/storm_wiki/modules/retriever.py"


# --------------------------------------------------------------------- helpers
def source_at(ref: str, relpath: str) -> str:
    """Return the contents of `relpath` at git `ref` (or the working tree)."""
    if ref == WORKTREE:
        return (REPO / relpath).read_text(encoding="utf-8")
    result = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:{relpath}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"git show {ref}:{relpath} failed:\n{result.stderr}")
    return result.stdout


def load_class_methods(source, class_name, method_names, extra_namespace):
    """Exec a class with only `method_names` kept, so it can be exercised alone."""
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.bases = [ast.parse("object").body[0].value]
    cls.keywords = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names
    ]
    namespace = dict(extra_namespace)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<ab>", "exec"), namespace)
    return namespace[class_name]


def load_static_method(source, class_name, method_name, extra_namespace):
    return load_class_methods(
        source, class_name, {method_name}, extra_namespace
    ).__dict__[method_name]


class Result:
    def __init__(self, name, same, expected_same=True, detail=""):
        self.name = name
        self.same = same
        self.expected_same = expected_same
        self.detail = detail

    @property
    def ok(self):
        # Only equivalence checks can fail. Known divergences are informational.
        return self.same or not self.expected_same

    def line(self):
        if self.same and self.expected_same:
            status = "SAME"
        elif self.same and not self.expected_same:
            status = "SAME (divergence gone)"
        elif not self.same and self.expected_same:
            status = "DIFF (unexpected)"
        else:
            status = "DIFF (known)"
        text = f"  [{status:22}] {self.name}"
        if self.detail:
            text += f" :: {self.detail}"
        return text


# ---------------------------------------------------------------- E11 citation
def check_citation(old_ref, new_ref):
    namespace = {"re": re}
    old = load_static_method(
        source_at(old_ref, CITATION),
        "ArticleTextProcessing",
        "update_citation_index",
        namespace,
    )
    new = load_static_method(
        source_at(new_ref, CITATION),
        "ArticleTextProcessing",
        "update_citation_index",
        namespace,
    )

    rng = random.Random(0)
    mismatches = 0
    cases = 0
    for _ in range(3000):
        k = rng.randint(1, 12)
        mapping = {i: rng.randint(1, 20) for i in rng.sample(range(1, 15), k)}
        parts = []
        for _ in range(rng.randint(0, 10)):
            if rng.random() < 0.5:
                parts.append(f"[{rng.randint(1, 16)}]")
            else:
                parts.append(rng.choice(["alpha", "beta", "[]", "[x]", "[12]]"]))
        text = " ".join(parts)
        cases += 1
        if old(text, mapping) != new(text, mapping):
            mismatches += 1
    return Result(
        "E11 citation index rewrite",
        mismatches == 0,
        detail=f"{mismatches}/{cases} cases differ",
    )


# ------------------------------------------------------------------ E1 encoder
class FakeEmbeddingResponse:
    def __init__(self, data, usage):
        self.data = data
        self._usage = usage

    def get(self, key, default=None):
        return self._usage if key == "usage" else default


def install_litellm_stub():
    fake = types.ModuleType("litellm")

    def embedding(model, input, caching=True, **kwargs):
        texts = list(input) if isinstance(input, (list, tuple)) else [input]
        data = []
        for index, text in enumerate(texts):
            digest = hashlib.md5(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            vector = np.random.default_rng(seed).normal(size=8)
            data.append({"embedding": vector.tolist(), "index": index})
        # Reversed order verifies that results are restored by `index`.
        return FakeEmbeddingResponse(list(reversed(data)), {"total_tokens": 1})

    fake.embedding = embedding
    fake.drop_params = True
    fake.telemetry = False
    sys.modules["litellm"] = fake

    caching_pkg = types.ModuleType("litellm.caching")
    caching_mod = types.ModuleType("litellm.caching.caching")

    class Cache:
        def __init__(self, **kwargs):
            pass

    caching_mod.Cache = Cache
    sys.modules["litellm.caching"] = caching_pkg
    sys.modules["litellm.caching.caching"] = caching_mod


def load_encoder_module(ref, name):
    install_litellm_stub()
    module = types.ModuleType(name)
    module.__file__ = str(REPO / ENCODER)
    exec(compile(source_at(ref, ENCODER), str(REPO / ENCODER), "exec"), module.__dict__)
    return module


def check_encoder(old_ref, new_ref):
    old = load_encoder_module(old_ref, "encoder_old").Encoder(
        encoder_type="openai", api_key="test"
    )
    new = load_encoder_module(new_ref, "encoder_new").Encoder(
        encoder_type="openai", api_key="test"
    )

    rng = random.Random(1)
    texts = [f"text-{i}-{rng.randint(0, 999)}" for i in range(200)]
    old_vec = np.array(old.encode(texts))
    new_vec = np.array(new.encode(texts))
    same_shape = old_vec.shape == new_vec.shape
    max_diff = float(np.abs(old_vec - new_vec).max()) if same_shape else float("nan")

    duplicate_texts = ["a", "bb", "a", "ccc"]
    old_dup = np.array(old.encode(duplicate_texts))
    new_dup = np.array(new.encode(duplicate_texts))
    duplicates_same = old_dup.shape == new_dup.shape and np.allclose(old_dup, new_dup)

    return [
        Result(
            "E1 encoder, 200 distinct texts",
            same_shape and max_diff < 1e-9,
            detail=f"shape {old_vec.shape} vs {new_vec.shape}, max|diff|={max_diff:.1e}",
        ),
        Result(
            "E1 encoder, duplicate texts keep input order",
            duplicates_same,
            expected_same=False,
            detail=(
                "old reorders duplicates (a,a,bb,ccc); new preserves a,bb,a,ccc"
                if not duplicates_same
                else "both preserve order"
            ),
        ),
    ]


# ---------------------------------------------------------------- E2 retrieval
class Info:
    def __init__(self, url, snippets, meta=None):
        self.url = url
        self.description = "description"
        self.snippets = list(snippets)
        self.title = "title"
        self.meta = meta or {"query": "q"}


def cosine_similarity(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    y_norm = y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-12, None)
    return x_norm @ y_norm.T


class StubEncoder:
    """Models a sentence transformer: raw vectors, optional L2 normalization."""

    def __init__(self, *args, **kwargs):
        self.dim = 32

    def _vec(self, text):
        digest = hashlib.md5(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        return np.random.default_rng(seed).normal(size=self.dim)

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False, **kw):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = np.array([self._vec(text) for text in items])
        if normalize_embeddings and len(vectors):
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors[0] if single else vectors


def load_retrieval_implementation(ref):
    """Load StormInformationTable with its own prepare_table_for_retrieval.

    The search path changed shape between versions: older code stored raw
    embeddings and scored with cosine_similarity, newer code stores normalized
    embeddings and scores with a dot product. Running each version's own
    prepare step keeps the comparison independent of which ref is which.
    """
    source = source_at(ref, STORM_DATACLASS)
    namespace = {
        "np": np,
        "copy": copy,
        "cosine_similarity": cosine_similarity,
        "Information": Info,
        "List": List,
        "Union": Union,
        "Optional": Optional,
        "Dict": Dict,
        "re": re,
        "threading": threading,
        "SentenceTransformer": StubEncoder,
    }
    class_name = "StormInformationTable"
    method_names = {"prepare_table_for_retrieval", "retrieve_information"}
    module_names = {
        "_SENTENCE_TRANSFORMER",
        "_SENTENCE_TRANSFORMER_LOCK",
        "_get_sentence_transformer",
    }

    for node in ast.parse(source).body:
        selected = None
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            node.bases = [ast.parse("object").body[0].value]
            node.keywords = []
            node.body = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name in method_names
            ]
            selected = node
        elif isinstance(node, ast.Assign) and _assigned_names(node) & module_names:
            selected = node
        elif isinstance(node, ast.FunctionDef) and node.name in module_names:
            selected = node
        if selected is not None:
            exec(
                compile(ast.Module(body=[selected], type_ignores=[]), "<ab>", "exec"),
                namespace,
            )

    return namespace[class_name]


def build_table(cls, snippets, urls):
    """Populate url_to_info, then let the implementation prepare embeddings."""
    table = cls.__new__(cls)
    table.url_to_info = {}
    for url, snippet in zip(urls, snippets):
        table.url_to_info.setdefault(url, Info(url, []))
        table.url_to_info[url].snippets.append(snippet)
    table.prepare_table_for_retrieval()
    return table


def selection(table, query, k):
    return tuple(
        (info.url, tuple(sorted(info.snippets)))
        for info in table.retrieve_information(query, k)
    )


def check_retrieval(old_ref, new_ref):
    old_cls = load_retrieval_implementation(old_ref)
    new_cls = load_retrieval_implementation(new_ref)

    rng = random.Random(2)
    snippets = [f"snippet {i} topic {rng.randint(0, 50)}" for i in range(300)]
    urls = [f"http://site{i % 40}.com/page" for i in range(300)]
    queries = [f"query about topic {i}" for i in range(5)]

    old_table = build_table(old_cls, snippets, urls)
    new_table = build_table(new_cls, snippets, urls)

    total = 0
    mismatched = 0
    for query in queries:
        for k in (1, 3, 5, 10):
            total += 1
            if selection(old_table, query, k) != selection(new_table, query, k):
                mismatched += 1

    # Exact ties: identical snippets on every URL make all scores equal.
    tie_snippets = [f"same text {i}" for i in range(50)]
    tie_urls = [f"http://tie{i}.com" for i in range(50)]
    old_tie = build_table(old_cls, tie_snippets, tie_urls)
    new_tie = build_table(new_cls, tie_snippets, tie_urls)
    old_tie.encoded_snippets = np.tile(old_tie.encoded_snippets[0], (50, 1))
    new_tie.encoded_snippets = old_tie.encoded_snippets.copy()
    old_pick = tuple(info.url for info in old_tie.retrieve_information("q", 5))
    new_pick = tuple(info.url for info in new_tie.retrieve_information("q", 5))
    old_repeat = old_pick == tuple(
        info.url for info in old_tie.retrieve_information("q", 5)
    )
    new_repeat = new_pick == tuple(
        info.url for info in new_tie.retrieve_information("q", 5)
    )

    return [
        Result(
            "E2 retrieval ranking, 300 snippets x 5 queries",
            mismatched == 0,
            detail=f"{mismatched}/{total} query/k combinations differ",
        ),
        Result(
            "E2 exact-tie selection",
            old_pick == new_pick,
            expected_same=False,
            detail=(
                f"old={[u.split('.')[0][-4:] for u in old_pick]}, "
                f"new={[u.split('.')[0][-4:] for u in new_pick]}, "
                f"repeatable old={old_repeat} new={new_repeat}"
            ),
        ),
    ]


# -------------------------------------------------------------- E5 content union
class TreeNode:
    def __init__(self, name, content=()):
        self.name = name
        self.content = set(content)
        self.children = []

    def collect_all_content(self):
        collected = set(self.content)
        for child in self.children:
            collected.update(child.collect_all_content())
        return collected


def check_content_union():
    """Post-order union (new call site) vs recomputing per node (old call site)."""
    rng = random.Random(3)

    def build(depth=0):
        node = TreeNode(
            f"n{depth}", {rng.randint(0, 200) for _ in range(rng.randint(0, 4))}
        )
        if depth < 4:
            for _ in range(rng.randint(0, 4)):
                node.children.append(build(depth + 1))
        return node

    mismatches = 0
    nodes_checked = 0
    for _ in range(200):
        root = build()
        by_id = {}

        def post_order(node):
            content = set(node.content)
            for child in node.children:
                content |= post_order(child)
            by_id[id(node)] = content
            return content

        post_order(root)

        stack = [root]
        while stack:
            node = stack.pop()
            nodes_checked += 1
            if node.collect_all_content() != by_id[id(node)]:
                mismatches += 1
            stack.extend(node.children)

    return Result(
        "E5 knowledge-base content union",
        mismatches == 0,
        detail=f"{mismatches}/{nodes_checked} nodes differ over 200 random trees",
    )


# -------------------------------------------------------------- E6 section insert
class ArticleSectionNode:
    def __init__(self, section_name, content=None):
        self.section_name = section_name
        self.content = content or ""
        self.children = []

    def add_child(self, child, insert_to_front=False):
        if insert_to_front:
            self.children.insert(0, child)
        else:
            self.children.append(child)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)

    def dump(self):
        return (self.section_name, self.content, [c.dump() for c in self.children])


def _article_class(ref):
    class_namespace = {
        "Dict": Dict,
        "Optional": Optional,
        "ArticleSectionNode": ArticleSectionNode,
    }
    cls = load_class_methods(
        source_at(ref, STORM_DATACLASS),
        "StormArticle",
        {"insert_or_create_section", "find_section"},
        class_namespace,
    )

    class Article(cls):
        def __init__(self):
            self.root = ArticleSectionNode("#root", "")

    return Article


def check_section_insert(old_ref, new_ref):
    old_article = _article_class(old_ref)()
    new_article = _article_class(new_ref)()

    unique_outline = {
        "Introduction": {
            "content": "intro",
            "subsections": {
                "Background": {"content": "bg", "subsections": {}},
                "Motivation": {
                    "content": "mot",
                    "subsections": {"History": {"content": "h", "subsections": {}}},
                },
            },
        },
        "Conclusion": {"content": "concl", "subsections": {}},
    }
    old_article.insert_or_create_section(copy.deepcopy(unique_outline))
    new_article.insert_or_create_section(copy.deepcopy(unique_outline))
    unique_same = old_article.root.dump() == new_article.root.dump()

    # Two branches share the parent name "X". The old recursion looked the name
    # up from the root again, so DetailB landed under A -> X.
    duplicate_outline = {
        "A": {
            "content": "a",
            "subsections": {
                "X": {
                    "content": "x1",
                    "subsections": {"DetailA": {"content": "d1", "subsections": {}}},
                }
            },
        },
        "B": {
            "content": "b",
            "subsections": {
                "X": {
                    "content": "x2",
                    "subsections": {"DetailB": {"content": "d2", "subsections": {}}},
                }
            },
        },
    }
    old_dup_article = _article_class(old_ref)()
    new_dup_article = _article_class(new_ref)()
    old_dup_article.insert_or_create_section(copy.deepcopy(duplicate_outline))
    new_dup_article.insert_or_create_section(copy.deepcopy(duplicate_outline))
    duplicate_same = old_dup_article.root.dump() == new_dup_article.root.dump()

    detail = ""
    if not duplicate_same:
        old_branch = old_dup_article.root.dump()[2][0]
        new_branch = new_dup_article.root.dump()[2][0]
        detail = (
            f"old A/X children={[c[0] for c in old_branch[2][0][2]]}, "
            f"new A/X children={[c[0] for c in new_branch[2][0][2]]}"
        )

    return [
        Result("E6 section insert, unique names", unique_same),
        Result(
            "E6 section insert, duplicate parent name",
            duplicate_same,
            expected_same=False,
            detail=detail,
        ),
    ]


# --------------------------------------------------------- E10 domain blacklist
def _assigned_names(node):
    names = set()
    for target in getattr(node, "targets", []):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def check_domain_filter(old_ref, new_ref):
    select_names = {
        "GENERALLY_UNRELIABLE",
        "DEPRECATED",
        "BLACKLISTED",
        "_COMBINED_UNRELIABLE_SOURCES",
        "_BLOCKED_HOSTS",
        "_BLOCKED_LABELS",
    }
    select_functions = {"is_valid_wikipedia_source", "_split_blocklist"}
    results = []
    for label, ref in (("old", old_ref), ("new", new_ref)):
        source = source_at(ref, RETRIEVER)
        tree = ast.parse(source)
        namespace = {"urlparse": urlparse}
        for node in tree.body:
            is_func = (
                isinstance(node, ast.FunctionDef) and node.name in select_functions
            )
            is_set = isinstance(node, ast.Assign) and bool(
                _assigned_names(node) & select_names
            )
            if is_func or is_set:
                exec(
                    compile(ast.Module(body=[node], type_ignores=[]), "<ab>", "exec"),
                    namespace,
                )
        results.append(namespace["is_valid_wikipedia_source"])

    probes = [
        "https://example.com/article",
        "https://www.amazon.com/dp/x",
        "https://amazon.co.uk/dp/x",
        "https://notamazon.com/dp/x",
        "https://arxiv.org/abs/2402.14207",
        "https://www.bestgore.com/x",
        "https://en.wikipedia.org/wiki/Storm",
        "https://sports.yahoo.com/news",
        "https://artsy.net/artwork",
        "https://thehistoryblog.com/post",
    ]
    old_valid = [url for url in probes if results[0](url)]
    new_valid = [url for url in probes if results[1](url)]
    return Result(
        "E10 domain blacklist (currently unused code)",
        old_valid == new_valid,
        expected_same=False,
        detail=(
            f"allowed old={len(old_valid)}/{len(probes)}, "
            f"new={len(new_valid)}/{len(probes)}"
        ),
    )


# ------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old", default="HEAD", help="baseline git ref (default: HEAD)"
    )
    parser.add_argument(
        "--new",
        default=WORKTREE,
        help=f"git ref to compare, or '{WORKTREE}' (default)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when an equivalence check reports a difference",
    )
    args = parser.parse_args()

    print(f"A/B comparison: old={args.old}  new={args.new}\n")

    results = []
    results.append(check_citation(args.old, args.new))
    results.extend(check_encoder(args.old, args.new))
    results.extend(check_retrieval(args.old, args.new))
    results.append(check_content_union())
    results.extend(check_section_insert(args.old, args.new))
    results.append(check_domain_filter(args.old, args.new))

    for result in results:
        print(result.line())

    unexpected = [result for result in results if not result.ok]
    print()
    print(
        f"{len(results)} checks, "
        f"{sum(1 for r in results if r.same)} matched, "
        f"{len(unexpected)} unexpected"
    )
    if unexpected:
        print("Unexpected differences:")
        for result in unexpected:
            print(f"  - {result.name}")

    if args.strict and unexpected:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
