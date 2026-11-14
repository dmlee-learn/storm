"""Regression tests for the optimization changes.

Run with `python tests/test_optimizations.py`. Only numpy is required; the
embedding provider and dspy are stubbed so no network access is needed.
"""

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
        data = [
            {"embedding": [float(len(text)), 1.0], "index": idx}
            for idx, text in enumerate(texts)
        ]
        return FakeEmbeddingResponse(data, {"total_tokens": 1})

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
    return fake


def load_encoder():
    spec = importlib.util.spec_from_file_location(
        "encoder_under_test", ROOT / "knowledge_storm" / "encoder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encoder_batches_dedupes_and_preserves_order():
    fake = install_litellm_stub()
    calls = []
    original_embedding = fake.embedding

    def recording_embedding(model, input, caching=True, **kwargs):
        calls.append(list(input) if isinstance(input, (list, tuple)) else [input])
        return original_embedding(model, input, caching=caching, **kwargs)

    fake.embedding = recording_embedding
    encoder = load_encoder().Encoder(encoder_type="openai", api_key="test")

    result = encoder.encode(["a", "bb", "ccc", "a"], batch_size=2)
    assert result.shape == (4, 2)
    assert list(result[:, 0]) == [1.0, 2.0, 3.0, 1.0]

    embedded_texts = [text for call in calls for text in call]
    assert sorted(embedded_texts) == ["a", "bb", "ccc"]

    single = encoder.encode("abcd")
    assert single.shape == (2,)
    assert single[0] == 4.0
    print("encoder batching/order/dedup OK")


def test_encoder_raises_on_short_response():
    fake = install_litellm_stub()
    fake.embedding = lambda *args, **kwargs: FakeEmbeddingResponse([], {})
    encoder = load_encoder().Encoder(encoder_type="openai", api_key="test")
    try:
        encoder.encode(["a", "b"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for a short embedding response")
    print("encoder failure handling OK")


def test_information_hash_matches_equality():
    pkg = types.ModuleType("ks_under_test")
    pkg.__path__ = [str(ROOT / "knowledge_storm")]
    sys.modules["ks_under_test"] = pkg

    utils_stub = types.ModuleType("ks_under_test.utils")
    utils_stub.ArticleTextProcessing = types.SimpleNamespace(
        remove_citations=lambda text: text
    )
    sys.modules["ks_under_test.utils"] = utils_stub

    dataclass_stub = types.ModuleType("ks_under_test.dataclass")
    dataclass_stub.KnowledgeBase = object
    dataclass_stub.ConversationTurn = object
    sys.modules["ks_under_test.dataclass"] = dataclass_stub

    dspy_stub = types.ModuleType("dspy")
    dspy_stub.Retrieve = object
    dspy_stub.dsp = types.SimpleNamespace(LM=object, HFModel=object)
    sys.modules["dspy"] = dspy_stub

    spec = importlib.util.spec_from_file_location(
        "ks_under_test.interface", ROOT / "knowledge_storm" / "interface.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first = module.Information("http://x", "d", ["a", "a"], "t", meta={"query": "q"})
    second = module.Information("http://x", "d", ["a"], "t", meta={"query": "q"})
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    print("Information hash/eq contract OK")


def load_utils_class():
    source = (ROOT / "knowledge_storm" / "utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ArticleTextProcessing"
    )
    class_node.body = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_citation_index"
    ]
    namespace = {"re": re}
    exec(
        compile(ast.Module(body=[class_node], type_ignores=[]), "<test>", "exec"),
        namespace,
    )
    return namespace["ArticleTextProcessing"].update_citation_index


def test_update_citation_index_single_pass():
    update = load_utils_class()
    assert update("[1] a [2] b [3]", {1: 2, 2: 3}) == "[2] a [3] b [3]"
    assert update("[9] stays", {1: 2}) == "[9] stays"
    assert update("no citations", {1: 2}) == "no citations"
    assert update("[1]", {}) == "[1]"
    assert update("[1]", {"1": 7}) == "[7]"
    print("update_citation_index OK")


def test_wikipedia_source_filter_is_case_insensitive():
    source = (
        ROOT / "knowledge_storm" / "storm_wiki" / "modules" / "retriever.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and (
            getattr(node, "name", None) == "is_valid_wikipedia_source"
            or any(
                getattr(target, "id", None)
                in {
                    "GENERALLY_UNRELIABLE",
                    "DEPRECATED",
                    "BLACKLISTED",
                    "_COMBINED_UNRELIABLE_SOURCES",
                }
                for target in getattr(node, "targets", [])
            )
        )
    ]
    urlparse = __import__("urllib.parse", fromlist=["urlparse"]).urlparse
    namespace = {"urlparse": urlparse}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<test>", "exec"), namespace
    )
    is_valid = namespace["is_valid_wikipedia_source"]
    assert is_valid("https://example.com/article")
    assert not is_valid("https://arxiv.org/abs/2402.14207")
    assert not is_valid("https://www.amazon.com/dp/x")
    print("is_valid_wikipedia_source OK")


if __name__ == "__main__":
    test_encoder_batches_dedupes_and_preserves_order()
    test_encoder_raises_on_short_response()
    test_information_hash_matches_equality()
    test_update_citation_index_single_pass()
    test_wikipedia_source_filter_is_case_insensitive()
    print("ALL TESTS PASSED")
