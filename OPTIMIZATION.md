# STORM 최적화 노트

## 배경

Rust 재작성 대신 Python 코드 안에서 병목을 제거한다. 판단 근거는 다음과 같다.

- STORM 한 편 생성에 LLM 호출이 약 50~150회 발생하고, 벽시계 시간의 대부분은 LLM/검색 API 대기다.
- Python이 CPU를 잡는 구간(임베딩 준비, 웹 추출, 트리 순회)은 전체의 5~10% 수준이다.
- 따라서 언어 교체보다 **호출 수 줄이기, 반복 계산 제거, I/O 병렬화**가 투자 대비 효과가 크다.

비목표: dspy 교체, 파이프라인 구조 변경, 프로바이더 추상화 재작성.

## 측정 방법

`LoggingWrapper`가 이미 이벤트별 소요 시간을 기록한다(`logging_wrapper.py:197`).
기사 한 편을 실행한 뒤 덤프하면 LLM / 검색 / 임베딩 / 파싱 비중을 확인할 수 있다.

```python
# 예시: 파이프라인 종료 후
print(engine.logging_wrapper.dump_logging_and_reset())
```

이 비중에서 LLM이 80% 이상이면 이 문서의 항목들은 "체감 지연"보다 "비용/안정성" 개선으로 봐야 한다.

## 우선순위 요약

| ID | 위치 | 문제 | 기대 효과 | 상태 |
|----|------|------|-----------|------|
| E1 | `knowledge_storm/encoder.py:112-178` | 텍스트 1개당 임베딩 요청 1회, O(n²) 정렬, 실패 시 결과 누락 | 호출 수/지연 대폭 감소 | 완료 |
| E2 | `knowledge_storm/storm_wiki/modules/storm_dataclass.py:109-145` | 매 실행 SentenceTransformer 재로드, 중복 스니펫 인코딩, 전체 argsort, 쿼리마다 코사인 재계산 | 실행당 수 초 + 검색 비용 감소 | 완료 |
| E3 | `knowledge_storm/collaborative_storm/modules/co_storm_agents.py:190-283` | 매 턴 전체 cited 스니펫 재인코딩, 워밍업 인코딩 결과 폐기, `hash(info)` 재계산 | O(N²) → O(N) | 완료 |
| E4 | `knowledge_storm/utils.py:684-704` | 다운로드는 병렬, trafilatura 추출은 직렬 | 페이지 수에 비례한 지연 감소 | 완료 |
| E5 | `knowledge_storm/collaborative_storm/modules/article_generation.py:39-90` | 노드마다 서브트리 전체 합집합 계산 | O(N²) → O(N) | 완료 |
| E6 | `knowledge_storm/storm_wiki/modules/storm_dataclass.py:209-247` | 섹션마다 root부터 DFS 재탐색 | O(N²) → O(N), 오매칭 버그 제거 | 완료 |
| E7 | `knowledge_storm/interface.py:70-91` | `__hash__` 중복 정의, `__eq__`와 해시 계약 불일치 | dict/set 오동작 제거 | 완료 |
| E8 | `knowledge_storm/rm.py` | SerperRM 공유 상태, 모든 HTTP 호출에 timeout 부재 | 병렬 결과 섞임/행(hang) 제거 | 완료 |
| E9 | `knowledge_storm/logging_wrapper.py:55-190` | `event_stack` 스레드 비안전, 예외 시 이벤트 미종료, 파이프라인 예외 삼킴 | 병렬 로그 정확성 | 완료 |
| E10 | `knowledge_storm/storm_wiki/modules/retriever.py:225-262` | 호출마다 ~200개 도메인 집합 재생성, 부분 문자열 매칭 오탐 | 호출당 상수 비용 제거, 오탐 제거 | 완료 |
| E11 | `knowledge_storm/utils.py:540-550` | 인용 번호마다 `str.replace` 2회 | O(k·n) → O(n) | 완료 |
| E12 | `knowledge_storm/collaborative_storm/modules/information_insertion_module.py:296-306` | 순차 삽입 루프에서 intent마다 KB 구조 재계산 | O(k·N) → O(N) | 완료 |

## 상세

### E1. Encoder 배치화

현재 `_get_text_embeddings`는 텍스트마다 `ThreadPoolExecutor`로 요청을 하나씩 보낸다.
OpenAI 임베딩 API는 요청당 수천 개 입력을 받으므로 배치로 묶어야 한다.

추가 문제:

- `embeddings.sort(key=lambda x: texts.index(x[0]))`가 O(n²)이고, 중복 텍스트가 있으면 순서가 깨진다.
- 배치 실패를 `print` 후 무시해 반환 배열 길이가 입력보다 짧아질 수 있다.

해결:

- `batch_size`(기본 128) 단위로 요청하고, 배치들을 `max_workers`개 스레드로 병렬 전송.
- `dict.fromkeys`로 중복 텍스트를 제거한 뒤 결과를 원래 순서로 복원.
- 실패는 `RuntimeError`로 올려 조용한 데이터 손실을 막는다.
- 토큰 집계는 메인 스레드에서 합산.

### E2. 정보 테이블 검색 경로

- `SentenceTransformer`를 모듈 레벨에서 1회만 로드한다(Streamlit 재실행 포함).
- 스니펫을 유니크하게 만든 뒤 인코딩하고, 원래 인덱스로 복원한다.
- `normalize_embeddings=True`로 저장해 두고 쿼리도 정규화해 내적만으로 코사인 유사도를 계산한다.
- top-k는 `np.argpartition`으로 뽑고 k개만 정렬한다.
- 반환용 `Information`은 `deepcopy` 대신 얕은 복사 + 새 `snippets` 리스트로 만든다.

### E3. Co-STORM moderator 임베딩 캐시

- `knowledge_base.info_uuid_to_info_dict`의 키가 이미 `hash(info)`이므로 cited 집합을 위해 해시를 다시 계산하지 않는다.
- 스니펫 문자열 → 임베딩 캐시를 moderator 인스턴스에 유지한다. 새 스니펫만 인코딩한다.
- `_get_sorted_unused_snippets`의 워밍업 인코딩 결과를 버리지 않고 같은 턴의 계산에 재사용한다.
- cited 정보가 없을 때 코사인 유사도 계산을 건너뛴다.

### E4. 웹 페이지 추출 병렬화

- HTML 다운로드(이미 병렬) 후 `trafilatura.extract`를 스레드 풀로 병렬 실행한다.
- 추출 실패는 해당 URL만 건너뛰고 나머지는 유지한다.
- TLS 검증(`verify=False`)은 기존 동작을 유지한다. 별도 보안 과제로 둔다.

### E5. 지식베이스 content 합집합 1회 계산

- `ArticleGenerationModule.forward`에서 후위 순회 1회로 노드별 합집합을 계산한다.
- `gen_section`에 선택적 `all_citation_index` 인자를 추가해 재계산을 막는다.

### E6. 섹션 삽입 시 노드 직접 전달

- `insert_or_create_section`에 `parent_section` 인자를 추가한다.
- 재귀 호출에서 `parent_section_name` 대신 현재 노드를 넘겨 root DFS를 제거한다.
- 동일 섹션명이 트리 다른 곳에 있을 때의 오매칭도 함께 해결된다.

### E7. `Information` 해시 계약

- 중복 정의된 첫 번째 `__hash__`를 제거한다.
- `__eq__`는 `set(snippets)`를 비교하므로 해시도 `tuple(sorted(set(snippets)))`를 사용한다.
- 스니펫 중복 시 `__eq__`는 True인데 해시가 달라지는 문제를 막는다.

### E8. 검색기 스레드 안전성과 타임아웃

- `SerperRM.forward`의 `self.results`/`self.result`를 지역 변수로 바꾼다.
- `query_params`를 복사해 인스턴스 설정을 변이하지 않는다.
- 모든 `requests` 호출에 `timeout=DEFAULT_HTTP_TIMEOUT`(10초)을 적용한다.

### E9. 로깅 스레드 안전성

- `event_stack`을 `threading.local()` 스택으로 바꿔 병렬 expert/section 이벤트가 섞이지 않게 한다.
- `logging_dict` 갱신은 `Lock`으로 보호한다.
- `log_event`에 `try/finally`를 넣어 예외 시에도 스택을 정리한다.
- `log_pipeline_stage`가 예외를 삼키지 않고 다시 올리도록 한다.
- 부수 효과: 병렬 스레드의 이벤트는 파이프라인 스테이지 직속 top-level 이벤트로 기록된다. 기존에는 메인 스레드 스택에 섞여 start/end가 뒤바뀌었다.

### E10. 신뢰도 도메인 집합

- `GENERALLY_UNRELIABLE | DEPRECATED | BLACKLISTED`를 모듈 로드 시 한 번만 `frozenset`으로 만든다.
- 도메인 비교를 소문자로 정규화한다. 기존 구현은 `"Amazon"` 같은 대문자 항목이 소문자 호스트와 매칭되지 않아 블랙리스트가 조용히 동작하지 않았다.
- 매칭을 부분 문자열에서 **호스트 경계**로 바꾼다.
  - 목록은 위키백과 문서 제목을 옮긴 것이라 206개 중 107개가 언더스코어를 포함한다(`Ad_Fontes_Media`). 호스트명에는 언더스코어가 없으므로 이런 항목은 매칭 대상에서 제외한다.
  - 점이 있는 항목(`bestgore.com`)은 호스트 동일/서브도메인으로 비교한다.
  - 점이 없는 항목(`amazon`, `medium`)은 호스트 라벨 정확 일치로 비교한다.
  - `parsed_url.hostname`을 쓰므로 포트/사용자정보가 섞여도 호스트만 본다.
- 이 변경으로 `notamazon.com`, `sports.yahoo.com`, `artsy.net`, `thehistoryblog.com` 같은 오탐이 사라졌다. 기존 구현은 `"RT"`가 소문자화되며 `sport`에 매칭되는 식이었다.

### E11. 인용 인덱스 갱신

- 인용 번호마다 `replace` 2회 하던 것을 정규식 1회 패스로 바꾼다.
- 단일 패스라 `1→2`, `2→3` 같은 연쇄 치환 위험도 없어진다.

## 적용 결과 (2026-11-15)

| ID | 변경 파일 | 검증 |
|----|-----------|------|
| E1 | `encoder.py` | 배치/순서/중복 제거/실패 전파 단위 테스트 통과 |
| E2 | `storm_wiki/modules/storm_dataclass.py` | 컴파일 + 정적 검토 |
| E3 | `collaborative_storm/modules/co_storm_agents.py` | 컴파일 + 정적 검토 |
| E4 | `utils.py` | 컴파일 + 정적 검토 |
| E5 | `collaborative_storm/modules/article_generation.py` | 컴파일 + 정적 검토 |
| E6 | `storm_wiki/modules/storm_dataclass.py` | 컴파일 + 정적 검토 |
| E7 | `interface.py` | 해시/동등성 계약 단위 테스트 통과 |
| E8 | `rm.py` | 컴파일 + 정적 검토 |
| E9 | `logging_wrapper.py` | 컴파일 + 정적 검토 |
| E10 | `storm_wiki/modules/retriever.py` | 도메인 필터 단위 테스트 통과 |
| E11 | `utils.py` | 인용 인덱스 단위 테스트 통과 |
| E12 | `collaborative_storm/modules/information_insertion_module.py` | 컴파일 + 정적 검토 |

검증 명령:

```powershell
python -m compileall -q knowledge_storm
python -m black --check knowledge_storm
```

참고: `sentence-transformers`/`dspy`가 설치되지 않은 환경이므로 E2/E3/E5/E6/E9는
실제 파이프라인 실행 대신 컴파일과 코드 검토로 확인했다. 실제 실행 시 다음을 확인한다.

- 섹션 검색 결과의 URL/스니펫 순서가 기존과 동일한지 (동점 처리 포함)
- Co-STORM moderator가 매 턴 동일한 스니펫 순위를 내는지
- 병렬 expert 실행 후 `dump_logging_and_reset()`의 이벤트 트리가 섞이지 않는지

### E12. 순차 삽입 시 구조 임베딩 재계산 제거

- `allow_create_new_node=True` 분기는 배치 시작 전에 구조 임베딩을 한 번 계산한다.
- 실제 노드 생성은 배치가 끝난 뒤 `insert_info_to_kb`에서 일어나므로 루프 안 재계산은 불필요하다.

## 후속 과제 (이번 범위 제외)

- `torch`/`transformers` 런타임 제거: `sentence-transformers`를 `fastembed`(ONNX) 또는 API 임베딩으로 교체. 배포 크기/콜드스타트 개선.
- `LM.history` 무제한 증가: 파이프라인 스테이지가 긴 Co-STORM 세션에서 메모리 증가. `deque(maxlen=...)` 검토.
- `KnowledgeBase.get_knowledge_base_structure_embedding`: 구조 변경 시에만 outline 문자열을 재생성하도록 dirty flag 도입.
- `WebPageHelper.httpx.Client(verify=False)`: TLS 검증 기본값 복원 검토.
- `httpx.Client` 수명 관리(`close()`).

## 시점별 A/B 비교 (`tests/ab_compare.py`)

최적화가 동작을 바꿨는지 확인하는 하네스다. git ref 두 개(또는 ref와 작업 트리)를
받아, 각 시점의 **실제 함수를 AST로 추출해 같은 입력에 실행**하고 결과를 비교한다.
네트워크/dspy/sentence-transformers 없이 numpy만 있으면 돌아간다.

```powershell
python tests/ab_compare.py                          # 직전 커밋 vs 작업 트리
python tests/ab_compare.py --old v1.1.0 --new HEAD  # 릴리스 간 비교
python tests/ab_compare.py --old HEAD~5 --new HEAD --strict
```

출력의 상태 값:

| 상태 | 의미 |
|------|------|
| `SAME` | 두 시점의 결과가 같다 |
| `DIFF (known)` | 의도된 동작 변경. 아래 "알려진 차이" 참고 |
| `DIFF (unexpected)` | 회귀. `--strict`이면 종료 코드 1 |
| `SAME (divergence gone)` | 예전에 있던 의도된 차이가 사라짐(같은 ref끼리 비교할 때 등) |

`--strict`은 **동등성이 기대되는 검사**(E11, E1 일반 텍스트, E2 순위, E5, E6 유일 이름)만
실패로 취급한다. 의도된 차이는 정보성으로만 표시된다.

검사 항목:

- E11: 인용 치환 3,000 랜덤 케이스 문자열 비교
- E1: 200개 고유 텍스트 임베딩 배열 비교 + 중복 텍스트 순서 보존
- E2: 300 스니펫 × 5 쿼리 × k=1,3,5,10 검색 결과 비교 + 동점 선택
- E5: 랜덤 트리 200개에서 노드별 content 합집합 비교
- E6: 섹션 삽입 트리 직렬화 비교 (유일 이름 / 부모 이름 중복)
- E10: 블랙리스트 필터 판정 비교 (정보성)

### 알려진 차이 (의도된 변경)

1. **E1 중복 텍스트 순서**: 기존 코드는 `texts.index()` 기반 정렬 때문에 중복 입력에서
   벡터 순서가 어긋났다. 새 코드는 입력 순서를 보존한다.
2. **E2 동점 선택**: 점수가 완전히 같을 때 기존은 인덱스가 큰 쪽, 새 코드는
   `argpartition`이 고른 쪽이 선택된다. 두 구현 모두 같은 입력에 대해 재현 가능하지만,
   `argpartition`의 동점 처리는 numpy 구현에 의존하므로 numpy 버전이 바뀌면 달라질 수 있다.
3. **E6 부모 이름 중복**: 기존에는 `B -> X`의 하위 섹션이 `A -> X` 밑으로 들어갔다.
   새 코드는 실제 부모 노드를 넘기므로 트리가 정확하다.
4. **E10 도메인 블랙리스트**: 206개 항목 중 201개가 대문자를 포함해 기존 필터는 사실상
   동작하지 않았다. 이제 소문자로 정규화하고 호스트 라벨 단위로 비교한다. 다만
   `is_valid_wikipedia_source`는 현재 호출부가 없어 산출물에는 영향이 없다.

## 검증

```powershell
python tests/test_optimizations.py
python tests/ab_compare.py --strict
python -m compileall -q knowledge_storm
python -m black --check knowledge_storm
```

`tests/test_optimizations.py`는 외부 API와 dspy를 스텁으로 대체하고 numpy만 요구한다.
커버리지: E1(배치/순서/중복/실패), E7(해시 계약), E10(도메인 필터), E11(인용 치환).

나머지 항목은 단위 테스트가 없으므로, 변경 후 최소 1회 실제 파이프라인 실행으로 다음을 확인한다.

- 동일 입력에 대해 생성 결과 개수/순서가 유지되는지
- 임베딩 반환 shape이 호출부 기대와 일치하는지
- 병렬 실행 시 로그 이벤트 트리가 깨지지 않는지
