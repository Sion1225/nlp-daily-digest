# NLP Daily Digest — Design Document

## 1. 목적

매일 오전 9시(KST) HuggingFace와 ArXiv에서 NLP 관련 핫 논문 5건을 자동 수집·요약해 Discord 채널로 전송한다. 독자가 링크를 클릭하지 않아도 논문의 핵심 기여와 방법론을 파악할 수 있도록 영어 1문장 + 한국어 1문장 + 상세 요약(3~4문장)을 제공한다.

---

## 2. 전체 흐름

```
GitHub Actions (매일 00:00 UTC = 09:00 KST)
│
├─ fetch_hf_papers()
│    └─ HuggingFace API (daily_papers)
│         ├─ 성공 → JSON 파싱, upvotes 맵 생성
│         └─ 실패 → HTML 스크래핑 폴백
│
├─ fetch_arxiv_papers(hf_upvotes)
│    └─ ArXiv Export API (cat:cs.CL OR cat:cs.AI, 최신 40건)
│         └─ NLP 키워드 필터 → HF upvotes 교차참조 → score 부여
│
├─ merge_papers()
│    └─ HF 최대 3건 우선 + ArXiv로 나머지 2건 보충 (중복 제거)
│
├─ summarize_papers()
│    └─ 논문별 Gemini API 호출
│         ├─ abstract 없으면 ArXiv HTML에서 추가 수집
│         └─ EN 1문장 / KO 1문장 / DETAIL 3~4문장 파싱
│
└─ send_to_discord()
     └─ 헤더 메시지 1개 + 논문별 메시지 1개 (총 6개 메시지)
```

---

## 3. 데이터 소스 설계

### 3-1. HuggingFace Daily Papers

| 항목 | 내용 |
|---|---|
| 엔드포인트 | `https://huggingface.co/api/daily_papers` |
| 특징 | HF 커뮤니티가 직접 큐레이션한 당일 논문. 업보트 수가 화제도 신호. |
| 폴백 | API 실패 시 `huggingface.co/papers` HTML 스크래핑 |
| 반환 | `(papers: list[dict], hf_upvotes: dict[str, int])` |

`hf_upvotes`는 `arxiv_id → upvotes` 맵으로, NLP 필터 통과 여부와 무관하게 전체 논문을 저장한다. ArXiv 논문과 교차참조 시 사용된다.

**score 계산:** `paper["upvotes"] + item["numComments"]`

### 3-2. ArXiv Export API

| 항목 | 내용 |
|---|---|
| 엔드포인트 | `http://export.arxiv.org/api/query` |
| 카테고리 | `cat:cs.CL OR cat:cs.AI` |
| 정렬 | 제출일 내림차순 (최신 40건) |
| 형식 | Atom XML → `xml.etree.ElementTree` 파싱 |
| 역할 | HF에 아직 올라오지 않은 최신 NLP 논문 보완 |

**score 계산:** HF upvotes 맵에 동일 arxiv_id가 있으면 그 upvotes를 사용, 없으면 0(제출일 순 유지).

#### 소스 변천 이력

| 시도 | 이유 | 결과 |
|---|---|---|
| Reddit JSON API | 화제도 신호(upvotes) 활용 | 403 Blocked |
| Papers With Code API | NLP task별 필터 | HTML 반환(차단) |
| **ArXiv Export API** | 안정적, 무인증, cs.CL 전용 | **채택** |

---

## 4. NLP 키워드 필터

제목(+ HF tags)에 아래 키워드가 포함된 논문만 통과시킨다. 대소문자 무시.

```python
NLP_KEYWORDS = {
    # 핵심 모델 / 아키텍처
    "nlp", "natural language", "language model", "large language", "llm",
    "transformer", "bert", "gpt", "pretraining", "pre-training",
    "sparse attention", "vision-language", "vision language",
    "multimodal language", "multimodal model", "emergent language",
    # 태스크
    "text generation", "machine translation", "translation",
    "summarization", "question answering", "dialogue", "chatbot",
    "sentiment", "named entity", "information extraction",
    "text classification", "speech recognition", "speech synthesis",
    "reading comprehension", "language understanding",
    # 학습 패러다임
    "tokenization", "embedding", "retrieval augmented", "rag",
    "instruction tuning", "fine-tuning", "alignment", "rlhf",
    "prompt", "in-context", "few-shot", "zero-shot",
    "reasoning", "chain of thought",
    # 코퍼스 / 데이터
    "text corpus", "text corpora", "corpora", "parallel corpus",
    "language corpus", "linguistic",
}
```

**HF 논문**은 tags 필드도 함께 검사한다 (HF가 논문에 태그를 붙이기 때문).  
**ArXiv 논문**은 이미 cs.CL/cs.AI 카테고리라 제목만 검사해도 충분하다.

ArXiv 40건 기준 실측치: 약 50~55% 통과 (20~22건). 슬롯 2개를 채우기에 충분하다.

---

## 5. 병합 및 선발 로직

```
HF 논문 (upvotes 순) → 최대 3건 선택
ArXiv 논문 (HF upvotes > 0이면 우선, 아니면 최신 순) → 나머지 슬롯 채움
중복 제거: arxiv_id 기준 (없으면 title)
최종 5건
```

**HF를 3건으로 캡핑하는 이유:** HF daily papers가 25건 이상 반환되어 제한 없이 쓰면 ArXiv 논문이 전혀 포함되지 않는다. ArXiv를 섞는 이유는 HF 큐레이션에 포함되지 않은 최신 논문도 노출하기 위해서다.

---

## 6. Gemini 요약 설계

### 프롬프트 구조

```
You are an expert NLP researcher writing a daily digest for Korean ML practitioners.

Paper title: {title}
Abstract: {abstract[:800]}  ← abstract 없으면 ArXiv HTML에서 추가 수집

Write exactly these three sections and nothing else:
EN: [One English sentence capturing the key contribution]
KO: [같은 내용을 한국어 한 문장으로 작성]
DETAIL: [핵심 방법론, 실험 결과, 의의를 포함하여 3~4문장으로 한국어로 상세히 설명]
```

### 응답 파싱

줄 단위로 파싱하며 `EN:` / `KO:` / `DETAIL:` 헤더로 섹션을 구분한다.  
`DETAIL`은 여러 줄일 수 있으므로 `current == "detail"` 상태가 유지되는 동안 계속 누적한다.

### 모델 선택

기본값 `gemini-2.5-flash`. GitHub Actions Variable `GEMINI_MODEL`로 재정의 가능.  
Flash 계열을 선택한 이유: 5건 × 1회 요청 기준 latency와 비용이 Pro 대비 크게 낮고, 단문 요약 품질은 충분하다.

### Fallback

Gemini 호출 실패 시: `en = ko = title`, `detail = abstract[:300]`.

---

## 7. Discord 전송 설계

논문 1건 = 메시지 1개. 메시지 구조:

```
**{i}. {title}**
🇺🇸 {en}
🇰🇷 {ko}

📋 **상세 요약**
{detail}

[논문 ArXiv](<arxiv_url>) | [{source}](<source_url>)
```

- 헤더 메시지 1개 + 논문 메시지 5개 = 총 6개 전송
- 메시지당 2000자 초과 시 말줄임 처리
- 메시지 사이 0.5초 대기 (Discord rate limit 방지)
- URL은 `<url>` 형식으로 감싸 Discord 자동 프리뷰 억제

**링크 구성:**

| 소스 | 논문 링크 | 출처 링크 |
|---|---|---|
| HuggingFace | ArXiv abs 페이지 | `huggingface.co/papers/{arxiv_id}` |
| ArXiv cs.CL | ArXiv abs 페이지 | `arxiv.org/list/cs.CL/recent` |

---

## 8. 설정

### GitHub Secrets (필수)

| 이름 | 설명 |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio에서 발급 |
| `DISCORD_WEBHOOK_URL` | Discord 채널 웹훅 URL |

### GitHub Variables (선택)

| 이름 | 기본값 | 설명 |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | 사용할 Gemini 모델 ID |

### 코드 내 상수

| 상수 | 값 | 설명 |
|---|---|---|
| `TOP_N` | 5 | 최종 선발 논문 수 |
| `HF_CAP` | 3 | HF 논문 최대 선택 수 |
| `ARXIV_CATEGORIES` | `cat:cs.CL OR cat:cs.AI` | ArXiv 검색 카테고리 |

---

## 9. GitHub Actions 스케줄

```yaml
on:
  schedule:
    - cron: '0 0 * * *'   # 매일 00:00 UTC = 09:00 KST
  workflow_dispatch:        # 수동 실행 허용
```

**주의:** Actions 탭에서 **Re-run**을 누르면 최신 코드가 아닌 원래 트리거 시점의 커밋으로 실행된다. 최신 코드 테스트는 반드시 **Run workflow** 버튼으로 새 실행을 트리거해야 한다.

---

## 10. 알려진 한계 및 향후 개선 아이디어

| 항목 | 현황 | 개선 방향 |
|---|---|---|
| ArXiv 화제도 신호 | HF upvotes 교차참조 (없으면 최신순) | Semantic Scholar 인용수 추가 |
| NLP 필터 정밀도 | 키워드 기반 → 간혹 false positive | 제목 임베딩 유사도 기반 필터 |
| abstract 수집 실패 | Gemini가 제목만으로 요약 | Semantic Scholar API로 abstract 보완 |
| 단일 언어 | 한국어 + 영어 | 사용자 설정 언어 추가 |
| 논문 수 고정 | TOP_N=5 하드코딩 | 환경변수로 노출 |
