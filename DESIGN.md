# NLP Daily Digest — Design Document

## 1. 목적

매일 오전 9시(KST) HuggingFace에서 NLP 관련 논문을 **일간 3건 · 주간 2건 · 월간 1건** 구조로 수집·요약해 Discord 채널로 전송한다. 독자가 링크를 클릭하지 않아도 논문의 핵심 기여와 방법론을 파악할 수 있도록 영어 1문장 + 한국어 1문장 + 상세 요약(3~4문장)을 제공한다.

---

## 2. 전체 흐름

```
GitHub Actions (매일 00:00 UTC = 09:00 KST)
│
├─ fetch_hf_daily()          ← 어제 HF papers (일간 핫 논문)
│    └─ _fetch_hf_day(yesterday)
│
├─ fetch_hf_range(days 2-8)  ← 주간 HF papers (7 API calls)
│
├─ fetch_hf_range(days 9-30) ← 월간 HF papers (step=3, ~7 API calls)
│
├─ fetch_arxiv_papers()      ← 폴백 전용 (HF 윈도우 부족 시)
│
├─ merge_papers()
│    ├─ 일간 상위 3건 (업보트 순)
│    ├─ 주간 상위 2건 (업보트 순, 일간과 중복 제거)
│    ├─ 월간 상위 1건 (업보트 순, 위와 중복 제거)
│    └─ 부족분 ArXiv로 보충
│
├─ summarize_papers()
│    └─ 논문별 Gemini API 호출 → EN / KO / DETAIL 파싱
│
└─ send_to_discord()
     └─ 헤더 1개 + 논문별 메시지 1개 (총 7개)
          각 메시지에 `📅 일간` / `📆 주간` / `🗓️ 월간` 라벨 표시
```

---

## 3. 시간 윈도우 설계

### 왜 전날(yesterday) 기준인가?

스크립트는 00:00 UTC에 실행된다. HF daily papers는 UTC 기준으로 매일 리셋되므로, 실행 시점에 당일 논문은 방금 올라와 업보트가 거의 없다(1~9개). 전날 논문은 하루 치 업보트가 누적되어 실제 화제도를 반영한다.

### 각 윈도우 범위

| 윈도우 | 날짜 범위 | API 호출 수 | 선발 수 |
|---|---|---|---|
| 일간 | `today - 1` (어제) | 1 | 3건 |
| 주간 | `today - 8` ~ `today - 2` | 7 | 2건 |
| 월간 | `today - 30` ~ `today - 9` (3일 간격) | ~7 | 1건 |

월간은 22일 전 기간을 3일 간격으로 샘플링하여 ~7회 호출로 처리한다. 정확도와 호출 수의 균형점이다.

---

## 4. 데이터 소스

### 4-1. HuggingFace Daily Papers (주 소스)

| 항목 | 내용 |
|---|---|
| 엔드포인트 | `https://huggingface.co/api/daily_papers?date=YYYY-MM-DD` |
| 특징 | HF 커뮤니티가 직접 큐레이션한 논문. 업보트 수가 화제도 신호. |
| 폴백 | API 실패 시 `huggingface.co/papers?date=YYYY-MM-DD` HTML 스크래핑 |
| score | `paper["upvotes"] + item["numComments"]` |

`_fetch_hf_day(date_str)` → `(papers, upvotes_map)` 형태로 반환하며, `upvotes_map`은 ArXiv 교차참조에도 사용된다.

### 4-2. ArXiv Export API (폴백 전용)

| 항목 | 내용 |
|---|---|
| 엔드포인트 | `http://export.arxiv.org/api/query` |
| 카테고리 | `cat:cs.CL OR cat:cs.AI` |
| 정렬 | 제출일 내림차순 (최신 40건) |
| 역할 | HF 윈도우가 부족할 때만 출동 |

실제로 일간·주간·월간 3개 윈도우가 모두 동작하면 ArXiv가 선발되는 경우는 드물다. HF API 장애 등 예외 상황 대비용이다.

ArXiv 논문의 score: HF upvotes_map에 동일 arxiv_id가 있으면 그 값, 없으면 0(최신순 유지).

#### 소스 변천 이력

| 시도 | 이유 | 결과 |
|---|---|---|
| Reddit JSON API | 업보트 신호 | 403 Blocked |
| Papers With Code API | NLP task 필터 | HTML 반환(차단) |
| ArXiv Export API | 안정적·무인증·cs.CL 전용 | 폴백으로 채택 |

### 4-3. Reddit (미적용 · 구현 방안)

Reddit r/MachineLearning, r/LanguageTechnology는 NLP 논문 화제도에 좋은 신호원이지만, 비인증 JSON API(`/hot.json`)는 403으로 차단된다.

**구현 가능한 방법: Reddit OAuth 앱 인증**

```
1. https://www.reddit.com/prefs/apps 에서 Script 타입 앱 생성
2. client_id, client_secret 발급
3. POST https://www.reddit.com/api/v1/access_token 으로 Bearer 토큰 발급
4. GET https://oauth.reddit.com/r/MachineLearning/hot.json
   (Authorization: Bearer {token}, User-Agent: nlp-digest/1.0)
```

필요한 추가 Secret: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`

구현 시 Reddit upvotes를 월간·주간 논문 score 보정에 활용하거나, 별도 "Reddit 화제 논문" 슬롯으로 구성하는 방안이 있다.

---

## 5. NLP 키워드 필터

제목(+ HF tags)에 아래 키워드가 포함된 논문만 통과. 대소문자 무시.

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

ArXiv 40건 기준 실측치: 약 50~55% 통과. HF 논문은 tags 필드도 함께 검사한다.

---

## 6. 병합 및 선발 로직

```
일간 pool (업보트 순) → 상위 3건
주간 pool (업보트 순) → 상위 2건 (일간 중복 제거)
월간 pool (업보트 순) → 상위 1건 (일간+주간 중복 제거)
ArXiv pool            → 남은 슬롯 보충 (위 모두 중복 제거)
```

중복 제거 기준: `arxiv_id` (없으면 title). 같은 논문이 일간·주간에 모두 있으면 일간 슬롯에서만 사용된다.

---

## 7. Gemini 요약 설계

### 프롬프트 구조

```
You are an expert NLP researcher writing a daily digest for Korean ML practitioners.

Paper title: {title}
Abstract: {abstract[:800]}

Write exactly these three sections and nothing else:
EN: [One English sentence capturing the key contribution]
KO: [같은 내용을 한국어 한 문장으로 작성]
DETAIL: [핵심 방법론, 실험 결과, 의의를 포함하여 3~4문장으로 한국어로 상세히 설명]
```

abstract가 없으면 ArXiv HTML에서 추가 수집 후 삽입한다.

### 응답 파싱

줄 단위로 파싱. `DETAIL`은 여러 줄일 수 있으므로 `current == "detail"` 상태 동안 계속 누적.

### 모델 선택

기본값 `gemini-2.5-flash`. GitHub Variables `GEMINI_MODEL`로 재정의 가능.

### Fallback

Gemini 호출 실패 시: `en = ko = title`, `detail = abstract[:300]`

---

## 8. Discord 전송 설계

논문 1건 = 메시지 1개. 메시지 구조:

```
**{i}. {title}** `{window_label}`
🇺🇸 {en}
🇰🇷 {ko}

📋 **상세 요약**
{detail}

[논문 ArXiv](<arxiv_url>) | [{source}](<source_url>)
```

윈도우 라벨:

| window | 라벨 |
|---|---|
| daily | `📅 일간` |
| weekly | `📆 주간` |
| monthly | `🗓️ 월간` |
| arxiv | `🔬 ArXiv` |

- 헤더 1개 + 논문 6개 = 총 7개 메시지
- 메시지 사이 0.5초 대기 (Discord rate limit 방지)
- URL은 `<url>` 형식으로 감싸 자동 프리뷰 억제

---

## 9. 설정

### GitHub Secrets (필수)

| 이름 | 설명 |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio에서 발급 |
| `DISCORD_WEBHOOK_URL` | Discord 채널 웹훅 URL |

### GitHub Secrets (Reddit 추가 시)

| 이름 | 설명 |
|---|---|
| `REDDIT_CLIENT_ID` | Reddit 앱 client_id |
| `REDDIT_CLIENT_SECRET` | Reddit 앱 client_secret |

### GitHub Variables (선택)

| 이름 | 기본값 | 설명 |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | 사용할 Gemini 모델 ID |

### 코드 내 상수

| 상수 | 값 | 설명 |
|---|---|---|
| `DAILY_CAP` | 3 | 일간 논문 선발 수 |
| `WEEKLY_CAP` | 2 | 주간 논문 선발 수 |
| `MONTHLY_CAP` | 1 | 월간 논문 선발 수 |
| `TOP_N` | 6 | 총 논문 수 (3+2+1) |
| `ARXIV_CATEGORIES` | `cat:cs.CL OR cat:cs.AI` | ArXiv 검색 카테고리 |

---

## 10. GitHub Actions 스케줄

```yaml
on:
  schedule:
    - cron: '0 0 * * *'   # 매일 00:00 UTC = 09:00 KST
  workflow_dispatch:        # 수동 실행 허용
```

**주의:** Actions 탭에서 **Re-run**을 누르면 최신 코드가 아닌 원래 트리거 시점의 커밋으로 실행된다. 최신 코드 테스트는 반드시 **Run workflow** 버튼으로 새 실행을 트리거해야 한다.

---

## 11. 알려진 한계 및 향후 개선 아이디어

| 항목 | 현황 | 개선 방향 |
|---|---|---|
| 월간 샘플링 누락 | 3일 간격으로 일부 날짜 건너뜀 | 전체 쿼리 또는 별도 캐시 |
| Reddit 차단 | JSON API 403 | OAuth 앱 인증으로 해결 가능 |
| NLP 필터 false positive | 키워드 기반 간혹 비NLP 논문 포함 | 제목 임베딩 유사도 기반 필터 |
| abstract 수집 실패 | Gemini가 제목만으로 요약 | Semantic Scholar API로 보완 |
| 논문 수 고정 | 상수로 하드코딩 | GitHub Variables로 노출 |
