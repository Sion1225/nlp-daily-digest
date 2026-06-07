# NLP Daily Digest

매일 오전 9시(KST) GitHub Actions가 HuggingFace Papers + Reddit에서 NLP 핫 페이퍼 3~5개를 골라 Gemini로 요약한 뒤 Discord로 전송합니다.

## 흐름

```
GitHub Actions (매일 09:00 KST)
  └─ scripts/digest.py
       ├─ HuggingFace daily papers API  ──┐
       ├─ Reddit (r/MachineLearning 등)  ──┼─ NLP 키워드 필터 → 상위 5건
       │                                  ┘
       ├─ ArXiv에서 초록 보완 (필요 시)
       ├─ Gemini API → EN 한 문장 + KO 한 문장
       └─ Discord 웹훅으로 전송
```

## 설정

### 1. GitHub Secrets 등록

리포지터리 → **Settings → Secrets and variables → Actions** 에서 아래 두 값을 등록합니다.

| Secret 이름 | 값 |
|---|---|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급 |
| `DISCORD_WEBHOOK_URL` | Discord 채널 설정 → 연동 → 웹훅에서 복사 |

### 2. (선택) 모델 변경

**Settings → Variables → Actions** 에서 `GEMINI_MODEL` 변수를 추가하면 모델을 바꿀 수 있습니다 (기본값: `gemini-2.0-flash`).

### 3. 수동 실행

Actions 탭 → **NLP Daily Digest** → **Run workflow** 버튼으로 즉시 테스트할 수 있습니다.

## 로컬 실행

```bash
pip install -r requirements.txt
GEMINI_API_KEY=your_key DISCORD_WEBHOOK_URL=your_url python scripts/digest.py
```
