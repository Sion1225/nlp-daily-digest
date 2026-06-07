#!/usr/bin/env python3
"""PWC task slug 유효성 검사 + 랭킹 기준 미리보기."""

import requests

TASKS = [
    # ── 고전 NLP ──────────────────────────────
    "language-modelling",
    "text-generation",
    "machine-translation",
    "question-answering",
    "text-classification",
    "named-entity-recognition",
    "summarization",
    "sentiment-analysis",
    "relation-extraction",
    "coreference-resolution",
    "natural-language-inference",
    "reading-comprehension",
    "open-domain-question-answering",
    "semantic-textual-similarity",
    "fact-checking",
    # ── 대화 / 생성 ───────────────────────────
    "dialogue-generation",
    "code-generation",
    "data-to-text-generation",
    # ── 추론 / 지식 ───────────────────────────
    "commonsense-reasoning",
    "mathematical-reasoning",
    "knowledge-base-question-answering",
    # ── 모델 펀더멘탈 ─────────────────────────
    "language-model-pretraining",
    "few-shot-learning",
    "zero-shot-learning",
    "transfer-learning",
    "knowledge-distillation",
    "model-compression",
    "word-embeddings",
]

BASE = "https://paperswithcode.com/api/v1/papers/"
HEADERS = {"User-Agent": "nlp-digest-test/1.0"}

ok_tasks, fail_tasks = [], []

print(f"\n{'TASK':<40} {'STATUS':>7}  {'건수':>4}  {'GitHub★ TOP1':>10}  샘플 제목")
print("-" * 110)

for task in TASKS:
    try:
        r = requests.get(
            BASE, params={"ordering": "-published", "page_size": 5, "task": task},
            headers=HEADERS, timeout=15,
        )
        if not r.ok:
            print(f"{task:<40} {'['+str(r.status_code)+']':>7}  {'—':>4}")
            fail_tasks.append(task)
            continue

        results = r.json().get("results", [])
        count = r.json().get("count", 0)

        if not results:
            print(f"{task:<40} {'OK':>7}  {'0':>4}  {'(결과 없음)'}")
            fail_tasks.append(task)
            continue

        top = results[0]
        gh = top.get("github_link") or {}
        stars = gh.get("stars", 0) or 0
        title = (top.get("title") or "")[:50]
        print(f"{task:<40} {'OK':>7}  {count:>4}  {stars:>10}★  {title}")
        ok_tasks.append(task)

    except Exception as e:
        print(f"{task:<40} {'ERROR':>7}  {e}")
        fail_tasks.append(task)

print("\n" + "=" * 110)
print(f"유효 slug: {len(ok_tasks)}개  /  실패: {len(fail_tasks)}개")
if fail_tasks:
    print("제거 권장:", fail_tasks)
