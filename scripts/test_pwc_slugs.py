#!/usr/bin/env python3
"""ArXiv cs.CL API 동작 확인 및 샘플 출력."""

import re
import requests
import xml.etree.ElementTree as ET

NS = "{http://www.w3.org/2005/Atom}"
NLP_KEYWORDS = {
    "nlp", "natural language", "language model", "large language", "llm",
    "transformer", "bert", "gpt", "text generation", "machine translation",
    "summarization", "question answering", "dialogue", "chatbot", "sentiment",
    "named entity", "information extraction", "text classification",
    "speech recognition", "speech synthesis", "tokenization", "embedding",
    "retrieval augmented", "rag", "instruction tuning", "fine-tuning",
    "alignment", "rlhf", "prompt", "in-context learning", "few-shot",
    "zero-shot", "reasoning", "chain of thought", "multimodal language",
    "vision language", "language understanding", "reading comprehension",
}

def is_nlp_related(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in NLP_KEYWORDS)

print("=== ArXiv cs.CL API 테스트 ===\n")

resp = requests.get(
    "http://export.arxiv.org/api/query",
    params={
        "search_query": "cat:cs.CL OR cat:cs.AI",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": 40,
    },
    headers={"User-Agent": "nlp-digest/1.0"},
    timeout=30,
)
print(f"Status      : {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")

root = ET.fromstring(resp.content)
entries = root.findall(f"{NS}entry")
print(f"전체 항목   : {len(entries)}건\n")

passed, skipped = [], []
for entry in entries:
    title_el  = entry.find(f"{NS}title")
    id_el     = entry.find(f"{NS}id")
    if title_el is None or id_el is None:
        continue
    title    = title_el.text.strip().replace("\n", " ")
    arxiv_id = re.sub(r"v\d+$", "", id_el.text.strip().split("/abs/")[-1])
    if is_nlp_related(title):
        passed.append((arxiv_id, title))
    else:
        skipped.append(title)

print(f"NLP 필터 통과: {len(passed)}건 / 미통과: {len(skipped)}건\n")
print("─── 통과된 논문 ──────────────────────────────────────────")
for arxiv_id, title in passed:
    print(f"  [{arxiv_id}] {title[:80]}")

print("\n─── 미통과 논문 전체 ────────────────────────────────────")
for t in skipped:
    print(f"  {t[:90]}")
