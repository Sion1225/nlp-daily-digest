#!/usr/bin/env python3
"""NLP Daily Digest: HuggingFace + Reddit → Gemini 요약 → Discord 전송."""

import os
import re
import logging
import requests
from datetime import date
from bs4 import BeautifulSoup
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
TOP_N = 5

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

ARXIV_RE = re.compile(r"arxiv\.org/abs/([\d.]+v?\d*)")


def is_nlp_related(title: str, tags: list[str] | None = None) -> bool:
    text = title.lower()
    if tags:
        text += " " + " ".join(t.lower() for t in tags)
    return any(kw in text for kw in NLP_KEYWORDS)


# ──────────────────────────────────────────────
# 1. 데이터 수집
# ──────────────────────────────────────────────

def fetch_hf_papers() -> list[dict]:
    """HuggingFace 데일리 페이퍼를 수집합니다 (API → HTML 폴백)."""
    log.info("HuggingFace 페이퍼 수집 중...")
    papers: list[dict] = []

    # ── API 시도 ──────────────────────────────
    try:
        resp = requests.get(
            "https://huggingface.co/api/daily_papers",
            headers={"User-Agent": "nlp-digest/1.0"},
            timeout=15,
        )
        if resp.ok:
            for item in resp.json():
                p = item.get("paper", {})
                title = p.get("title", "").strip()
                arxiv_id = p.get("id", "")
                tags = [t.get("id", "") for t in p.get("tags", [])]
                upvotes = p.get("upvotes", 0) + item.get("numComments", 0)
                if title and is_nlp_related(title, tags):
                    papers.append({
                        "title": title,
                        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                        "source_url": f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else "",
                        "arxiv_id": arxiv_id,
                        "score": upvotes,
                        "source": "HuggingFace",
                        "abstract": p.get("summary", ""),
                    })
    except Exception as exc:
        log.warning("HF API 실패: %s", exc)

    # ── HTML 폴백 ─────────────────────────────
    if not papers:
        try:
            resp = requests.get(
                "https://huggingface.co/papers",
                headers={"User-Agent": "Mozilla/5.0 (compatible; nlp-digest/1.0)"},
                timeout=15,
            )
            soup = BeautifulSoup(resp.text, "html.parser")

            for a_tag in soup.select("a[href*='/papers/']"):
                href = a_tag.get("href", "")
                arxiv_id = href.split("/")[-1]
                # /papers/ 페이지 링크만 (분류 링크 제외)
                if not re.match(r"^\d{4}\.\d{4,}", arxiv_id):
                    continue
                title = a_tag.get_text(strip=True)
                if not title or not is_nlp_related(title):
                    continue
                papers.append({
                    "title": title,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "source_url": f"https://huggingface.co/papers/{arxiv_id}",
                    "arxiv_id": arxiv_id,
                    "score": 0,
                    "source": "HuggingFace",
                    "abstract": "",
                })
        except Exception as exc:
            log.warning("HF HTML 파싱 실패: %s", exc)

    log.info("HuggingFace 페이퍼 %d건 수집", len(papers))
    return papers


def fetch_pwc_papers() -> list[dict]:
    """Papers With Code API에서 최신 NLP 페이퍼를 수집합니다."""
    log.info("Papers With Code 페이퍼 수집 중...")
    papers: list[dict] = []
    seen: set[str] = set()

    try:
        resp = requests.get(
            "https://paperswithcode.com/api/v1/papers/?ordering=-published&page_size=30",
            headers={"User-Agent": "nlp-digest/1.0"},
            timeout=15,
        )
        resp.raise_for_status()
        for item in resp.json().get("results", []):
            title: str = item.get("title", "").strip()
            arxiv_id: str = item.get("arxiv_id", "") or ""
            abstract: str = item.get("abstract", "") or ""
            paper_slug: str = item.get("id", "")
            gh = item.get("github_link") or {}
            score: int = gh.get("stars", 0)

            if not title or not is_nlp_related(title, tags=None):
                continue

            key = arxiv_id or title
            if key in seen:
                continue
            seen.add(key)

            papers.append({
                "title": title,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://paperswithcode.com/paper/{paper_slug}",
                "source_url": f"https://paperswithcode.com/paper/{paper_slug}",
                "arxiv_id": arxiv_id,
                "score": score,
                "source": "Papers With Code",
                "abstract": abstract,
            })
    except Exception as exc:
        log.warning("Papers With Code 실패: %s", exc)

    log.info("Papers With Code 페이퍼 %d건 수집", len(papers))
    return papers


# ──────────────────────────────────────────────
# 2. 병합 및 중복 제거
# ──────────────────────────────────────────────

def merge_papers(hf: list[dict], pwc: list[dict], n: int = TOP_N) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []

    # HF 큐레이션 우선, 이후 PWC 순서로 삽입
    for p in hf + pwc:
        key = p["arxiv_id"] or p["title"]
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)

    # HF 큐레이션 페이퍼 우선, 그 다음 PWC GitHub 스타 순
    merged.sort(key=lambda x: (x["source"] != "HuggingFace", -x["score"]))
    return merged[:n]


# ──────────────────────────────────────────────
# 3. 초록 보완
# ──────────────────────────────────────────────

def fetch_abstract(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    try:
        resp = requests.get(
            f"https://export.arxiv.org/abs/{arxiv_id}",
            headers={"User-Agent": "nlp-digest/1.0"},
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        block = soup.select_one("blockquote.abstract")
        if block:
            return block.get_text(strip=True).removeprefix("Abstract:").strip()
    except Exception:
        pass
    return ""


# ──────────────────────────────────────────────
# 4. Gemini 요약
# ──────────────────────────────────────────────

def summarize_papers(papers: list[dict]) -> list[dict]:
    log.info("Gemini(%s)로 요약 중...", GEMINI_MODEL)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    for paper in papers:
        abstract = paper.get("abstract") or fetch_abstract(paper["arxiv_id"])
        paper["abstract"] = abstract  # build_message에서 재사용
        abstract_block = f"Abstract: {abstract[:800]}" if abstract else ""

        prompt = (
            "You are an expert NLP researcher writing a daily digest for Korean ML practitioners.\n\n"
            f"Paper title: {paper['title']}\n"
            f"{abstract_block}\n\n"
            "Write exactly these three sections and nothing else:\n"
            "EN: [One English sentence capturing the key contribution]\n"
            "KO: [같은 내용을 한국어 한 문장으로 작성]\n"
            "DETAIL: [핵심 방법론, 실험 결과, 의의를 포함하여 3~4문장으로 한국어로 상세히 설명]"
        )

        try:
            response = model.generate_content(prompt)
            en, ko, detail_parts = "", "", []
            current = None
            for line in response.text.strip().splitlines():
                if line.startswith("EN:"):
                    current = "en"
                    en = line[3:].strip()
                elif line.startswith("KO:"):
                    current = "ko"
                    ko = line[3:].strip()
                elif line.startswith("DETAIL:"):
                    current = "detail"
                    detail_parts.append(line[7:].strip())
                elif current == "detail" and line.strip():
                    detail_parts.append(line.strip())
            paper["en"] = en or paper["title"]
            paper["ko"] = ko or paper["title"]
            paper["detail"] = " ".join(detail_parts)
        except Exception as exc:
            log.warning("Gemini 요약 실패 ('%s'): %s", paper["title"][:40], exc)
            paper["en"] = paper["title"]
            paper["ko"] = paper["title"]
            paper["detail"] = paper.get("abstract", "")[:300]

    return papers


# ──────────────────────────────────────────────
# 5. Discord 전송
# ──────────────────────────────────────────────

def build_message(papers: list[dict]) -> str:
    today = date.today().strftime("%Y-%m-%d")
    lines = [f"**:books: NLP Daily Digest — {today}**", ""]
    for i, p in enumerate(papers, 1):
        title = p["title"]
        if len(title) > 110:
            title = title[:110] + "…"

        detail = p.get("detail", "").strip()
        paper_link = f"[ArXiv]({p['url']})" if p.get("url") else ""
        source_link = f"[{p['source']}]({p['source_url']})" if p.get("source_url") else p["source"]

        block = [
            f"**{i}. {title}**",
            f"> :flag_us: {p['en']}",
            f"> :flag_kr: {p['ko']}",
        ]
        if detail:
            block.append(f"> :notepad_spiral: {detail}")
        block += [f"> 논문: {paper_link}  |  출처: {source_link}", ""]
        lines += block

    return "\n".join(lines)


def send_to_discord(message: str) -> None:
    # Discord 메시지 최대 2000자 → 초과 시 분할
    chunks: list[str] = []
    while message:
        if len(message) <= 2000:
            chunks.append(message)
            break
        cut = message.rfind("\n", 0, 2000)
        if cut == -1:
            cut = 2000
        chunks.append(message[:cut])
        message = message[cut:].lstrip()

    for chunk in chunks:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": chunk},
            timeout=15,
        )
        resp.raise_for_status()
        log.info("Discord 전송 완료 (%d자)", len(chunk))


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main() -> None:
    hf_papers = fetch_hf_papers()
    pwc_papers = fetch_pwc_papers()
    papers = merge_papers(hf_papers, pwc_papers, n=TOP_N)

    if not papers:
        log.error("수집된 페이퍼가 없습니다. 종료합니다.")
        return

    log.info("최종 선정 페이퍼 %d건", len(papers))
    papers = summarize_papers(papers)
    message = build_message(papers)
    send_to_discord(message)
    log.info("완료!")


if __name__ == "__main__":
    main()
