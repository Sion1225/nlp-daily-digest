#!/usr/bin/env python3
"""NLP Daily Digest: HuggingFace + Reddit → Gemini 요약 → Discord 전송."""

import os
import re
import time
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


PWC_NLP_TASKS = [
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

def fetch_pwc_papers() -> list[dict]:
    """Papers With Code NLP task별 최신 논문을 수집합니다."""
    log.info("Papers With Code 페이퍼 수집 중...")
    papers: list[dict] = []
    seen: set[str] = set()

    for task in PWC_NLP_TASKS:
        try:
            resp = requests.get(
                f"https://paperswithcode.com/api/v1/papers/"
                f"?ordering=-published&page_size=5&task={task}",
                headers={"User-Agent": "nlp-digest/1.0"},
                timeout=15,
            )
            if not resp.ok:
                log.warning("PWC task '%s' 응답 %s", task, resp.status_code)
                continue
            for item in resp.json().get("results", []):
                title: str = item.get("title", "").strip()
                arxiv_id: str = item.get("arxiv_id", "") or ""
                abstract: str = item.get("abstract", "") or ""
                paper_slug: str = item.get("id", "")

                if not title:
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
                    "score": 0,
                    "source": "Papers With Code",
                    "abstract": abstract,
                })
        except Exception as exc:
            log.warning("PWC task '%s' 실패: %s", task, exc)

    log.info("Papers With Code 페이퍼 %d건 수집", len(papers))
    return papers


# ──────────────────────────────────────────────
# 2. 병합 및 중복 제거
# ──────────────────────────────────────────────

HF_CAP = 3  # HF 페이퍼 최대 선택 수 (나머지는 PWC로 채움)

def merge_papers(hf: list[dict], pwc: list[dict], n: int = TOP_N) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []

    # HF 큐레이션 최대 HF_CAP건 우선
    for p in hf:
        if len(result) >= HF_CAP:
            break
        key = p["arxiv_id"] or p["title"]
        if key not in seen:
            seen.add(key)
            result.append(p)

    # 나머지 슬롯은 PWC로 보충
    for p in pwc:
        if len(result) >= n:
            break
        key = p["arxiv_id"] or p["title"]
        if key not in seen:
            seen.add(key)
            result.append(p)

    return result


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
            raw = response.text.strip()

            en, ko, detail_parts = "", "", []
            current = None
            for line in raw.splitlines():
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
            log.info("파싱 결과 — EN: %r | DETAIL 길이: %d자", en[:60], len(paper["detail"]))
        except Exception as exc:
            log.warning("Gemini 요약 실패 ('%s'): %s", paper["title"][:40], exc)
            paper["en"] = paper["title"]
            paper["ko"] = paper["title"]
            paper["detail"] = paper.get("abstract", "")[:300]

    return papers


# ──────────────────────────────────────────────
# 5. Discord 전송 (논문 1건 = 메시지 1개)
# ──────────────────────────────────────────────

def _post(content: str) -> None:
    resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=15)
    if not resp.ok:
        log.error("Discord 오류 %s: %s", resp.status_code, resp.text)
    resp.raise_for_status()


def send_to_discord(papers: list[dict]) -> None:
    today = date.today().strftime("%Y-%m-%d")

    _post(f"📚 **NLP Daily Digest — {today}** ({len(papers)}건)")

    for i, p in enumerate(papers, 1):
        title = p["title"]
        if len(title) > 180:
            title = title[:180] + "…"

        detail = p.get("detail", "").strip()
        arxiv_url = p.get("url", "")
        source_url = p.get("source_url", "")
        source = p.get("source", "")

        lines = [
            f"**{i}. {title}**",
            f"🇺🇸 {p['en']}",
            f"🇰🇷 {p['ko']}",
        ]
        if detail:
            lines += ["", "📋 **상세 요약**", detail]

        link_parts = []
        if arxiv_url:
            link_parts.append(f"[논문 ArXiv](<{arxiv_url}>)")
        if source_url:
            link_parts.append(f"[{source}](<{source_url}>)")
        if link_parts:
            lines += ["", " | ".join(link_parts)]

        content = "\n".join(lines)
        if len(content) > 2000:
            content = content[:1997] + "…"

        time.sleep(0.5)  # Discord rate limit 방지
        _post(content)
        log.info("Discord 전송 완료 — %d/%d (%s)", i, len(papers), source)


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
    send_to_discord(papers)
    log.info("완료!")


if __name__ == "__main__":
    main()
