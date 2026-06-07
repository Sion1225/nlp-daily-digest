#!/usr/bin/env python3
"""NLP Daily Digest: HuggingFace + Reddit → Gemini 요약 → Discord 전송."""

import os
import re
import time
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from bs4 import BeautifulSoup
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

DAILY_CAP   = 3   # 일간 핫 논문
WEEKLY_CAP  = 2   # 주간 핫 논문
MONTHLY_CAP = 1   # 월간 핫 논문
TOP_N = DAILY_CAP + WEEKLY_CAP + MONTHLY_CAP  # = 6

WINDOW_LABEL = {
    "daily":   "📅 일간",
    "weekly":  "📆 주간",
    "monthly": "🗓️ 월간",
    "arxiv":   "🔬 ArXiv",
}

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

ARXIV_RE = re.compile(r"arxiv\.org/abs/([\d.]+v?\d*)")


def is_nlp_related(title: str, tags: list[str] | None = None) -> bool:
    text = title.lower()
    if tags:
        text += " " + " ".join(t.lower() for t in tags)
    return any(kw in text for kw in NLP_KEYWORDS)


# ──────────────────────────────────────────────
# 1. 데이터 수집
# ──────────────────────────────────────────────

def _fetch_hf_day(date_str: str) -> tuple[list[dict], dict[str, int]]:
    """특정 날짜의 HF papers를 수집합니다 (내부 함수)."""
    papers: list[dict] = []
    upvotes: dict[str, int] = {}

    try:
        resp = requests.get(
            "https://huggingface.co/api/daily_papers",
            params={"date": date_str},
            headers={"User-Agent": "nlp-digest/1.0"},
            timeout=15,
        )
        if resp.ok:
            for item in resp.json():
                p = item.get("paper", {})
                title = p.get("title", "").strip()
                arxiv_id = p.get("id", "")
                tags = [t.get("id", "") for t in p.get("tags", [])]
                score = p.get("upvotes", 0) + item.get("numComments", 0)
                if arxiv_id:
                    upvotes[arxiv_id] = score
                if title and is_nlp_related(title, tags):
                    papers.append({
                        "title": title,
                        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                        "source_url": f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else "",
                        "arxiv_id": arxiv_id,
                        "score": score,
                        "source": "HuggingFace",
                        "abstract": p.get("summary", ""),
                    })
    except Exception as exc:
        log.warning("HF API 실패 (%s): %s", date_str, exc)

    if not papers:
        try:
            resp = requests.get(
                f"https://huggingface.co/papers?date={date_str}",
                headers={"User-Agent": "Mozilla/5.0 (compatible; nlp-digest/1.0)"},
                timeout=15,
            )
            soup = BeautifulSoup(resp.text, "html.parser")
            for a_tag in soup.select("a[href*='/papers/']"):
                href = a_tag.get("href", "")
                arxiv_id = href.split("/")[-1]
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
            log.warning("HF HTML 실패 (%s): %s", date_str, exc)

    return papers, upvotes


def fetch_hf_daily() -> tuple[list[dict], dict[str, int]]:
    """어제 HF papers (일간 핫 논문)."""
    d = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("HF 일간 수집 (%s)...", d)
    papers, upvotes = _fetch_hf_day(d)
    papers.sort(key=lambda x: x["score"], reverse=True)
    log.info("HF 일간 %d건", len(papers))
    return papers, upvotes


def fetch_hf_range(start: date, end: date, step_days: int = 1) -> tuple[list[dict], dict[str, int]]:
    """날짜 범위의 HF papers를 집계합니다 (주간/월간용)."""
    best: dict[str, dict] = {}   # arxiv_id → 최고 upvotes 논문
    all_upvotes: dict[str, int] = {}

    current = start
    while current <= end:
        papers, upvotes = _fetch_hf_day(current.strftime("%Y-%m-%d"))
        all_upvotes.update(upvotes)
        for p in papers:
            key = p["arxiv_id"] or p["title"]
            if key not in best or p["score"] > best[key]["score"]:
                best[key] = p
        current += timedelta(days=step_days)
        time.sleep(0.3)

    result = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return result, all_upvotes


ARXIV_NS = "{http://www.w3.org/2005/Atom}"
# cs.CL: NLP 전용 / cs.AI: AI 전반 / cs.LG: 머신러닝
ARXIV_CATEGORIES = "cat:cs.CL OR cat:cs.AI"

def fetch_arxiv_papers(hf_upvotes: dict[str, int] | None = None) -> list[dict]:
    """ArXiv cs.CL 최신 논문을 수집합니다 (HF에 없는 논문 보완용)."""
    log.info("ArXiv 논문 수집 중...")
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": ARXIV_CATEGORIES,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": 40,
            },
            headers={"User-Agent": "nlp-digest/1.0"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.warning("ArXiv 수집 실패: %s", exc)
        return []

    papers: list[dict] = []
    seen: set[str] = set()

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        log.warning("ArXiv XML 파싱 실패: %s", exc)
        return []

    for entry in root.findall(f"{ARXIV_NS}entry"):
        title_el  = entry.find(f"{ARXIV_NS}title")
        summary_el = entry.find(f"{ARXIV_NS}summary")
        id_el     = entry.find(f"{ARXIV_NS}id")
        if title_el is None or id_el is None:
            continue

        title    = title_el.text.strip().replace("\n", " ")
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
        arxiv_url = id_el.text.strip()
        # URL 형식: http://arxiv.org/abs/2406.12345v2 → 2406.12345
        arxiv_id  = re.sub(r"v\d+$", "", arxiv_url.split("/abs/")[-1])

        if not is_nlp_related(title):
            continue
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)

        # HF에 같은 논문이 있으면 upvotes를 score로 사용, 없으면 0 (최신순)
        score = (hf_upvotes or {}).get(arxiv_id, 0)

        papers.append({
            "title": title,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "source_url": "https://arxiv.org/list/cs.CL/recent",
            "arxiv_id": arxiv_id,
            "score": score,
            "source": "ArXiv cs.CL",
            "abstract": abstract,
        })

    papers.sort(key=lambda x: x["score"], reverse=True)
    log.info("ArXiv 논문 %d건 수집 (HF업보트>0: %d건)", len(papers),
             sum(1 for p in papers if p["score"] > 0))
    return papers


# ──────────────────────────────────────────────
# 2. 병합 및 중복 제거
# ──────────────────────────────────────────────

def merge_papers(
    daily: list[dict],
    weekly: list[dict],
    monthly: list[dict],
    arxiv: list[dict],
) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []

    def _fill(pool: list[dict], cap: int, window: str) -> None:
        count = 0
        for p in pool:
            if count >= cap:
                break
            key = p["arxiv_id"] or p["title"]
            if key not in seen:
                seen.add(key)
                p["window"] = window
                result.append(p)
                count += 1

    _fill(daily,   DAILY_CAP,   "daily")
    _fill(weekly,  WEEKLY_CAP,  "weekly")
    _fill(monthly, MONTHLY_CAP, "monthly")

    # 각 버킷이 부족할 경우 ArXiv로 보충
    remaining = TOP_N - len(result)
    if remaining > 0:
        log.info("ArXiv로 %d건 보충", remaining)
        _fill(arxiv, remaining, "arxiv")

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
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    _post(f"📚 **NLP Daily Digest — {yesterday}** ({len(papers)}건)")

    for i, p in enumerate(papers, 1):
        title = p["title"]
        if len(title) > 180:
            title = title[:180] + "…"

        detail = p.get("detail", "").strip()
        arxiv_url = p.get("url", "")
        source_url = p.get("source_url", "")
        source = p.get("source", "")

        window_tag = WINDOW_LABEL.get(p.get("window", ""), "")
        lines = [
            f"**{i}. {title}** `{window_tag}`",
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
    today = date.today()

    # ── 일간: 어제 (하루 치 업보트 누적) ─────
    daily_papers, daily_upvotes = fetch_hf_daily()

    # ── 주간: 2~8일 전 (7일간) ──────────────
    week_end   = today - timedelta(days=2)
    week_start = today - timedelta(days=8)
    log.info("HF 주간 수집 (%s ~ %s)...", week_start, week_end)
    weekly_papers, weekly_upvotes = fetch_hf_range(week_start, week_end)
    log.info("HF 주간 %d건", len(weekly_papers))

    # ── 월간: 9~30일 전 (3일 간격 샘플링) ───
    month_end   = today - timedelta(days=9)
    month_start = today - timedelta(days=30)
    log.info("HF 월간 수집 (%s ~ %s, 3일 간격)...", month_start, month_end)
    monthly_papers, monthly_upvotes = fetch_hf_range(month_start, month_end, step_days=3)
    log.info("HF 월간 %d건", len(monthly_papers))

    # ArXiv upvotes 맵 통합 (일간 → 주간 → 월간 순으로 덮어쓰기)
    combined_upvotes = {**monthly_upvotes, **weekly_upvotes, **daily_upvotes}
    arxiv_papers = fetch_arxiv_papers(combined_upvotes)

    papers = merge_papers(daily_papers, weekly_papers, monthly_papers, arxiv_papers)

    if not papers:
        log.error("수집된 페이퍼가 없습니다. 종료합니다.")
        return

    log.info("최종 선정 페이퍼 %d건 (일간 %d / 주간 %d / 월간 %d)",
             len(papers),
             sum(1 for p in papers if p.get("window") == "daily"),
             sum(1 for p in papers if p.get("window") == "weekly"),
             sum(1 for p in papers if p.get("window") == "monthly"))

    papers = summarize_papers(papers)
    send_to_discord(papers)
    log.info("완료!")


if __name__ == "__main__":
    main()
