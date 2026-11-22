"""
Collects AliveCor content into data/docs.jsonl:
  - Kardia blog + product/technology pages (from kardia.com sitemaps)
  - AliveCor help center articles (from the public Zendesk API)

Run it again at any time: only new or changed documents are added/updated.
"""
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser

import requests
import trafilatura
from bs4 import BeautifulSoup

DATA = Path("data")
DATA.mkdir(exist_ok=True)
DOCS = DATA / "docs.jsonl"
STATE = DATA / "state.json"

HEADERS = {"User-Agent": "cardio-knowledge-assistant (student project)"}
SITEMAPS = {
    "blog": "https://kardia.com/article/sitemap-index.xml",
    "product": "https://kardia.com/shop/sitemap-index.xml",
    "technology": "https://kardia.com/technology/sitemap-index.xml",
}
ZENDESK = "https://alivecor.zendesk.com/api/v2/help_center/en-us"
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_docs():
    if not DOCS.exists():
        return {}
    lines = DOCS.read_text(encoding="utf-8").splitlines()
    return {d["id"]: d for d in map(json.loads, lines) if d}


def save_docs(docs):
    with DOCS.open("w", encoding="utf-8") as f:
        for d in docs.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


# ---------- website (sitemaps) ----------

robots = robotparser.RobotFileParser("https://kardia.com/robots.txt")
robots.read()


def sitemap_urls(url):
    """Returns all page URLs in a sitemap (follows nested sitemap indexes)."""
    root = ET.fromstring(get(url).content)
    if root.tag.endswith("sitemapindex"):
        return [u for loc in root.findall("sm:sitemap/sm:loc", NS)
                for u in sitemap_urls(loc.text.strip())]
    return [loc.text.strip() for loc in root.findall("sm:url/sm:loc", NS)]


def ingest_web(docs):
    new = changed = 0
    for source, sitemap in SITEMAPS.items():
        for url in sitemap_urls(sitemap):
            if url.rstrip("/").endswith("/blog"):          # listing page, not an article
                continue
            if not robots.can_fetch(HEADERS["User-Agent"], url):
                continue
            html = get(url).text
            meta = trafilatura.extract_metadata(html)
            # the page's meta description helps when the main content is loaded by JavaScript
            description = meta.description if meta and meta.description else ""
            body = trafilatura.extract(html, include_comments=False) or ""
            text = "\n".join(part for part in (description, body) if part).strip()
            if not text:
                continue
            doc_id = f"web:{url}"
            old = docs.get(doc_id)
            if old and old["hash"] == sha(text):          # unchanged
                continue
            docs[doc_id] = {
                "id": doc_id,
                "source": source,
                "url": url,
                "title": (meta.title if meta and meta.title else url),
                "date": (meta.date if meta else None),
                "section": None,
                "text": text,
                "hash": sha(text),
                "fetched_at": now(),
            }
            changed += bool(old)
            new += not old
            time.sleep(1)                                  # be polite to the server
    return new, changed


# ---------- help center (Zendesk API) ----------

def paginate(url, key):
    while url:
        page = get(url).json()
        yield from page[key]
        url = page.get("next_page")


def ingest_helpcenter(docs, since):
    sections = {s["id"]: s["name"]
                for s in paginate(f"{ZENDESK}/sections.json?per_page=100", "sections")}
    new = changed = 0
    for a in paginate(f"{ZENDESK}/articles.json?per_page=100", "articles"):
        if a["draft"] or a["updated_at"] <= since:        # only new/updated since last run
            continue
        text = BeautifulSoup(a["body"] or "", "html.parser").get_text("\n", strip=True)
        if not text:
            continue
        doc_id = f"zd:{a['id']}"
        old = docs.get(doc_id)
        if old and old["hash"] == sha(text):          # only metadata changed, text is the same
            continue
        docs[doc_id] = {
            "id": doc_id,
            "source": "help",
            "url": a["html_url"],
            "title": a["title"],
            "date": a["created_at"][:10],
            "section": sections.get(a["section_id"]),
            "text": text,
            "hash": sha(text),
            "fetched_at": now(),
        }
        changed += bool(old)
        new += not old
    return new, changed


if __name__ == "__main__":
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    docs = load_docs()
    started = now()

    print("website      new/changed:", ingest_web(docs))
    print("help center  new/changed:",
          ingest_helpcenter(docs, state.get("helpcenter_last_run", "1970-01-01T00:00:00Z")))

    state["helpcenter_last_run"] = started
    save_docs(docs)
    STATE.write_text(json.dumps(state, indent=2))
    print("total documents:", len(docs))