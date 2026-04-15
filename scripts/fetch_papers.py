#!/usr/bin/env python3
"""
Fetch latest OCD (Obsessive-Compulsive Disorder) research papers from PubMed E-utilities API.
Targets OCD-relevant journals across psychiatry, clinical psychology, CBT,
neuroscience, neuropsychiatry, and psychopharmacology.
"""

import json
import sys
import argparse
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote_plus

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

JOURNAL_BATCHES = [
    [
        "J Obsessive Compuls Relat Disord",
        "Am J Psychiatry",
        "JAMA Psychiatry",
        "Br J Psychiatry",
        "Psychol Med",
        "Mol Psychiatry",
        "Biol Psychiatry",
        "J Clin Psychiatry",
        "Eur Psychiatry",
        "CNS Spectr",
    ],
    [
        "Psychiatry Res",
        "J Psychiatr Res",
        "Compr Psychiatry",
        "Neuropsychopharmacology",
        "Acta Psychiatr Scand",
        "World Psychiatry",
        "BJPsych Open",
        "Front Psychiatry",
        "BMC Psychiatry",
        "Transl Psychiatry",
    ],
    [
        "Behav Res Ther",
        "J Anxiety Disord",
        "Cogn Behav Ther",
        "Behav Ther",
        "J Behav Ther Exp Psychiatry",
        "Behav Cogn Psychother",
        "Cogn Behav Pract",
        "J Consult Clin Psychol",
        "Clin Psychol Rev",
        "Psychother Psychosom",
    ],
    [
        "Neuroimage Clin",
        "Brain Stimul",
        "Neurosci Biobehav Rev",
        "Eur Arch Psychiatry Clin Neurosci",
        "Biol Psychiatry Cogn Neurosci Neuroimaging",
        "Int Clin Psychopharmacol",
        "Eur Neuropsychopharmacol",
        "J Child Psychol Psychiatry",
        "Hum Brain Mapp",
        "Cereb Cortex",
    ],
]

HEADERS = {"User-Agent": "OCDBrainBot/1.0 (research aggregator)"}


def build_batch_query(journals: list[str], days: int = 7) -> str:
    journal_part = " OR ".join([f'"{j}"[Journal]' for j in journals])
    ocd_core = '("obsessive-compulsive disorder"[Title/Abstract] OR OCD[Title/Abstract] OR "obsessive compulsive"[Title/Abstract] OR OCRD[Title/Abstract] OR compulsivity[Title/Abstract])'
    lookback = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y/%m/%d")
    date_part = f'"{lookback}"[Date - Publication] : "3000"[Date - Publication]'
    return f"({journal_part}) AND {ocd_core} AND {date_part}"


def search_papers_batch(days: int = 7, retmax_per_batch: int = 20) -> list[str]:
    all_pmids = []
    seen = set()
    for i, batch in enumerate(JOURNAL_BATCHES):
        if i > 0:
            time.sleep(1)
        query = build_batch_query(batch, days=days)
        params = f"?db=pubmed&term={quote_plus(query)}&retmax={retmax_per_batch}&sort=date&retmode=json"
        url = PUBMED_SEARCH + params
        for attempt in range(3):
            try:
                req = Request(url, headers=HEADERS)
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                pmids = data.get("esearchresult", {}).get("idlist", [])
                for pmid in pmids:
                    if pmid not in seen:
                        seen.add(pmid)
                        all_pmids.append(pmid)
                print(
                    f"[INFO] Batch {i + 1}/{len(JOURNAL_BATCHES)}: found {len(pmids)} papers (total unique: {len(all_pmids)})",
                    file=sys.stderr,
                )
                break
            except Exception as e:
                print(
                    f"[WARN] PubMed search batch {i + 1} attempt {attempt + 1} failed: {e}",
                    file=sys.stderr,
                )
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                continue
    return all_pmids


def fetch_details(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    chunk_size = 15
    all_papers = []
    for chunk_start in range(0, len(pmids), chunk_size):
        if chunk_start > 0:
            time.sleep(1)
        chunk = pmids[chunk_start : chunk_start + chunk_size]
        ids = ",".join(chunk)
        params = f"?db=pubmed&id={ids}&retmode=xml"
        url = PUBMED_FETCH + params
        xml_data = None
        for attempt in range(3):
            try:
                req = Request(url, headers=HEADERS)
                with urlopen(req, timeout=60) as resp:
                    xml_data = resp.read().decode()
                break
            except Exception as e:
                print(
                    f"[WARN] PubMed fetch attempt {attempt + 1} failed: {e}",
                    file=sys.stderr,
                )
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        if xml_data is None:
            print(
                f"[ERROR] PubMed fetch failed for chunk starting at {chunk_start}",
                file=sys.stderr,
            )
            continue

        try:
            root = ET.fromstring(xml_data)
            for article in root.findall(".//PubmedArticle"):
                medline = article.find(".//MedlineCitation")
                art = medline.find(".//Article") if medline else None
                if art is None:
                    continue

                title_el = art.find(".//ArticleTitle")
                title = (
                    (title_el.text or "").strip()
                    if title_el is not None and title_el.text
                    else ""
                )

                abstract_parts = []
                for abs_el in art.findall(".//Abstract/AbstractText"):
                    label = abs_el.get("Label", "")
                    text = "".join(abs_el.itertext()).strip()
                    if label and text:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)[:2000]

                journal_el = art.find(".//Journal/Title")
                journal = (
                    (journal_el.text or "").strip()
                    if journal_el is not None and journal_el.text
                    else ""
                )

                pub_date = art.find(".//PubDate")
                date_str = ""
                if pub_date is not None:
                    year = pub_date.findtext("Year", "")
                    month = pub_date.findtext("Month", "")
                    day = pub_date.findtext("Day", "")
                    parts = [p for p in [year, month, day] if p]
                    date_str = " ".join(parts)

                pmid_el = medline.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

                keywords = []
                for kw in medline.findall(".//KeywordList/Keyword"):
                    if kw.text:
                        keywords.append(kw.text.strip())

                authors = []
                for author in art.findall(".//AuthorList/Author")[:6]:
                    last = author.findtext("LastName", "")
                    fore = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {fore}".strip())
                if len(art.findall(".//AuthorList/Author")) > 6:
                    authors.append("et al.")

                all_papers.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "authors": "; ".join(authors),
                        "journal": journal,
                        "date": date_str,
                        "abstract": abstract,
                        "url": link,
                        "keywords": keywords,
                    }
                )
        except ET.ParseError as e:
            print(f"[ERROR] XML parse failed for chunk: {e}", file=sys.stderr)

    return all_papers


def main():
    parser = argparse.ArgumentParser(description="Fetch OCD papers from PubMed")
    parser.add_argument("--days", type=int, default=7, help="Lookback days")
    parser.add_argument(
        "--max-papers", type=int, default=50, help="Max papers to fetch"
    )
    parser.add_argument("--output", default="-", help="Output file (- for stdout)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    query = build_batch_query(JOURNAL_BATCHES[0], days=args.days)
    print(
        f"[INFO] Searching PubMed for OCD papers from last {args.days} days (batched)...",
        file=sys.stderr,
    )

    pmids = search_papers_batch(days=args.days, retmax_per_batch=args.max_papers // 4)
    print(f"[INFO] Found {len(pmids)} unique papers total", file=sys.stderr)

    if not pmids:
        print("NO_CONTENT", file=sys.stderr)
        if args.json:
            print(
                json.dumps(
                    {
                        "date": datetime.now(timezone(timedelta(hours=8))).strftime(
                            "%Y-%m-%d"
                        ),
                        "count": 0,
                        "papers": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    papers = fetch_details(pmids)
    print(f"[INFO] Fetched details for {len(papers)} papers", file=sys.stderr)

    output_data = {
        "date": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d"),
        "count": len(papers),
        "papers": papers,
    }

    out_str = json.dumps(output_data, ensure_ascii=False, indent=2)

    if args.output == "-":
        print(out_str)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"[INFO] Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
