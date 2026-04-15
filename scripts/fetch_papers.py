#!/usr/bin/env python3
"""
Fetch latest OCD (Obsessive-Compulsive Disorder) research papers from PubMed E-utilities API.
Targets OCD-relevant journals across psychiatry, clinical psychology, CBT,
neuroscience, neuropsychiatry, and psychopharmacology.
"""

import json
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote_plus

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

JOURNALS = [
    "Journal of Obsessive-Compulsive and Related Disorders",
    "American Journal of Psychiatry",
    "JAMA Psychiatry",
    "British Journal of Psychiatry",
    "Psychological Medicine",
    "Molecular Psychiatry",
    "Biological Psychiatry",
    "Biiological Psychiatry Cognitive Neuroscience and Neuroimaging",
    "Journal of Clinical Psychiatry",
    "European Psychiatry",
    "CNS Spectrums",
    "Psychiatry Research",
    "Journal of Psychiatric Research",
    "Comprehensive Psychiatry",
    "International Clinical Psychopharmacology",
    "European Neuropsychopharmacology",
    "Neuropsychopharmacology",
    "Acta Psychiatrica Scandinavica",
    "World Psychiatry",
    "BJPsych Open",
    "Frontiers in Psychiatry",
    "BMC Psychiatry",
    "Behaviour Research and Therapy",
    "Journal of Anxiety Disorders",
    "Cognitive Behaviour Therapy",
    "Cognitive Therapy and Research",
    "Clinical Psychology Review",
    "Journal of Consulting and Clinical Psychology",
    "Behavior Therapy",
    "Cognitive and Behavioral Practice",
    "Journal of Behavior Therapy and Experimental Psychiatry",
    "Behavioural and Cognitive Psychotherapy",
    "Psychotherapy and Psychosomatics",
    "Clinical Child and Family Psychology Review",
    "Journal of Child Psychology and Psychiatry",
    "Child Psychiatry and Human Development",
    "NeuroImage Clinical",
    "Neuroscience and Biobehavioral Reviews",
    "Progress in Neuro-Psychopharmacology and Biological Psychiatry",
    "Human Brain Mapping",
    "Cerebral Cortex",
    "Brain Stimulation",
    "Translational Psychiatry",
    "European Archives of Psychiatry and Clinical Neuroscience",
    "Frontiers in Behavioral Neuroscience",
    "Frontiers in Human Neuroscience",
]

OCD_KEYWORDS_CORE = [
    '"obsessive-compulsive disorder"',
    '"obsessive compulsive disorder"',
    "OCD",
    '"obsessive compulsive"',
    '"obsessive-compulsive and related disorders"',
    "OCRD",
    "compulsivity",
    "obsessions",
    "compulsions",
    '"intrusive thoughts"',
]

OCD_KEYWORDS_SYMPTOMS = [
    '"intrusive thoughts"',
    '"mental rituals"',
    "checking",
    "washing",
    "contamination",
    "symmetry",
    "ordering",
    '"taboo obsessions"',
    '"unacceptable thoughts"',
    "scrupulosity",
    '"harm obsessions"',
    '"sexual obsessions"',
    '"religious obsessions"',
    '"somatic obsessions"',
    "hoarding",
    '"body dysmorphic disorder"',
    "trichotillomania",
    '"excoriation disorder"',
]

OCD_KEYWORDS_ASSESSMENT = [
    '"Yale-Brown Obsessive Compulsive Scale"',
    "Y-BOCS",
    '"Children Yale-Brown Obsessive Compulsive Scale"',
    "CY-BOCS",
    '"Dimensional Obsessive-Compulsive Scale"',
    "DOCS",
    '"obsessive beliefs"',
    '"family accommodation"',
    '"symptom dimensions"',
    '"treatment response"',
    "remission",
    "relapse",
]

OCD_KEYWORDS_TREATMENT = [
    '"cognitive behavioral therapy"',
    "CBT",
    '"exposure and response prevention"',
    "ERP",
    '"exposure therapy"',
    '"inhibitory learning"',
    '"metacognitive therapy"',
    '"acceptance and commitment therapy"',
    "ACT",
    '"family-based CBT"',
    "teletherapy",
    '"internet-delivered CBT"',
    "SSRI",
    "clomipramine",
    "fluoxetine",
    "sertraline",
    "fluvoxamine",
    "escitalopram",
    "paroxetine",
    "augmentation",
    '"antipsychotic augmentation"',
    "aripiprazole",
    "risperidone",
    '"glutamate-modulating"',
    "memantine",
    "lamotrigine",
]

OCD_KEYWORDS_NEUROSCIENCE = [
    '"cortico-striato-thalamo-cortical"',
    "CSTC",
    '"orbitofrontal cortex"',
    '"anterior cingulate cortex"',
    "striatum",
    "caudate",
    '"resting-state connectivity"',
    '"functional connectivity"',
    "fMRI",
    "DTI",
    '"error monitoring"',
    '"habit learning"',
    '"cognitive control"',
    '"response inhibition"',
    "biomarkers",
    "genetics",
    '"polygenic risk"',
    "inflammation",
    "glutamate",
    "serotonin",
]

OCD_KEYWORDS_POPULATIONS = [
    '"pediatric OCD"',
    '"child OCD"',
    '"adolescent OCD"',
    '"perinatal OCD"',
    '"postpartum OCD"',
    '"tic-related OCD"',
    '"autism and OCD"',
    '"OCD and ADHD"',
    '"OCD and depression"',
    '"OCD and suicidality"',
    '"treatment-resistant OCD"',
    '"severe OCD"',
]

OCD_KEYWORDS_ADVANCED = [
    '"treatment-resistant"',
    "refractory",
    "DBS",
    '"deep brain stimulation"',
    "rTMS",
    '"brain stimulation"',
    "neuromodulation",
    "neuroimaging",
    "connectivity",
    "biomarker",
    "genetics",
    "glutamate",
    "habit",
]

HEADERS = {"User-Agent": "OCDBrainBot/1.0 (research aggregator)"}


def build_query(days: int = 7, max_journals: int = 20) -> str:
    journal_part = " OR ".join([f'"{j}"[Journal]' for j in JOURNALS[:max_journals]])

    ocd_core = " OR ".join([f"{k}[Title/Abstract]" for k in OCD_KEYWORDS_CORE])
    ocd_symptoms = " OR ".join([f"{k}[Title/Abstract]" for k in OCD_KEYWORDS_SYMPTOMS])
    ocd_treatment = " OR ".join(
        [f"{k}[Title/Abstract]" for k in OCD_KEYWORDS_TREATMENT]
    )
    ocd_context = " OR ".join(
        [
            f"{k}[Title/Abstract]"
            for k in OCD_KEYWORDS_NEUROSCIENCE
            + OCD_KEYWORDS_POPULATIONS
            + OCD_KEYWORDS_ADVANCED
        ]
    )

    lookback = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y/%m/%d")
    date_part = f'"{lookback}"[Date - Publication] : "3000"[Date - Publication]'

    query = (
        f"(({journal_part})) AND "
        f"({ocd_core} OR "
        f"({ocd_symptoms}) OR "
        f"(({ocd_core}) AND ({ocd_treatment} OR {ocd_context}))) AND "
        f"{date_part}"
    )
    return query


def search_papers(query: str, retmax: int = 50) -> list[str]:
    params = (
        f"?db=pubmed&term={quote_plus(query)}&retmax={retmax}&sort=date&retmode=json"
    )
    url = PUBMED_SEARCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"[ERROR] PubMed search failed: {e}", file=sys.stderr)
        return []


def fetch_details(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    ids = ",".join(pmids)
    params = f"?db=pubmed&id={ids}&retmode=xml"
    url = PUBMED_FETCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=60) as resp:
            xml_data = resp.read().decode()
    except Exception as e:
        print(f"[ERROR] PubMed fetch failed: {e}", file=sys.stderr)
        return []

    papers = []
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

            papers.append(
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
        print(f"[ERROR] XML parse failed: {e}", file=sys.stderr)

    return papers


def main():
    parser = argparse.ArgumentParser(description="Fetch OCD papers from PubMed")
    parser.add_argument("--days", type=int, default=7, help="Lookback days")
    parser.add_argument(
        "--max-papers", type=int, default=50, help="Max papers to fetch"
    )
    parser.add_argument("--output", default="-", help="Output file (- for stdout)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    query = build_query(days=args.days)
    print(
        f"[INFO] Searching PubMed for OCD papers from last {args.days} days...",
        file=sys.stderr,
    )

    pmids = search_papers(query, retmax=args.max_papers)
    print(f"[INFO] Found {len(pmids)} papers", file=sys.stderr)

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
