#!/usr/bin/env python3
"""
Generate OCD (Obsessive-Compulsive Disorder) daily report HTML using Zhipu AI.
Reads papers JSON, analyzes with AI (GLM-5-Turbo with fallback chain), generates styled HTML.
Uses the Psychiatry-brain warm color scheme.
"""

import json
import sys
import os
import re
import glob
import time
import argparse
from datetime import datetime, timezone, timedelta

import httpx

API_BASE = os.environ.get(
    "ZHIPU_API_BASE", "https://open.bigmodel.cn/api/coding/paas/v4"
)
MODEL_PRIMARY = "GLM-5-Turbo"
MODEL_FALLBACKS = ["GLM-4.7", "GLM-4.7-Flash"]
ALL_MODELS = [MODEL_PRIMARY] + MODEL_FALLBACKS
MAX_TOKENS = 100000
REQUEST_TIMEOUT = 660
MAX_RETRIES = 3

SYSTEM_PROMPT = (
    "你是精神醫學領域的資深研究員，專精於強迫症 (Obsessive-Compulsive Disorder, OCD) "
    "及強迫症相關障礙症 (OCRD) 的臨床研究與學術文獻分析。你的任務是：\n"
    "1. 從提供的醫學文獻中，篩選出與 OCD 最相關且最具臨床意義與研究價值的論文\n"
    "2. 對每篇論文進行繁體中文摘要、分類、PICO 分析\n"
    "3. 評估其臨床實用性（高/中/低）\n"
    "4. 生成適合醫療專業人員閱讀的日報\n\n"
    "輸出格式要求：\n"
    "- 語言：繁體中文（台灣用語）\n"
    "- 專業但易懂\n"
    "- 每篇論文需包含：中文標題、一句話總結、PICO分析、臨床實用性、分類標籤\n"
    "- 最後提供今日精選 TOP 5（最重要/最影響臨床實踐的論文）\n"
    "- 回傳格式必須是純 JSON，不要用 markdown code block 包裹，不要有任何額外文字\n"
    "- 確保輸出是合法的 JSON 格式，所有字串必須正確轉義"
)

OCD_TAGS = [
    "強迫症",
    "暴露與反應預防",
    "ERP",
    "CBT",
    "SSRI",
    "clomipramine",
    "藥物增效",
    "aripiprazole",
    "risperidone",
    "DBS 腦深層刺激",
    "rTMS",
    "神經調控",
    "神經影像",
    "CSTC 迴路",
    "眼眶額葉皮質",
    "前扣帶迴",
    "紋狀體",
    "生物標記",
    "遺傳學",
    "麩胺酸",
    "血清素",
    "發炎",
    "習慣學習",
    "認知控制",
    "反應抑制",
    "兒少強迫症",
    "家庭調適",
    "產後強迫症",
    " tic相關OCD",
    "自閉症與OCD",
    "治療阻抗型OCD",
    "儲物症",
    "身體畸形症",
    "拔毛症",
    "摳皮症",
    "解離症",
    "焦慮症",
    "自殺風險",
    "認知行為治療",
    "後設認知治療",
    "接納承諾治療",
    "網路治療",
    "精準精神醫學",
    "臨床試驗",
    "系統性回顧/統合分析",
]


def collect_excluded_pmids(days: int = 7) -> set[str]:
    excluded = set()
    tz_utc = timezone.utc
    cutoff = datetime.now(tz_utc) - timedelta(days=days)
    html_files = sorted(glob.glob("docs/ocd-*.html"), reverse=True)
    for f in html_files[:days]:
        basename = os.path.basename(f)
        date_part = basename.replace("ocd-", "").replace(".html", "")
        try:
            file_date = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=tz_utc)
            if file_date < cutoff:
                continue
        except ValueError:
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                content = fh.read()
            pmids = re.findall(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", content)
            if pmids:
                print(
                    f"[INFO] Excluding {len(pmids)} PMIDs from {basename}",
                    file=sys.stderr,
                )
                excluded.update(pmids)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}", file=sys.stderr)
    if excluded:
        print(
            f"[INFO] Total excluded PMIDs (last {days} days): {len(excluded)}",
            file=sys.stderr,
        )
    return excluded


def load_papers(input_path: str) -> dict:
    if input_path == "-":
        data = json.load(sys.stdin)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data


def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else ""
        text = text.rstrip("`").strip()
    if text.startswith("json\n"):
        text = text[5:]
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return text


def try_parse_json(text: str) -> dict | None:
    text = clean_json_response(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    patterns = [
        r'\{[\s\S]*"top_picks"[\s\S]*\}',
        r'\{[\s\S]*"all_papers"[\s\S]*\}',
        r'\{[\s\S]*"date"[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(0)
            depth = 0
            end = len(candidate)
            for i, ch in enumerate(candidate):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = candidate[:end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    print(
        f"[WARN] All JSON parse attempts failed. Raw text preview: {text[:500]}",
        file=sys.stderr,
    )
    return None


def analyze_papers(api_key: str, papers_data: dict) -> dict | None:
    tz_taipei = timezone(timedelta(hours=8))
    date_str = papers_data.get("date", datetime.now(tz_taipei).strftime("%Y-%m-%d"))
    paper_count = papers_data.get("count", 0)
    papers_text = json.dumps(
        papers_data.get("papers", []), ensure_ascii=False, indent=2
    )

    prompt = f"""以下是 {date_str} 從 PubMed 抓取的最新強迫症 (OCD) 及強迫症相關障礙症 (OCRD) 相關文獻（共 {paper_count} 篇）。

請進行以下分析，並以 JSON 格式回傳（不要用 markdown code block）：

{{
  "date": "{date_str}",
  "market_summary": "2-3句話總結今天 OCD 文獻的整體趨勢與亮點，包含重要的臨床發現",
  "top_picks": [
    {{
      "rank": 1,
      "title_zh": "中文標題",
      "title_en": "English Title",
      "authors": "主要作者",
      "journal": "期刊名",
      "summary": "一句話總結（繁體中文，點出核心發現與臨床意義）",
      "pico": {{
        "population": "研究對象",
        "intervention": "介入措施",
        "comparison": "對照組",
        "outcome": "主要結果"
      }},
      "clinical_utility": "高/中/低",
      "utility_reason": "為什麼實用的一句話說明",
      "tags": ["標籤1", "標籤2"],
      "url": "原文連結",
      "emoji": "相關emoji"
    }}
  ],
  "all_papers": [
    {{
      "title_zh": "中文標題",
      "title_en": "English Title",
      "journal": "期刊名",
      "summary": "一句話總結",
      "clinical_utility": "高/中/低",
      "tags": ["標籤1"],
      "url": "連結",
      "emoji": "emoji"
    }}
  ],
  "keywords": ["關鍵字1", "關鍵字2"],
  "topic_distribution": {{
    "強迫症": 3,
    "ERP": 2
  }}
}}

原始文獻資料：
{papers_text}

請篩選出最重要的 TOP 5-8 篇論文放入 top_picks（按重要性排序），其餘放入 all_papers。
每篇 paper 的 tags 請從以下選擇最合適的：{", ".join(OCD_TAGS)}
記住：回傳純 JSON，不要用 ```json``` 包裹，確保 JSON 格式正確可解析。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for model in ALL_MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                print(
                    f"[INFO] Trying {model} (attempt {attempt + 1}/{MAX_RETRIES})...",
                    file=sys.stderr,
                )
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": MAX_TOKENS,
                }
                resp = httpx.post(
                    f"{API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(
                        f"[WARN] Rate limited on {model}, waiting {wait}s...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                if resp.status_code == 400 and "model" in resp.text.lower():
                    print(
                        f"[WARN] Model {model} not available, trying next...",
                        file=sys.stderr,
                    )
                    break
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"].strip()

                result = try_parse_json(raw_text)
                if result is None:
                    print(
                        f"[WARN] JSON parse failed on {model} attempt {attempt + 1}, retrying...",
                        file=sys.stderr,
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(5)
                    continue

                print(
                    f"[INFO] Analysis complete with {model}: "
                    f"{len(result.get('top_picks', []))} top picks, "
                    f"{len(result.get('all_papers', []))} total",
                    file=sys.stderr,
                )
                result["_model_used"] = model
                return result

            except httpx.HTTPStatusError as e:
                print(
                    f"[ERROR] HTTP {e.response.status_code} on {model}: {e.response.text[:300]}",
                    file=sys.stderr,
                )
                if e.response.status_code == 429:
                    wait = 60 * (attempt + 1)
                    time.sleep(wait)
                    continue
                break
            except httpx.TimeoutException:
                print(
                    f"[WARN] Timeout on {model} attempt {attempt + 1}", file=sys.stderr
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(10)
                continue
            except Exception as e:
                print(
                    f"[ERROR] {model} failed: {type(e).__name__}: {e}", file=sys.stderr
                )
                break

    print("[ERROR] All models and attempts failed", file=sys.stderr)
    return None


def generate_html(analysis: dict) -> str:
    date_str = analysis.get(
        "date", datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    )
    date_parts = date_str.split("-")
    if len(date_parts) == 3:
        date_display = f"{date_parts[0]}年{int(date_parts[1])}月{int(date_parts[2])}日"
    else:
        date_display = date_str

    summary = analysis.get("market_summary", "")
    top_picks = analysis.get("top_picks", [])
    all_papers = analysis.get("all_papers", [])
    keywords = analysis.get("keywords", [])
    topic_dist = analysis.get("topic_distribution", {})
    model_used = analysis.get("_model_used", MODEL_PRIMARY)

    top_picks_html = ""
    for pick in top_picks:
        tags_html = "".join(
            f'<span class="tag">{t}</span>' for t in pick.get("tags", [])
        )
        util = pick.get("clinical_utility", "中")
        utility_class = (
            "utility-high"
            if util == "高"
            else ("utility-mid" if util == "中" else "utility-low")
        )
        pico = pick.get("pico", {})
        pico_html = ""
        if pico:
            pico_html = f"""
            <div class="pico-grid">
              <div class="pico-item"><span class="pico-label">P</span><span class="pico-text">{pico.get("population", "-")}</span></div>
              <div class="pico-item"><span class="pico-label">I</span><span class="pico-text">{pico.get("intervention", "-")}</span></div>
              <div class="pico-item"><span class="pico-label">C</span><span class="pico-text">{pico.get("comparison", "-")}</span></div>
              <div class="pico-item"><span class="pico-label">O</span><span class="pico-text">{pico.get("outcome", "-")}</span></div>
            </div>"""

        top_picks_html += f"""
        <div class="news-card featured">
          <div class="card-header">
            <span class="rank-badge">#{pick.get("rank", "")}</span>
            <span class="emoji-icon">{pick.get("emoji", "📄")}</span>
            <span class="{utility_class}">{util}實用性</span>
          </div>
          <h3>{pick.get("title_zh", pick.get("title_en", ""))}</h3>
          <p class="journal-source">{pick.get("journal", "")} · {pick.get("title_en", "")}</p>
          <p>{pick.get("summary", "")}</p>
          {pico_html}
          <div class="card-footer">
            {tags_html}
            <a href="{pick.get("url", "#")}" target="_blank">閱讀原文 →</a>
          </div>
        </div>"""

    all_papers_html = ""
    for paper in all_papers:
        tags_html = "".join(
            f'<span class="tag">{t}</span>' for t in paper.get("tags", [])
        )
        util = paper.get("clinical_utility", "中")
        utility_class = (
            "utility-high"
            if util == "高"
            else ("utility-mid" if util == "中" else "utility-low")
        )
        all_papers_html += f"""
        <div class="news-card">
          <div class="card-header-row">
            <span class="emoji-sm">{paper.get("emoji", "📄")}</span>
            <span class="{utility_class} utility-sm">{util}</span>
          </div>
          <h3>{paper.get("title_zh", paper.get("title_en", ""))}</h3>
          <p class="journal-source">{paper.get("journal", "")}</p>
          <p>{paper.get("summary", "")}</p>
          <div class="card-footer">
            {tags_html}
            <a href="{paper.get("url", "#")}" target="_blank">PubMed →</a>
          </div>
        </div>"""

    keywords_html = "".join(f'<span class="keyword">{k}</span>' for k in keywords)
    topic_bars_html = ""
    if topic_dist:
        max_count = max(topic_dist.values()) if topic_dist else 1
        for topic, count in topic_dist.items():
            width_pct = int((count / max_count) * 100)
            topic_bars_html += f"""
            <div class="topic-row">
              <span class="topic-name">{topic}</span>
              <div class="topic-bar-bg"><div class="topic-bar" style="width:{width_pct}%"></div></div>
              <span class="topic-count">{count}</span>
            </div>"""

    total_count = len(top_picks) + len(all_papers)

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OCD Brain · 強迫症文獻日報 · {date_display}</title>
<meta name="description" content="{date_display} 強迫症 (OCD) 文獻日報，由 AI 自動彙整 PubMed 最新論文"/>
<style>
  :root {{ --bg: #f6f1e8; --surface: #fffaf2; --line: #d8c5ab; --text: #2b2118; --muted: #766453; --accent: #8c4f2b; --accent-soft: #ead2bf; --card-bg: color-mix(in srgb, var(--surface) 92%, white); }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: radial-gradient(circle at top, #fff6ea 0, var(--bg) 55%, #ead8c6 100%); color: var(--text); font-family: "Noto Sans TC", "PingFang TC", "Helvetica Neue", Arial, sans-serif; min-height: 100vh; overflow-x: hidden; }}
  .container {{ position: relative; z-index: 1; max-width: 880px; margin: 0 auto; padding: 60px 32px 80px; }}
  header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 52px; animation: fadeDown 0.6s ease both; }}
  .logo {{ width: 52px; height: 52px; border-radius: 14px; background: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 24px; flex-shrink: 0; box-shadow: 0 4px 20px rgba(140,79,43,0.25); }}
  .header-text h1 {{ font-size: 22px; font-weight: 700; color: var(--text); letter-spacing: -0.3px; }}
  .header-meta {{ display: flex; gap: 8px; margin-top: 6px; flex-wrap: wrap; align-items: center; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; letter-spacing: 0.3px; }}
  .badge-date {{ background: var(--accent-soft); border: 1px solid var(--line); color: var(--accent); }}
  .badge-count {{ background: rgba(140,79,43,0.06); border: 1px solid var(--line); color: var(--muted); }}
  .badge-source {{ background: transparent; color: var(--muted); font-size: 11px; padding: 0 4px; }}
  .summary-card {{ background: var(--card-bg); border: 1px solid var(--line); border-radius: 24px; padding: 28px 32px; margin-bottom: 32px; box-shadow: 0 20px 60px rgba(61,36,15,0.06); animation: fadeUp 0.5s ease 0.1s both; }}
  .summary-card h2 {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.6px; color: var(--accent); margin-bottom: 16px; }}
  .summary-text {{ font-size: 15px; line-height: 1.8; color: var(--text); }}
  .section {{ margin-bottom: 36px; animation: fadeUp 0.5s ease both; }}
  .section-title {{ display: flex; align-items: center; gap: 10px; font-size: 17px; font-weight: 700; color: var(--text); margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--line); }}
  .section-icon {{ width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; background: var(--accent-soft); }}
  .news-card {{ background: var(--card-bg); border: 1px solid var(--line); border-radius: 24px; padding: 22px 26px; margin-bottom: 12px; box-shadow: 0 8px 30px rgba(61,36,15,0.04); transition: background 0.2s, border-color 0.2s, transform 0.2s; }}
  .news-card:hover {{ transform: translateY(-2px); box-shadow: 0 12px 40px rgba(61,36,15,0.08); }}
  .news-card.featured {{ border-left: 3px solid var(--accent); }}
  .news-card.featured:hover {{ border-color: var(--accent); }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
  .rank-badge {{ background: var(--accent); color: #fff7f0; font-weight: 700; font-size: 12px; padding: 2px 8px; border-radius: 6px; }}
  .emoji-icon {{ font-size: 18px; }}
  .card-header-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .emoji-sm {{ font-size: 14px; }}
  .news-card h3 {{ font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 8px; line-height: 1.5; }}
  .journal-source {{ font-size: 12px; color: var(--accent); margin-bottom: 8px; opacity: 0.8; }}
  .news-card p {{ font-size: 13.5px; line-height: 1.75; color: var(--muted); }}
  .card-footer {{ margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  .tag {{ padding: 2px 9px; background: var(--accent-soft); border-radius: 999px; font-size: 11px; color: var(--accent); }}
  .news-card a {{ font-size: 12px; color: var(--accent); text-decoration: none; opacity: 0.7; margin-left: auto; }}
  .news-card a:hover {{ opacity: 1; }}
  .utility-high {{ color: #5a7a3a; font-size: 11px; font-weight: 600; padding: 2px 8px; background: rgba(90,122,58,0.1); border-radius: 4px; }}
  .utility-mid {{ color: #9f7a2e; font-size: 11px; font-weight: 600; padding: 2px 8px; background: rgba(159,122,46,0.1); border-radius: 4px; }}
  .utility-low {{ color: var(--muted); font-size: 11px; font-weight: 600; padding: 2px 8px; background: rgba(118,100,83,0.08); border-radius: 4px; }}
  .utility-sm {{ font-size: 10px; }}
  .pico-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; padding: 12px; background: rgba(255,253,249,0.8); border-radius: 14px; border: 1px solid var(--line); }}
  .pico-item {{ display: flex; gap: 8px; align-items: baseline; }}
  .pico-label {{ font-size: 10px; font-weight: 700; color: #fff7f0; background: var(--accent); padding: 2px 6px; border-radius: 4px; flex-shrink: 0; }}
  .pico-text {{ font-size: 12px; color: var(--muted); line-height: 1.4; }}
  .keywords-section {{ margin-bottom: 36px; }}
  .keywords {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
  .keyword {{ padding: 5px 14px; background: var(--accent-soft); border: 1px solid var(--line); border-radius: 20px; font-size: 12px; color: var(--accent); cursor: default; transition: background 0.2s; }}
  .keyword:hover {{ background: rgba(140,79,43,0.18); }}
  .topic-section {{ margin-bottom: 36px; }}
  .topic-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
  .topic-name {{ font-size: 13px; color: var(--muted); width: 120px; flex-shrink: 0; text-align: right; }}
  .topic-bar-bg {{ flex: 1; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }}
  .topic-bar {{ height: 100%; background: linear-gradient(90deg, var(--accent), #c47a4a); border-radius: 4px; transition: width 0.6s ease; }}
  .topic-count {{ font-size: 12px; color: var(--accent); width: 24px; }}
  .clinic-banner {{ margin-top: 48px; animation: fadeUp 0.5s ease 0.3s both; }}
  .clinic-links {{ display: flex; flex-direction: column; gap: 12px; }}
  .clinic-link {{ display: flex; align-items: center; gap: 14px; padding: 18px 24px; background: var(--card-bg); border: 1px solid var(--line); border-radius: 24px; text-decoration: none; color: var(--text); transition: all 0.2s; box-shadow: 0 8px 30px rgba(61,36,15,0.04); }}
  .clinic-link:hover {{ border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 12px 40px rgba(61,36,15,0.08); }}
  .clinic-icon {{ font-size: 28px; flex-shrink: 0; }}
  .clinic-info {{ flex: 1; }}
  .clinic-name {{ font-size: 15px; font-weight: 700; color: var(--text); }}
  .clinic-desc {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .clinic-arrow {{ font-size: 18px; color: var(--accent); font-weight: 700; }}
  footer {{ margin-top: 32px; padding-top: 22px; border-top: 1px solid var(--line); font-size: 11.5px; color: var(--muted); display: flex; justify-content: space-between; animation: fadeUp 0.5s ease 0.5s both; }}
  footer a {{ color: var(--muted); text-decoration: none; }}
  footer a:hover {{ color: var(--accent); }}
  @keyframes fadeDown {{ from {{ opacity: 0; transform: translateY(-16px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(16px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  @media (max-width: 600px) {{ .container {{ padding: 36px 18px 60px; }} .summary-card, .news-card {{ padding: 20px 18px; }} .pico-grid {{ grid-template-columns: 1fr; }} footer {{ flex-direction: column; gap: 6px; text-align: center; }} .topic-name {{ width: 80px; font-size: 11px; }} .clinic-links {{ gap: 8px; }} }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">🧠</div>
    <div class="header-text">
      <h1>OCD Brain · 強迫症文獻日報</h1>
      <div class="header-meta">
        <span class="badge badge-date">📅 {date_display}</span>
        <span class="badge badge-count">📊 {total_count} 篇文獻</span>
        <span class="badge badge-source">Powered by PubMed + Zhipu AI</span>
      </div>
    </div>
  </header>

  <div class="summary-card">
    <h2>📋 今日 OCD 文獻趨勢</h2>
    <p class="summary-text">{summary}</p>
  </div>

  {"<div class='section'><div class='section-title'><span class='section-icon'>⭐</span>今日精選 TOP Picks</div>" + top_picks_html + "</div>" if top_picks_html else ""}

  {"<div class='section'><div class='section-title'><span class='section-icon'>📚</span>其他值得關注的文獻</div>" + all_papers_html + "</div>" if all_papers_html else ""}

  {"<div class='topic-section section'><div class='section-title'><span class='section-icon'>📊</span>主題分佈</div>" + topic_bars_html + "</div>" if topic_bars_html else ""}

  {"<div class='keywords-section section'><div class='section-title'><span class='section-icon'>🏷️</span>關鍵字</div><div class='keywords'>" + keywords_html + "</div></div>" if keywords_html else ""}

  <div class="clinic-banner">
    <div class="clinic-links">
      <a href="https://www.leepsyclinic.com/" class="clinic-link" target="_blank">
        <span class="clinic-icon">🏥</span>
        <div class="clinic-info">
          <div class="clinic-name">李政洋身心診所首頁</div>
          <div class="clinic-desc">專業身心科診療服務</div>
        </div>
        <span class="clinic-arrow">→</span>
      </a>
      <a href="https://blog.leepsyclinic.com/" class="clinic-link" target="_blank">
        <span class="clinic-icon">📬</span>
        <div class="clinic-info">
          <div class="clinic-name">訂閱電子報</div>
          <div class="clinic-desc">定期接收最新身心健康資訊</div>
        </div>
        <span class="clinic-arrow">→</span>
      </a>
    </div>
  </div>

  <footer>
    <span>資料來源：PubMed · 分析模型：{model_used}</span>
    <span><a href="https://github.com/u8901006/ocd-brain">GitHub</a></span>
  </footer>
</div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate OCD daily report HTML")
    parser.add_argument("--input", required=True, help="Input papers JSON file")
    parser.add_argument("--output", required=True, help="Output HTML file")
    parser.add_argument(
        "--api-key", default=os.environ.get("ZHIPU_API_KEY", ""), help="Zhipu API key"
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "[ERROR] No API key provided. Set ZHIPU_API_KEY env var or use --api-key",
            file=sys.stderr,
        )
        sys.exit(1)

    papers_data = load_papers(args.input)
    excluded_pmids = collect_excluded_pmids(days=7)
    if papers_data and papers_data.get("papers") and excluded_pmids:
        before = len(papers_data["papers"])
        papers_data["papers"] = [
            p for p in papers_data["papers"] if p.get("pmid", "") not in excluded_pmids
        ]
        papers_data["count"] = len(papers_data["papers"])
        print(
            f"[INFO] Dedup: {before} -> {len(papers_data['papers'])} papers (removed {before - len(papers_data['papers'])} duplicates)",
            file=sys.stderr,
        )
    if not papers_data or not papers_data.get("papers"):
        print("[WARN] No papers found, generating empty report", file=sys.stderr)
        tz_taipei = timezone(timedelta(hours=8))
        analysis = {
            "date": datetime.now(tz_taipei).strftime("%Y-%m-%d"),
            "market_summary": "今日 PubMed 暫無新的強迫症 (OCD) 相關文獻更新。請明天再查看。",
            "top_picks": [],
            "all_papers": [],
            "keywords": [],
            "topic_distribution": {},
        }
    else:
        analysis = analyze_papers(args.api_key, papers_data)
        if not analysis:
            print("[ERROR] Analysis failed, cannot generate report", file=sys.stderr)
            sys.exit(1)

    html = generate_html(analysis)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Report saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
