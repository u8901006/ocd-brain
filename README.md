# OCD Brain

強迫症 (Obsessive-Compulsive Disorder) 文獻日報，每日自動更新。

## 運作方式

1. GitHub Actions 每日台北時間 11:00 自動執行
2. 從 PubMed 抓取最新 OCD 相關文獻
3. 使用 NVIDIA Nemotron 進行分析與摘要
4. 生成 HTML 日報並部署到 GitHub Pages

## 網站

https://u8901006.github.io/ocd-brain/

## 技術

- PubMed E-utilities API
- NVIDIA Nemotron 3 Super（fallback: Nemotron 3 Nano）
- GitHub Actions + GitHub Pages
- Python 3.12
