# OCD Brain

強迫症 (Obsessive-Compulsive Disorder) 文獻日報，每日自動更新。

## 運作方式

1. GitHub Actions 每日台北時間 11:00 自動執行
2. 從 PubMed 抓取最新 OCD 相關文獻
3. 使用 Zhipu AI (GLM-5-Turbo) 進行分析與摘要
4. 生成 HTML 日報並部署到 GitHub Pages

## 網站

https://u8901006.github.io/ocd-brain/

## 技術

- PubMed E-utilities API
- Zhipu AI (GLM-5-Turbo → GLM-4.7 → GLM-4.7-Flash fallback)
- GitHub Actions + GitHub Pages
- Python 3.12
