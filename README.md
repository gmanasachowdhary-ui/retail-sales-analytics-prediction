# Retail Sales Analytics & Prediction

End-to-end applied data science project using **synthetic retail transactions**.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/analyze.py
```
Outputs are saved in `reports/` and `reports/figures/`.

## Dataset
1,800 generated transactions from 2024-01-01 through 2025-06-30. Fields: transaction ID, date, region, category, product, quantity, unit price, discount rate, and unit cost. It is not actual retailer/customer data.

## Project workflow
- Validate records and engineer revenue/profit metrics
- Analyze category, region, and monthly trends
- Build a daily revenue Random Forest baseline with lag and rolling features
- Use chronological 80/20 holdout; report MAE and RMSE versus previous-day baseline
- Create visualizations and a summary with recommendations

Model results are illustrative; incorporate holidays, promotions, stockouts, and marketing before operational use.

## Structure
`data/` dataset · `src/analyze.py` pipeline · `notebooks/analysis_walkthrough.md` walkthrough · `reports/` generated summary and charts
