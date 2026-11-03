# Analysis summary
## Dataset validation
- Transactions: **1,800**
- Date range: **2024-01-01 – 2025-06-30**
- Categories: **4** | Regions: **4**
- Missing values: **0**
- Duplicate transaction IDs: **0**

## Commercial metrics
- Total revenue: **109,194.99**
- Gross profit: **34,512.78**
- Gross margin: **31.6%**
- Highest revenue category: **Electronics**
- Highest revenue region: **North**

## Forecast evaluation (chronological holdout)
- Random Forest MAE: **127.04** | RMSE: **157.56**
- Previous-day baseline MAE: **160.32** | RMSE: **213.12**

## Recommendations
1. Use category/region contribution to guide inventory reviews, then validate with operational data.
2. Add holidays, promotions, stockouts, and marketing signals before operational forecasting.
3. Monitor errors over time and compare against simple baselines.

*Synthetic dataset; results are instructional and do not represent real business performance.*
