# Walkthrough
1. Load and validate data; check duplicates and missing values.
2. Compute revenue = quantity × unit price × (1 − discount rate); gross profit = revenue − quantity × unit cost.
3. Explore monthly, category, and region aggregates.
4. Aggregate revenue daily and create lag-1, lag-7, and prior-week rolling features.
5. Train on the first 80% of days and test on the final 20%.
6. Compare Random Forest with previous-day baseline using MAE/RMSE. Run `python src/analyze.py`.
