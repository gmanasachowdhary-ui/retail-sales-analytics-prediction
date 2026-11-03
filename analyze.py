from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"retail_transactions.csv"
OUT=ROOT/"reports"; FIG=OUT/"figures"; FIG.mkdir(parents=True,exist_ok=True)
def main():
    df=pd.read_csv(DATA,parse_dates=["date"])
    required={"transaction_id","date","region","category","product","quantity","unit_price","discount_rate","unit_cost"}
    if required-set(df.columns): raise ValueError(f"Missing columns: {required-set(df.columns)}")
    if df.transaction_id.duplicated().any() or df.isna().any().any(): raise ValueError("Duplicate IDs or missing data detected")
    if (df[["quantity","unit_price","unit_cost"]]<0).any().any(): raise ValueError("Negative quantity/price/cost")
    df["revenue"]=df.quantity*df.unit_price*(1-df.discount_rate)
    df["gross_profit"]=df.revenue-df.quantity*df.unit_cost
    daily=df.groupby("date",as_index=False).agg(revenue=("revenue","sum")).set_index("date").asfreq("D",fill_value=0).reset_index()
    daily["dow"]=daily.date.dt.dayofweek; daily["month"]=daily.date.dt.month
    daily["lag_1"]=daily.revenue.shift(1); daily["lag_7"]=daily.revenue.shift(7)
    daily["rolling_7"]=daily.revenue.shift(1).rolling(7).mean()
    md=daily.dropna().reset_index(drop=True); split=int(len(md)*.8); train,test=md.iloc[:split],md.iloc[split:]
    features=["dow","month","lag_1","lag_7","rolling_7"]
    model=RandomForestRegressor(n_estimators=250,max_depth=8,min_samples_leaf=3,random_state=42)
    model.fit(train[features],train.revenue); pred=model.predict(test[features]); naive=test.lag_1.to_numpy()
    mae=mean_absolute_error(test.revenue,pred); rmse=np.sqrt(mean_squared_error(test.revenue,pred))
    nmae=mean_absolute_error(test.revenue,naive); nrmse=np.sqrt(mean_squared_error(test.revenue,naive))
    monthly=df.set_index("date").resample("MS").revenue.sum()
    plt.figure(figsize=(10,5)); monthly.plot(marker="o"); plt.title("Monthly Revenue"); plt.xlabel("Month"); plt.ylabel("Revenue"); plt.tight_layout(); plt.savefig(FIG/"monthly_revenue.png",dpi=160); plt.close()
    cat=df.groupby("category").revenue.sum().sort_values()
    plt.figure(figsize=(8,5)); cat.plot(kind="barh"); plt.title("Revenue by Category"); plt.xlabel("Revenue"); plt.tight_layout(); plt.savefig(FIG/"revenue_by_category.png",dpi=160); plt.close()
    plt.figure(figsize=(10,5)); plt.plot(test.date,test.revenue,label="Actual"); plt.plot(test.date,pred,label="Random Forest"); plt.plot(test.date,naive,label="Previous-day baseline"); plt.title("Daily Revenue: Test Period"); plt.xlabel("Date"); plt.ylabel("Revenue"); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"forecast_test.png",dpi=160); plt.close()
    summary=f"""# Analysis summary
## Dataset validation
- Transactions: **{len(df):,}**
- Date range: **{df.date.min().date()} – {df.date.max().date()}**
- Categories: **{df.category.nunique()}** | Regions: **{df.region.nunique()}**
- Missing values: **{int(df.isna().sum().sum())}**
- Duplicate transaction IDs: **{int(df.transaction_id.duplicated().sum())}**

## Commercial metrics
- Total revenue: **{df.revenue.sum():,.2f}**
- Gross profit: **{df.gross_profit.sum():,.2f}**
- Gross margin: **{df.gross_profit.sum()/df.revenue.sum():.1%}**
- Highest revenue category: **{cat.index[-1]}**
- Highest revenue region: **{df.groupby('region').revenue.sum().idxmax()}**

## Forecast evaluation (chronological holdout)
- Random Forest MAE: **{mae:,.2f}** | RMSE: **{rmse:,.2f}**
- Previous-day baseline MAE: **{nmae:,.2f}** | RMSE: **{nrmse:,.2f}**

## Recommendations
1. Use category/region contribution to guide inventory reviews, then validate with operational data.
2. Add holidays, promotions, stockouts, and marketing signals before operational forecasting.
3. Monitor errors over time and compare against simple baselines.

*Synthetic dataset; results are instructional and do not represent real business performance.*
"""
    (OUT/"summary.md").write_text(summary); print(summary)
if __name__=="__main__": main()
