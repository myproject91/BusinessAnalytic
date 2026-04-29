import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def run_statistical_analysis(df: pd.DataFrame, numeric_cols: list) -> dict:
    if not numeric_cols:
        return {}

    result = {}
    subset = df[numeric_cols].copy()

    result['descriptive'] = subset.describe().round(4).to_dict()

    result['distribution'] = {}
    for col in numeric_cols:
        series = subset[col].dropna()
        result['distribution'][col] = {
            'skewness': round(float(series.skew()), 4),
            'kurtosis': round(float(series.kurtosis()), 4),
            'is_normal': bool(
                scipy_stats.normaltest(series).pvalue > 0.05
                if len(series) >= 8 else False
            )
        }

    if len(numeric_cols) >= 2:
        corr = subset.corr(method='pearson').round(4)
        result['correlation'] = corr.to_dict()
        high_corr = []
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i+1:]:
                val = corr.loc[col_a, col_b]
                if abs(val) >= 0.7:
                    high_corr.append({
                        'col_a': col_a,
                        'col_b': col_b,
                        'correlation': round(float(val), 4)
                    })
        result['high_correlation_pairs'] = high_corr

    return result


def detect_anomalies(df: pd.DataFrame, numeric_cols: list) -> dict:
    if not numeric_cols:
        return {}

    anomalies = {}
    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        flagged = df[mask][[col]].copy()

        if flagged.empty:
            continue

        anomalies[col] = {
            'lower_bound': round(float(lower), 4),
            'upper_bound': round(float(upper), 4),
            'count': int(flagged.shape[0]),
            'percent': round(flagged.shape[0] / len(df) * 100, 2),
            'records': [
                {'index': int(i), 'value': round(float(v), 4)}
                for i, v in flagged[col].items()
            ]
        }

    return anomalies


def run_category_analysis(df: pd.DataFrame, cat_cols: list, numeric_cols: list) -> dict:
    if not cat_cols or not numeric_cols:
        return {}

    result = {}
    for cat in cat_cols:
        result[cat] = {}
        for num in numeric_cols:
            grouped = df.groupby(cat)[num].agg(
                count='count', mean='mean', median='median',
                std='std', min='min', max='max', sum='sum'
            ).round(4)
            grouped = grouped.sort_values('mean', ascending=False)
            result[cat][num] = grouped.to_dict(orient='index')

    return result
