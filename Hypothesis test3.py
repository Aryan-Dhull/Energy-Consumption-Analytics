import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

df = pd.read_csv("data_for_predictions.csv") 

churned = df[df['churn'] == 1]
non_churned = df[df['churn'] == 0]

price_variability_features = [
    'off_peak_peak_var_max_monthly_diff',
    'peak_mid_peak_var_max_monthly_diff',
    'off_peak_mid_peak_var_max_monthly_diff'
]

for feature in price_variability_features:
    churned_data = churned[feature].dropna()
    non_churned_data = non_churned[feature].dropna()

    t_stat, p_value = ttest_ind(churned_data, non_churned_data, equal_var=False)

    print(f"Feature: {feature}")
    print(f"T-test statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("=> Significant difference between churned and non-churned customers.")
    else:
        print("=> No significant difference between churned and non-churned customers.")
    print("-" * 50)
