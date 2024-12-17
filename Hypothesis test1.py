import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data_for_predictions.csv')

data.head()

churned = data[data['churn'] == 1]
non_churned = data[data['churn'] == 0]

feature = 'off_peak_peak_var_max_monthly_diff'

t_stat, p_value = ttest_ind(churned[feature], non_churned[feature], nan_policy='omit')

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("There is a significant difference in price variability between churned and non-churned customers.")
else:
    print("There is no significant difference in price variability between churned and non-churned customers.")

plt.figure(figsize=(10, 6))
sns.kdeplot(churned[feature], label='Churned', fill=True, color='red', alpha=0.5)
sns.kdeplot(non_churned[feature], label='Non-Churned', fill=True, color='blue', alpha=0.5)
plt.title(f"Distribution of {feature} by Churn Status")
plt.xlabel(feature)
plt.ylabel("Density")
plt.legend()
plt.show()
