import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data_for_predictions.csv')

data.head()

churned = data[data['churn'] == 1]
non_churned = data[data['churn'] == 0]

from scipy.stats import shapiro

stat_churned, p_churned = shapiro(churned['tenure'])
stat_non_churned, p_non_churned = shapiro(non_churned['tenure'])

print(f"P-value for churned customers normality test: {p_churned}")
print(f"P-value for non-churned customers normality test: {p_non_churned}")

t_stat, p_value = ttest_ind(churned['tenure'], non_churned['tenure'], nan_policy='omit')

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("There is a significant difference in tenure between churned and non-churned customers.")
else:
    print("There is no significant difference in tenure between churned and non-churned customers.")

plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='tenure', data=data)
plt.title("Customer Tenure by Churn Status")
plt.xlabel("Churn Status (0 = Non-Churned, 1 = Churned)")
plt.ylabel("Customer Tenure (months)")
plt.show()
