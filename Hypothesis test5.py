import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data_for_predictions.csv')

data.head()

channel_churn_table = pd.crosstab(data['channel_sales'], data['churn'])

origin_churn_table = pd.crosstab(data['origin'], data['churn'])

chi2_stat, p_value, dof, expected = chi2_contingency(channel_churn_table)

print(f"Chi-Square Statistic (Channel): {chi2_stat}")
print(f"P-value (Channel): {p_value}")

alpha = 0.05
if p_value < alpha:
    print("There is a significant association between channel and churn.")
else:
    print("There is no significant association between channel and churn.")

chi2_stat, p_value, dof, expected = chi2_contingency(origin_churn_table)

print(f"Chi-Square Statistic (Origin): {chi2_stat}")
print(f"P-value (Origin): {p_value}")

if p_value < alpha:
    print("There is a significant association between origin and churn.")
else:
    print("There is no significant association between origin and churn.")


plt.figure(figsize=(10, 6))
sns.heatmap(channel_churn_table, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Churn Count by Channel")
plt.xlabel("Churn Status")
plt.ylabel("Channel")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(origin_churn_table, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Churn Count by Origin")
plt.xlabel("Churn Status")
plt.ylabel("Origin")
plt.show()
