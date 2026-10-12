import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

data_path = os.path.join("Dataset", "HXPC13_DI_v3_11-13-2019.csv")
df = pd.read_csv(data_path, low_memory=False)

# Let's test two transaction definitions:
# Definition A: All learners (301,609 transactions)
# Definition B: Multi-course learners only (30,267 transactions, baskets with >= 2 courses)

print("=== DEFINITION A: ALL LEARNERS (N=301,609) ===")
# Create one-hot encoded basket
basket_all = pd.crosstab(df['userid_DI'], df['course_id']).astype(bool)
print("Basket shape:", basket_all.shape)

frequent_itemsets_all = apriori(basket_all, min_support=0.005, use_colnames=True)
print(f"Frequent itemsets at min_support=0.005: {len(frequent_itemsets_all)}")
print(frequent_itemsets_all.sort_values(by='support', ascending=False))

rules_all = association_rules(frequent_itemsets_all, metric="confidence", min_threshold=0.05)
print(f"\nRules at min_support=0.005, min_confidence=0.05: {len(rules_all)}")
if len(rules_all) > 0:
    for idx, r in rules_all.head(15).iterrows():
        ant = list(r['antecedents'])
        con = list(r['consequents'])
        print(f"{ant} -> {con} | Sup: {r['support']:.4f}, Conf: {r['confidence']:.4f}, Lift: {r['lift']:.4f}")

print("\n=== DEFINITION B: MULTI-COURSE LEARNERS ONLY (N=30,267) ===")
basket_multi = basket_all[basket_all.sum(axis=1) >= 2]
print("Multi-basket shape:", basket_multi.shape)

frequent_itemsets_multi = apriori(basket_multi, min_support=0.05, use_colnames=True)
print(f"Frequent itemsets at min_support=0.05: {len(frequent_itemsets_multi)}")
print(frequent_itemsets_multi.sort_values(by='support', ascending=False))

rules_multi = association_rules(frequent_itemsets_multi, metric="confidence", min_threshold=0.15)
print(f"\nRules at min_support=0.05, min_confidence=0.15: {len(rules_multi)}")
if len(rules_multi) > 0:
    for idx, r in rules_multi.head(20).iterrows():
        ant = list(r['antecedents'])
        con = list(r['consequents'])
        print(f"{ant} -> {con} | Sup: {r['support']:.4f}, Conf: {r['confidence']:.4f}, Lift: {r['lift']:.4f}")
