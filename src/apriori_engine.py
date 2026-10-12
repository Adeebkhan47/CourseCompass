"""
apriori_engine.py
Comprehensive Apriori algorithm implementation and rule generator for CourseCompass.
Includes:
- Candidate generation (join step)
- Apriori pruning (subset check)
- Support counting and frequent itemset generation
- Detailed step-by-step execution trace
- Association rule extraction with Support, Confidence, Lift, Leverage, Conviction
- Natural language explanation generation
- Parity mode with mlxtend
"""

import itertools
import math
import pandas as pd
from typing import List, Set, Dict, Tuple, Any

class AprioriEngine:
    def __init__(self, transactions: List[Set[str]]):
        self.transactions = transactions
        self.num_transactions = len(transactions)
        self.trace: List[Dict[str, Any]] = []

    def get_support(self, itemset: frozenset) -> float:
        """Computes support for an itemset."""
        if self.num_transactions == 0:
            return 0.0
        count = sum(1 for t in self.transactions if itemset.issubset(t))
        return count / self.num_transactions

    def _generate_candidates_k(self, prev_frequent: List[frozenset], k: int) -> Tuple[List[frozenset], int]:
        """
        Generates candidate k-itemsets C_k from frequent (k-1)-itemsets L_{k-1}.
        Uses the F_{k-1} x F_{k-1} join procedure and applies Apriori subset pruning.
        Returns:
            - candidates: valid candidate itemsets of size k
            - pruned_count: count of candidate itemsets pruned by Apriori property
        """
        candidates = []
        pruned_count = 0
        prev_list = sorted([sorted(list(itemset)) for itemset in prev_frequent])
        prev_set = set(prev_frequent)

        n = len(prev_list)
        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = prev_list[i]
                itemset2 = prev_list[j]

                # Join condition: first k-2 items must match
                if itemset1[:k - 2] == itemset2[:k - 2]:
                    candidate = frozenset(itemset1).union(frozenset(itemset2))
                    if len(candidate) == k:
                        # Apriori pruning: all (k-1) subsets must be frequent
                        has_infrequent_subset = False
                        for subset in itertools.combinations(candidate, k - 1):
                            if frozenset(subset) not in prev_set:
                                has_infrequent_subset = True
                                break
                        
                        if has_infrequent_subset:
                            pruned_count += 1
                        else:
                            if candidate not in candidates:
                                candidates.append(candidate)
        return candidates, pruned_count

    def find_frequent_itemsets(self, min_support: float) -> Tuple[pd.DataFrame, Dict[frozenset, float]]:
        """
        Executes Apriori workflow to discover all frequent itemsets with support >= min_support.
        Records step-by-step candidate generation and pruning details in self.trace.
        """
        self.trace = []
        item_counts: Dict[str, int] = {}
        for t in self.transactions:
            for item in t:
                item_counts[item] = item_counts.get(item, 0) + 1

        # Level 1
        l1: Dict[frozenset, float] = {}
        c1_total = len(item_counts)
        for item, count in item_counts.items():
            sup = count / self.num_transactions
            if sup >= min_support:
                l1[frozenset([item])] = sup

        self.trace.append({
            "level": 1,
            "candidates_generated": c1_total,
            "pruned_by_apriori": 0,
            "pruned_by_min_support": c1_total - len(l1),
            "surviving_frequent": len(l1),
            "itemsets": list(l1.keys())
        })

        frequent_dict: Dict[frozenset, float] = dict(l1)
        current_l = list(l1.keys())
        k = 2

        while current_l:
            candidates_k, pruned_by_apriori = self._generate_candidates_k(current_l, k)
            if not candidates_k:
                break

            # Count support for candidates
            lk: Dict[frozenset, float] = {}
            cand_counts = {c: 0 for c in candidates_k}
            for t in self.transactions:
                for c in candidates_k:
                    if c.issubset(t):
                        cand_counts[c] += 1

            for c, count in cand_counts.items():
                sup = count / self.num_transactions
                if sup >= min_support:
                    lk[c] = sup

            pruned_by_support = len(candidates_k) - len(lk)
            self.trace.append({
                "level": k,
                "candidates_generated": len(candidates_k) + pruned_by_apriori,
                "pruned_by_apriori": pruned_by_apriori,
                "pruned_by_min_support": pruned_by_support,
                "surviving_frequent": len(lk),
                "itemsets": list(lk.keys())
            })

            frequent_dict.update(lk)
            current_l = list(lk.keys())
            k += 1

        # Create structured DataFrame
        rows = []
        for itemset, sup in frequent_dict.items():
            rows.append({
                "itemset": sorted(list(itemset)),
                "itemset_str": ", ".join(sorted(list(itemset))),
                "length": len(itemset),
                "support": sup,
                "support_count": int(round(sup * self.num_transactions))
            })

        df_itemsets = pd.DataFrame(rows)
        if not df_itemsets.empty:
            df_itemsets = df_itemsets.sort_values(by=["length", "support"], ascending=[True, False]).reset_index(drop=True)
        else:
            df_itemsets = pd.DataFrame(columns=["itemset", "itemset_str", "length", "support", "support_count"])

        return df_itemsets, frequent_dict

    def generate_rules(self, frequent_dict: Dict[frozenset, float], min_confidence: float) -> pd.DataFrame:
        """
        Generates association rules from frequent itemsets:
        For every itemset I with length >= 2, generates all non-empty proper subsets A -> B (where B = I - A).
        Computes support, confidence, lift, leverage, conviction, and natural language explanations.
        """
        rules = []

        for itemset, itemset_sup in frequent_dict.items():
            k = len(itemset)
            if k < 2:
                continue

            itemset_list = list(itemset)
            # Generate all non-empty proper subsets
            for i in range(1, k):
                for antecedent_tuple in itertools.combinations(itemset_list, i):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    ant_sup = frequent_dict.get(antecedent)
                    if ant_sup is None:
                        ant_sup = self.get_support(antecedent)

                    con_sup = frequent_dict.get(consequent)
                    if con_sup is None:
                        con_sup = self.get_support(consequent)

                    if ant_sup == 0:
                        continue

                    confidence = itemset_sup / ant_sup

                    if confidence >= min_confidence:
                        # Lift = Confidence / P(B)
                        lift = confidence / con_sup if con_sup > 0 else 0.0
                        # Leverage = P(A & B) - P(A)*P(B)
                        leverage = itemset_sup - (ant_sup * con_sup)
                        # Conviction = (1 - P(B)) / (1 - Confidence)
                        if math.isclose(confidence, 1.0):
                            conviction = float('inf')
                        elif con_sup < 1.0:
                            conviction = (1.0 - con_sup) / (1.0 - confidence)
                        else:
                            conviction = 1.0

                        ant_sorted = sorted(list(antecedent))
                        con_sorted = sorted(list(consequent))

                        ant_str = " + ".join(ant_sorted)
                        con_str = " + ".join(con_sorted)
                        rule_repr = f"{{{ant_str}}} → {{{con_str}}}"

                        # Generate natural language explanation
                        nl_text = (
                            f"Learners who enrolled in [{ant_str}] have a "
                            f"{confidence*100:.1f}% chance of also taking [{con_str}] "
                            f"({lift:.2f}x higher than baseline enrollment rate)."
                        )

                        rules.append({
                            "antecedent": ant_sorted,
                            "consequent": con_sorted,
                            "antecedent_str": ant_str,
                            "consequent_str": con_str,
                            "rule_repr": rule_repr,
                            "support": itemset_sup,
                            "confidence": confidence,
                            "lift": lift,
                            "leverage": leverage,
                            "conviction": conviction,
                            "antecedent_support": ant_sup,
                            "consequent_support": con_sup,
                            "rule_length": len(itemset),
                            "nl_explanation": nl_text
                        })

        df_rules = pd.DataFrame(rules)
        if not df_rules.empty:
            df_rules = df_rules.sort_values(by=["confidence", "lift", "support"], ascending=[False, False, False]).reset_index(drop=True)
        else:
            df_rules = pd.DataFrame(columns=[
                "antecedent", "consequent", "antecedent_str", "consequent_str", "rule_repr",
                "support", "confidence", "lift", "leverage", "conviction",
                "antecedent_support", "consequent_support", "rule_length", "nl_explanation"
            ])

        return df_rules

def run_mlxtend_apriori(one_hot_df: pd.DataFrame, min_support: float, min_confidence: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper to run mlxtend for parity verification."""
    from mlxtend.frequent_patterns import apriori as mlx_apriori, association_rules as mlx_rules
    
    freq_items = mlx_apriori(one_hot_df, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return freq_items, pd.DataFrame()
    
    rules = mlx_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    return freq_items, rules
