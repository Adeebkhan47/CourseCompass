"""
recommender.py
Rule-based Course Recommendation Engine with:
- Zero self-recommendation (excludes user-selected courses)
- Multi-rule evidence aggregation
- Documented ranking strategies (Confidence, Lift, Hybrid)
- Dynamic "Why this recommendation?" rationales
- What-If analysis for adding/removing courses
"""

import math
import pandas as pd
from typing import List, Dict, Set, Any, Optional

class CourseRecommender:
    def __init__(self, rules_df: pd.DataFrame):
        """
        Initializes recommender with generated association rules DataFrame.
        """
        self.rules_df = rules_df

    def get_recommendations(
        self,
        user_basket: List[str],
        ranking_strategy: str = "Confidence-First",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generates course recommendations given a user's selected basket.
        
        Guarantees:
        - NEVER recommends an item already selected in user_basket.
        - Gathers all matching rules (where antecedent is a subset of user_basket).
        - Ranks recommendations according to ranking_strategy:
            * 'Confidence-First': max(confidence)
            * 'Lift-First': max(lift)
            * 'Hybrid': max_conf * sqrt(max_lift) * (1 + 0.05 * (rule_count - 1))
        """
        if not user_basket or self.rules_df.empty:
            return []

        user_basket_set = set(user_basket)
        candidate_evidence: Dict[str, List[Dict[str, Any]]] = {}

        for _, rule in self.rules_df.iterrows():
            ant_set = set(rule['antecedent'])
            con_list = rule['consequent']

            # Check if user has enrolled in all items in antecedent
            if ant_set.issubset(user_basket_set):
                for candidate in con_list:
                    # CRITICAL: Exclude items already in user's basket
                    if candidate in user_basket_set:
                        continue

                    if candidate not in candidate_evidence:
                        candidate_evidence[candidate] = []

                    candidate_evidence[candidate].append({
                        "rule_repr": rule['rule_repr'],
                        "antecedent": rule['antecedent'],
                        "consequent": rule['consequent'],
                        "antecedent_str": rule['antecedent_str'],
                        "consequent_str": rule['consequent_str'],
                        "support": rule['support'],
                        "confidence": rule['confidence'],
                        "lift": rule['lift'],
                        "leverage": rule.get('leverage', 0.0),
                        "conviction": rule.get('conviction', 1.0),
                        "nl_explanation": rule['nl_explanation']
                    })

        if not candidate_evidence:
            return []

        recommendations = []
        for course, evidence_list in candidate_evidence.items():
            # Sort evidence by confidence and lift
            sorted_evidence = sorted(evidence_list, key=lambda x: (x['confidence'], x['lift']), reverse=True)
            best_rule = sorted_evidence[0]

            max_confidence = max(e['confidence'] for e in evidence_list)
            max_lift = max(e['lift'] for e in evidence_list)
            max_support = max(e['support'] for e in evidence_list)
            rule_count = len(evidence_list)

            # Compute score based on selected strategy
            if ranking_strategy == "Confidence-First":
                rank_score = max_confidence
            elif ranking_strategy == "Lift-First":
                rank_score = max_lift
            else:  # Hybrid
                rank_score = max_confidence * math.sqrt(max_lift) * (1.0 + 0.05 * min(rule_count - 1, 4))

            # Dynamic rationale construction
            primary_ant = best_rule['antecedent_str']
            primary_conf = best_rule['confidence'] * 100
            primary_lift = best_rule['lift']

            rationale = (
                f"Recommended because learners enrolled in [{primary_ant}] "
                f"also chose this course with {primary_conf:.1f}% confidence and {primary_lift:.2f}x lift. "
                f"({rule_count} supporting rule{'s' if rule_count > 1 else ''} found)."
            )

            recommendations.append({
                "course": course,
                "rank_score": rank_score,
                "max_confidence": max_confidence,
                "max_lift": max_lift,
                "max_support": max_support,
                "rule_count": rule_count,
                "primary_rule": best_rule['rule_repr'],
                "rationale": rationale,
                "all_evidence": sorted_evidence
            })

        # Sort recommendations by computed rank score descending
        recommendations.sort(key=lambda x: x['rank_score'], reverse=True)
        return recommendations[:top_k]

    def what_if_analysis(
        self,
        current_basket: List[str],
        removed_item: str,
        ranking_strategy: str = "Confidence-First"
    ) -> Dict[str, Any]:
        """
        Analyzes the impact on recommendations when a course is removed from the user's basket.
        Returns:
            - current_recs: list of courses recommended before removal
            - counterfactual_recs: list of courses recommended after removal
            - lost_courses: courses that were recommended before but lost after removal
            - newly_added: courses that appeared after removal (e.g., if freed from basket exclusion)
        """
        current_recs = self.get_recommendations(current_basket, ranking_strategy, top_k=10)
        counterfactual_basket = [c for c in current_basket if c != removed_item]
        counterfactual_recs = self.get_recommendations(counterfactual_basket, ranking_strategy, top_k=10)

        current_courses = [r['course'] for r in current_recs]
        cf_courses = [r['course'] for r in counterfactual_recs]

        lost = [c for c in current_courses if c not in cf_courses]
        added = [c for c in cf_courses if c not in current_courses]

        return {
            "current_basket": current_basket,
            "counterfactual_basket": counterfactual_basket,
            "removed_item": removed_item,
            "current_recs": current_recs,
            "counterfactual_recs": counterfactual_recs,
            "lost_courses": lost,
            "newly_added": added
        }
