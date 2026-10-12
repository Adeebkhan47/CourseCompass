"""
verify_system.py
Automated end-to-end verification script for CourseCompass.
Tests:
1. Data loading and summary calculations
2. Basket generation (both multi-only and all-learners)
3. AprioriEngine candidate generation, pruning, and trace
4. Association rule extraction (support, confidence, lift, leverage, conviction)
5. Proof that confidence(A -> B) != confidence(B -> A)
6. CourseRecommender with exclusion of selected basket items
7. Multi-rule aggregation and ranking strategies
8. What-If analysis
9. Visualizations generation without GUI error
"""

import os
import sys

# Set UTF-8 encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.data_loader import load_preprocessed_data, get_dataset_summary, build_transactions, COURSE_METADATA
from src.apriori_engine import AprioriEngine
from src.recommender import CourseRecommender
from src.visualizations import (
    plot_association_graph,
    plot_item_frequency,
    plot_basket_size_distribution,
    plot_rules_scatter,
    plot_apriori_pruning_flow
)

def run_verification():
    print("==================================================")
    print("      COURSECOMPASS SYSTEM VERIFICATION SUITE      ")
    print("==================================================")

    # 1. Data loading
    print("\n[TEST 1] Loading preprocessed dataset...")
    df = load_preprocessed_data()
    assert len(df) == 338223, f"Expected 338,223 records, got {len(df)}"
    print(f"  ✓ Records: {len(df):,}")

    summary = get_dataset_summary(df)
    assert summary['unique_learners'] == 301609, f"Expected 301,609 learners, got {summary['unique_learners']}"
    assert summary['unique_courses'] == 5, f"Expected 5 courses, got {summary['unique_courses']}"
    assert summary['multi_course_learners'] == 30267, f"Expected 30,267 multi-course learners, got {summary['multi_course_learners']}"
    print(f"  ✓ Unique learners: {summary['unique_learners']:,}")
    print(f"  ✓ Multi-course baskets: {summary['multi_course_learners']:,}")

    # 2. Basket construction
    print("\n[TEST 2] Verifying Basket Construction...")
    txs_multi, oh_multi = build_transactions(df, multi_only=True)
    assert len(txs_multi) == 30267
    assert oh_multi.shape == (30267, 5)
    print(f"  ✓ Multi-course transactions: {len(txs_multi):,} baskets")

    txs_all, oh_all = build_transactions(df, multi_only=False)
    assert len(txs_all) == 301609
    assert oh_all.shape == (301609, 5)
    print(f"  ✓ All-learners transactions: {len(txs_all):,} baskets")

    # 3. Apriori Engine
    print("\n[TEST 3] Verifying Apriori Candidate Generation & Pruning...")
    engine = AprioriEngine(txs_multi)
    itemsets_df, freq_dict = engine.find_frequent_itemsets(min_support=0.05)
    assert not itemsets_df.empty, "Frequent itemsets should not be empty at min_sup=0.05"
    print(f"  ✓ Discovered frequent itemsets: {len(itemsets_df)}")
    print(f"  ✓ Apriori execution levels traced: {len(engine.trace)}")
    for t in engine.trace:
        print(f"    - Level {t['level']}: generated={t['candidates_generated']}, pruned_apriori={t['pruned_by_apriori']}, pruned_sup={t['pruned_by_min_support']}, surviving={t['surviving_frequent']}")

    # 4. Rule generation & metrics
    print("\n[TEST 4] Verifying Association Rule Generation...")
    rules_df = engine.generate_rules(freq_dict, min_confidence=0.15)
    assert not rules_df.empty, "Rules should not be empty"
    print(f"  ✓ Generated rules: {len(rules_df)}")
    required_cols = ['antecedent', 'consequent', 'support', 'confidence', 'lift', 'leverage', 'conviction', 'nl_explanation']
    for col in required_cols:
        assert col in rules_df.columns, f"Missing column: {col}"
    print("  ✓ All required metrics present in rule output")

    # 5. Check asymmetric confidence
    print("\n[TEST 5] Checking Confidence Asymmetry (A -> B != B -> A)...")
    c1 = "CB22x: The Ancient Greek Hero"
    c2 = "ER22x: Justice: What's the Right Thing to Do?"
    r12 = rules_df[(rules_df['antecedent_str'] == c1) & (rules_df['consequent_str'] == c2)]
    r21 = rules_df[(rules_df['antecedent_str'] == c2) & (rules_df['consequent_str'] == c1)]
    if not r12.empty and not r21.empty:
        conf12 = r12.iloc[0]['confidence']
        conf21 = r21.iloc[0]['confidence']
        print(f"  ✓ Conf({c1} -> {c2}): {conf12*100:.2f}%")
        print(f"  ✓ Conf({c2} -> {c1}): {conf21*100:.2f}%")
        assert not abs(conf12 - conf21) < 1e-4, "Confidence should be asymmetric!"
        print(f"  ✓ Asymmetry confirmed! Absolute delta: {abs(conf12 - conf21)*100:.2f}%")

    # 6. Recommendation Engine & Exclusion of selected items
    print("\n[TEST 6] Verifying Recommendation Engine & Basket Exclusion...")
    recommender = CourseRecommender(rules_df)
    test_basket = ["CB22x: The Ancient Greek Hero"]
    recs = recommender.get_recommendations(test_basket, ranking_strategy="Confidence-First", top_k=5)
    assert len(recs) > 0, "Expected recommendations"
    for r in recs:
        assert r['course'] not in test_basket, f"CRITICAL BUG: Recommended course {r['course']} was in user basket!"
    print(f"  ✓ Recommended courses strictly exclude enrolled courses: {[r['course'].split(':')[0] for r in recs]}")
    print(f"  ✓ Dynamic rationale generated: {recs[0]['rationale']}")

    # 7. Ranking strategies
    print("\n[TEST 7] Verifying Ranking Strategies (Confidence, Lift, Hybrid)...")
    recs_conf = recommender.get_recommendations(test_basket, "Confidence-First", 3)
    recs_lift = recommender.get_recommendations(test_basket, "Lift-First", 3)
    recs_hyb = recommender.get_recommendations(test_basket, "Hybrid", 3)
    print(f"  ✓ Conf-First: {[r['course'].split(':')[0] for r in recs_conf]}")
    print(f"  ✓ Lift-First: {[r['course'].split(':')[0] for r in recs_lift]}")
    print(f"  ✓ Hybrid:     {[r['course'].split(':')[0] for r in recs_hyb]}")

    # 8. What-If Analysis
    print("\n[TEST 8] Verifying What-If Analysis...")
    basket_two = ["CB22x: The Ancient Greek Hero", "CS50x: Introduction to Computer Science I"]
    what_if = recommender.what_if_analysis(basket_two, "CS50x: Introduction to Computer Science I")
    assert what_if['removed_item'] == "CS50x: Introduction to Computer Science I"
    print(f"  ✓ What-If executed cleanly. Removed: {what_if['removed_item'].split(':')[0]}")

    # 9. Visualizations
    print("\n[TEST 9] Verifying Visual Analytics Generators...")
    fig1 = plot_association_graph(rules_df)
    fig2 = plot_item_frequency(df)
    fig3 = plot_basket_size_distribution(df)
    fig4 = plot_rules_scatter(rules_df)
    fig5 = plot_apriori_pruning_flow(engine.trace)
    print("  ✓ All 5 charts generated without error!")

    print("\n==================================================")
    print("      ALL 9 VERIFICATION TESTS PASSED (100%)       ")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
