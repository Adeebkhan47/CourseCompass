"""
app.py
CourseCompass: Association Rule Mining–Based Course Recommendation System
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import time

from src.data_loader import (
    load_preprocessed_data,
    get_dataset_summary,
    build_transactions,
    COURSE_METADATA
)
from src.apriori_engine import AprioriEngine
from src.recommender import CourseRecommender
from src.visualizations import (
    plot_association_graph,
    plot_item_frequency,
    plot_basket_size_distribution,
    plot_rules_scatter,
    plot_apriori_pruning_flow
)

# Page configuration
st.set_page_config(
    page_title="CourseCompass | Course Recommendation Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and modern typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }
    
    .main-header p {
        margin: 8px 0 0 0;
        font-size: 1.05rem;
        color: #C7D2FE;
        font-weight: 400;
    }
    
    .metric-card {
        background: #1E222D;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #60A5FA;
        margin-top: 4px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .rule-card {
        background: #181E29;
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-top: 1px solid #262F3E;
        border-right: 1px solid #262F3E;
        border-bottom: 1px solid #262F3E;
    }
    
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-primary { background: #312E81; color: #C7D2FE; }
    .badge-success { background: #064E3B; color: #6EE7B7; }
    .badge-warning { background: #78350F; color: #FCD34D; }
    .badge-info { background: #0C4A6E; color: #7DD3FC; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data(show_spinner=False)
def load_data():
    return load_preprocessed_data()

# Header banner
st.markdown("""
<div class="main-header">
    <h1>CourseCompass 🧭</h1>
    <p>Association Rule Mining–Based Course Recommendation System | HarvardX Enrollment Analytics</p>
</div>
""", unsafe_allow_html=True)

try:
    df_enrollments = load_data()
    summary = get_dataset_summary(df_enrollments)
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please run `py scripts/prepare_data.py` first.")
    st.stop()

# Sidebar: Global controls & settings
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/compass.png", width=64)
    st.title("Control Panel")
    
    st.subheader("1. Population Definition")
    population_mode = st.radio(
        "Transaction Basket Scope:",
        options=["Multi-Course Learners (Basket ≥ 2)", "All Registered Learners (All Baskets)"],
        index=0,
        help="Multi-Course Learners (N=30,267) represent learners enrolled in 2+ courses, ideal for Market Basket Analysis. All Learners (N=301,609) includes learners taking only 1 course."
    )
    is_multi_only = (population_mode == "Multi-Course Learners (Basket ≥ 2)")

    st.subheader("2. Apriori Thresholds")
    if is_multi_only:
        default_support = 0.05
        min_sup_slider = st.slider("Minimum Support (min_sup)", min_value=0.01, max_value=0.40, value=0.05, step=0.01, format="%.2f")
        default_conf = 0.20
        min_conf_slider = st.slider("Minimum Confidence (min_conf)", min_value=0.05, max_value=0.90, value=0.20, step=0.05, format="%.2f")
    else:
        min_sup_slider = st.slider("Minimum Support (min_sup)", min_value=0.001, max_value=0.050, value=0.005, step=0.001, format="%.3f")
        min_conf_slider = st.slider("Minimum Confidence (min_conf)", min_value=0.02, max_value=0.50, value=0.08, step=0.01, format="%.2f")

    st.subheader("3. Recommendation Ranking")
    ranking_strategy = st.selectbox(
        "Candidate Ranking Strategy:",
        options=["Confidence-First", "Lift-First", "Hybrid (Conf × √Lift)"],
        index=0,
        help="Confidence-First ranks by maximum rule confidence. Lift-First ranks by co-enrollment multiplier. Hybrid balances confidence and lift."
    )

    st.divider()
    st.markdown("### ℹ️ Dataset Quick Info")
    st.write(f"• **Records:** {summary['total_records']:,}")
    st.write(f"• **Total Learners:** {summary['unique_learners']:,}")
    st.write(f"• **Multi-Course Baskets:** {summary['multi_course_learners']:,} ({summary['multi_course_pct']:.1f}%)")
    st.write(f"• **Unique Courses:** {summary['unique_courses']}")

# Cache Apriori computation
@st.cache_data(show_spinner=False)
def compute_mining(multi_only: bool, min_sup: float, min_conf: float):
    transactions, one_hot_df = build_transactions(df_enrollments, multi_only=multi_only)
    engine = AprioriEngine(transactions)
    df_itemsets, frequent_dict = engine.find_frequent_itemsets(min_sup)
    df_rules = engine.generate_rules(frequent_dict, min_conf)
    return transactions, one_hot_df, engine, df_itemsets, df_rules

transactions, one_hot_df, engine, df_itemsets, df_rules = compute_mining(
    is_multi_only, min_sup_slider, min_conf_slider
)

recommender = CourseRecommender(df_rules)
all_course_titles = sorted(list(df_enrollments['course_title'].unique()))

# Navigation Tabs
tabs = st.tabs([
    "📊 Dataset Overview",
    "⚙️ Apriori & Math Logic",
    "📦 Frequent Itemsets",
    "🔍 Rules Explorer",
    "🎯 Smart Recommendations",
    "🎛️ Threshold Playground",
    "📈 Visual Analytics",
    "📝 Evaluation & Audit"
])

# ==========================================
# TAB 1: DATASET OVERVIEW
# ==========================================
with tabs[0]:
    st.subheader("HarvardX Person-Course Dataset Overview")
    st.markdown("""
    The system utilizes the official de-identified **HarvardX Person-Course Dataset (Academic Year 2013)** 
    hosted by Harvard University on Harvard Dataverse (`doi:10.7910/DVN/26147`).
    Each row captures a student's interaction and registration in a HarvardX course.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Enrollments</div>
            <div class="metric-value">{summary['total_records']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unique Learners</div>
            <div class="metric-value">{summary['unique_learners']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Multi-Course Baskets</div>
            <div class="metric-value">{summary['multi_course_learners']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unique Courses</div>
            <div class="metric-value">{summary['unique_courses']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📚 Official Course Catalog & Metadata")
    catalog_data = []
    for cid, meta in COURSE_METADATA.items():
        cnt = (df_enrollments['course_id'] == cid).sum()
        pct = (cnt / summary['total_records']) * 100
        catalog_data.append({
            "Course Code": meta['code'],
            "Title": meta['title'],
            "Instructor": meta['instructor'],
            "Field / Department": meta['field'],
            "Total Enrollments": f"{cnt:,}",
            "Share of Total": f"{pct:.1f}%",
            "Description": meta['description']
        })
    st.dataframe(pd.DataFrame(catalog_data), use_container_width=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_freq = plot_item_frequency(df_enrollments)
        st.pyplot(fig_freq)
    with col_chart2:
        fig_basket = plot_basket_size_distribution(df_enrollments)
        st.pyplot(fig_basket)

# ==========================================
# TAB 2: APRIORI & MATH LOGIC
# ==========================================
with tabs[1]:
    st.subheader("Apriori Algorithm: Internal Workflow & Candidate Pruning")
    st.markdown("""
    The **Apriori Algorithm** relies on the **Apriori Principle (Downward Closure Property)**:
    > *All non-empty subsets of a frequent itemset must also be frequent.*
    
    If candidate $C_k$ contains any $(k-1)$-subset that is infrequent, $C_k$ is immediately **pruned** 
    without expensive scans across the entire transaction database.
    """)

    st.markdown("### 🔬 Execution Trace at Current Thresholds")
    trace_rows = []
    for t in engine.trace:
        trace_rows.append({
            "Itemset Size (k)": f"k = {t['level']}",
            "Candidates Generated (C_k)": t['candidates_generated'],
            "Pruned by Apriori Property": t['pruned_by_apriori'],
            "Pruned by Min Support": t['pruned_by_min_support'],
            "Surviving Frequent Itemsets (L_k)": t['surviving_frequent']
        })
    st.table(pd.DataFrame(trace_rows))

    fig_flow = plot_apriori_pruning_flow(engine.trace)
    st.pyplot(fig_flow)

    st.divider()
    st.markdown("### 🧮 Mathematical Understanding: Why Confidence(A → B) ≠ Confidence(B → A)")
    st.markdown("""
    While the **Support** of $A \\to B$ and $B \\to A$ is symmetric:
    $$\\text{support}(A \\to B) = \\text{support}(B \\to A) = P(A \\cap B) = \\frac{\\text{count}(A \\cup B)}{N}$$

    The **Confidence** is asymmetric and directional:
    $$\\text{confidence}(A \\to B) = P(B \\mid A) = \\frac{P(A \\cap B)}{P(A)} = \\frac{\\text{support}(A \\cup B)}{\\text{support}(A)}$$
    $$\\text{confidence}(B \\to A) = P(A \\mid B) = \\frac{P(A \\cap B)}{P(B)} = \\frac{\\text{support}(A \\cup B)}{\\text{support}(B)}$$

    Because $\\text{support}(A) \\neq \\text{support}(B)$ for courses of different popularities, the denominators differ, resulting in completely different conditional probabilities!
    """)

    # Live empirical proof from our dataset
    st.markdown("#### 🔍 Live Dataset Proof of Asymmetric Confidence:")
    sample_a = "CB22x: The Ancient Greek Hero"
    sample_b = "ER22x: Justice: What's the Right Thing to Do?"
    
    rule_ab = df_rules[(df_rules['antecedent_str'] == sample_a) & (df_rules['consequent_str'] == sample_b)]
    rule_ba = df_rules[(df_rules['antecedent_str'] == sample_b) & (df_rules['consequent_str'] == sample_a)]
    
    if not rule_ab.empty and not rule_ba.empty:
        r1 = rule_ab.iloc[0]
        r2 = rule_ba.iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"""
            **Direction: {sample_a} → {sample_b}**
            * Joint Support $P(A \\cap B)$: **{r1['support']:.4f}**
            * Antecedent Support $P(A)$: **{r1['antecedent_support']:.4f}**
            * **Confidence:** $\\frac{{{r1['support']:.4f}}}{{{r1['antecedent_support']:.4f}}} =$ **{r1['confidence']*100:.2f}%**
            """)
        with c2:
            st.info(f"""
            **Direction: {sample_b} → {sample_a}**
            * Joint Support $P(A \\cap B)$: **{r2['support']:.4f}**
            * Antecedent Support $P(B)$: **{r2['antecedent_support']:.4f}**
            * **Confidence:** $\\frac{{{r2['support']:.4f}}}{{{r2['antecedent_support']:.4f}}} =$ **{r2['confidence']*100:.2f}%**
            """)
        st.success(f"**Empirical Verification:** Confidence differs by **{abs(r1['confidence'] - r2['confidence'])*100:.2f} percentage points** because Justice is taken by significantly more learners than The Ancient Greek Hero.")
    else:
        st.write("Adjust support/confidence sliders in sidebar to inspect live empirical rule pairs.")

# ==========================================
# TAB 3: FREQUENT ITEMSETS
# ==========================================
with tabs[2]:
    st.subheader(f"Discovered Frequent Itemsets (Total: {len(df_itemsets)})")
    st.write(f"Itemsets meeting the minimum support threshold of **{min_sup_slider:.3f}** ({min_sup_slider*100:.1f}%):")

    if df_itemsets.empty:
        st.warning("No frequent itemsets found. Lower the minimum support threshold in the sidebar.")
    else:
        filter_len = st.multiselect(
            "Filter by Itemset Size (Length):",
            options=sorted(df_itemsets['length'].unique()),
            default=sorted(df_itemsets['length'].unique())
        )
        filtered_items = df_itemsets[df_itemsets['length'].isin(filter_len)].copy()
        
        search_query = st.text_input("Filter itemsets containing course keyword:", "")
        if search_query:
            filtered_items = filtered_items[filtered_items['itemset_str'].str.contains(search_query, case=False)]

        st.dataframe(
            filtered_items[['length', 'itemset_str', 'support', 'support_count']].rename(columns={
                'length': 'Size (k)',
                'itemset_str': 'Course Itemset',
                'support': 'Support',
                'support_count': 'Learner Count (N)'
            }),
            use_container_width=True
        )

        st.download_button(
            "📥 Download Frequent Itemsets (CSV)",
            data=filtered_items.to_csv(index=False).encode('utf-8'),
            file_name="frequent_itemsets.csv",
            mime="text/csv"
        )

# ==========================================
# TAB 4: RULES EXPLORER
# ==========================================
with tabs[3]:
    st.subheader(f"Association Rules Explorer (Total: {len(df_rules)})")
    st.write(f"Rules meeting min_sup = **{min_sup_slider:.3f}** and min_conf = **{min_conf_slider:.2f}**:")

    if df_rules.empty:
        st.warning("No association rules found. Lower minimum support or confidence in the sidebar.")
    else:
        search_rule = st.text_input("Search rules containing specific course:", "", key="rule_search_box")
        df_display_rules = df_rules.copy()
        if search_rule:
            df_display_rules = df_display_rules[
                df_display_rules['antecedent_str'].str.contains(search_rule, case=False) |
                df_display_rules['consequent_str'].str.contains(search_rule, case=False)
            ]

        # Display metric distribution summary
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Rules", len(df_display_rules))
        col_m2.metric("Max Confidence", f"{df_display_rules['confidence'].max()*100:.1f}%")
        col_m3.metric("Max Lift", f"{df_display_rules['lift'].max():.2f}x")
        col_m4.metric("Avg Support", f"{df_display_rules['support'].mean()*100:.1f}%")

        # Table display
        table_cols = [
            'rule_repr', 'support', 'confidence', 'lift', 'leverage', 'conviction', 'nl_explanation'
        ]
        renamed_cols = {
            'rule_repr': 'Association Rule (A → B)',
            'support': 'Support',
            'confidence': 'Confidence',
            'lift': 'Lift',
            'leverage': 'Leverage',
            'conviction': 'Conviction',
            'nl_explanation': 'Natural Language Explanation'
        }

        st.dataframe(
            df_display_rules[table_cols].rename(columns=renamed_cols),
            use_container_width=True
        )

        st.download_button(
            "📥 Download Discovered Rules (CSV)",
            data=df_display_rules.to_csv(index=False).encode('utf-8'),
            file_name="association_rules.csv",
            mime="text/csv"
        )

# ==========================================
# TAB 5: SMART RECOMMENDATIONS
# ==========================================
with tabs[4]:
    st.subheader("🎯 Smart Course Basket & Recommendation Engine")
    st.markdown("""
    Select courses to simulate a learner's enrollment basket.
    CourseCompass identifies matching association rules where the antecedent is enrolled,
    **strictly excludes already-selected courses**, aggregates evidence, and provides full rationales.
    """)

    selected_courses = st.multiselect(
        "🛒 Current Learner Basket (Select 1 or more enrolled courses):",
        options=all_course_titles,
        default=[all_course_titles[0]] if len(all_course_titles) > 0 else []
    )

    if not selected_courses:
        st.info("Please select at least one course in your basket to generate recommendations.")
    else:
        recs = recommender.get_recommendations(
            user_basket=selected_courses,
            ranking_strategy=ranking_strategy,
            top_k=5
        )

        st.markdown(f"### 💡 Recommendations for your Basket ({len(selected_courses)} selected):")

        if not recs:
            st.warning("""
            No courses could be recommended based on current rules.
            Try lowering minimum confidence/support in the sidebar or adding more courses.
            """)
        else:
            for i, rec in enumerate(recs, 1):
                st.markdown(f"""
                <div class="rule-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #60A5FA;">#{i} {rec['course']}</h4>
                        <div>
                            <span class="badge badge-primary">Rank Score: {rec['rank_score']:.3f}</span>
                            <span class="badge badge-success">Confidence: {rec['max_confidence']*100:.1f}%</span>
                            <span class="badge badge-warning">Lift: {rec['max_lift']:.2f}x</span>
                            <span class="badge badge-info">{rec['rule_count']} Evidence Rule(s)</span>
                        </div>
                    </div>
                    <p style="margin: 10px 0 6px 0; color: #E2E8F0; font-size: 0.95rem;">
                        <strong>Why this recommendation?</strong> {rec['rationale']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"🔍 View detailed rule evidence for {rec['course'].split(':')[0]}"):
                    evidence_df = pd.DataFrame(rec['all_evidence'])
                    st.dataframe(
                        evidence_df[['rule_repr', 'support', 'confidence', 'lift', 'nl_explanation']].rename(columns={
                            'rule_repr': 'Supporting Rule',
                            'support': 'Support',
                            'confidence': 'Confidence',
                            'lift': 'Lift',
                            'nl_explanation': 'Interpretation'
                        }),
                        use_container_width=True
                    )

        # What-If Analysis
        if len(selected_courses) >= 2:
            st.divider()
            st.markdown("### 🔄 What-If Dynamic Analysis: Course Removal Simulator")
            st.markdown("Observe how recommendations shift when removing a specific course from the current basket:")

            course_to_remove = st.selectbox(
                "Simulate removing which course?",
                options=selected_courses
            )

            what_if = recommender.what_if_analysis(selected_courses, course_to_remove, ranking_strategy)

            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown(f"**Current Basket ({len(selected_courses)} items):**")
                for c in selected_courses:
                    st.write(f"- {c}")
                st.markdown("**Current Recommendations:**")
                for r in what_if['current_recs']:
                    st.write(f"• **{r['course'].split(':')[0]}** (Conf: {r['max_confidence']*100:.1f}%)")

            with col_after:
                st.markdown(f"**Counterfactual Basket without [{course_to_remove.split(':')[0]}]:**")
                for c in what_if['counterfactual_basket']:
                    st.write(f"- {c}")
                st.markdown("**Updated Recommendations:**")
                for r in what_if['counterfactual_recs']:
                    st.write(f"• **{r['course'].split(':')[0]}** (Conf: {r['max_confidence']*100:.1f}%)")

            if what_if['lost_courses']:
                st.error(f"⚠️ Recommendations lost by removing {course_to_remove.split(':')[0]}: {', '.join(what_if['lost_courses'])}")
            if what_if['newly_added']:
                st.success(f"✨ Newly unlocked recommendations: {', '.join(what_if['newly_added'])}")

# ==========================================
# TAB 6: THRESHOLD PLAYGROUND
# ==========================================
with tabs[5]:
    st.subheader("🎛️ Threshold Playground & Setting Comparison")
    st.markdown("""
    Compare system outcomes, rule set sizes, and recommendation behavior 
    under two distinct support and confidence settings side by side.
    """)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Setting A (High Specificity / Conservative)")
        s1 = st.slider("Setting A min_sup:", 0.01, 0.30, 0.10, 0.01, key="s1")
        c1 = st.slider("Setting A min_conf:", 0.10, 0.80, 0.40, 0.05, key="c1")
    with col_t2:
        st.markdown("#### Setting B (High Coverage / Exploratory)")
        s2 = st.slider("Setting B min_sup:", 0.01, 0.30, 0.03, 0.01, key="s2")
        c2 = st.slider("Setting B min_conf:", 0.10, 0.80, 0.15, 0.05, key="c2")

    # Compute both settings
    @st.cache_data(show_spinner=False)
    def compare_thresholds(multi_only, sup_a, conf_a, sup_b, conf_b):
        txs, _ = build_transactions(df_enrollments, multi_only=multi_only)
        eng = AprioriEngine(txs)
        
        items_a, freq_dict_a = eng.find_frequent_itemsets(sup_a)
        rules_a = eng.generate_rules(freq_dict_a, conf_a)
        
        items_b, freq_dict_b = eng.find_frequent_itemsets(sup_b)
        rules_b = eng.generate_rules(freq_dict_b, conf_b)
        
        return len(items_a), len(rules_a), rules_a, len(items_b), len(rules_b), rules_b

    cnt_items_a, cnt_rules_a, rules_df_a, cnt_items_b, cnt_rules_b, rules_df_b = compare_thresholds(
        is_multi_only, s1, c1, s2, c2
    )

    st.markdown("### 📊 Direct Outcome Comparison Table")
    comp_df = pd.DataFrame({
        "Metric": [
            "Minimum Support Threshold",
            "Minimum Confidence Threshold",
            "Frequent Itemsets Discovered",
            "Association Rules Generated",
            "Average Rule Confidence",
            "Average Rule Lift"
        ],
        "Setting A (Conservative)": [
            f"{s1:.2f}",
            f"{c1:.2f}",
            cnt_items_a,
            cnt_rules_a,
            f"{rules_df_a['confidence'].mean()*100:.1f}%" if not rules_df_a.empty else "N/A",
            f"{rules_df_a['lift'].mean():.2f}x" if not rules_df_a.empty else "N/A"
        ],
        "Setting B (Exploratory)": [
            f"{s2:.2f}",
            f"{c2:.2f}",
            cnt_items_b,
            cnt_rules_b,
            f"{rules_df_b['confidence'].mean()*100:.1f}%" if not rules_df_b.empty else "N/A",
            f"{rules_df_b['lift'].mean():.2f}x" if not rules_df_b.empty else "N/A"
        ]
    })
    st.table(comp_df)

    st.markdown("### 🔍 Recommendation Impact on a Test Basket")
    test_basket = [all_course_titles[0]]
    st.write(f"Testing recommendations for test basket: **{test_basket[0]}**")
    
    rec_a = CourseRecommender(rules_df_a).get_recommendations(test_basket, top_k=3)
    rec_b = CourseRecommender(rules_df_b).get_recommendations(test_basket, top_k=3)
    
    c_out1, c_out2 = st.columns(2)
    with c_out1:
        st.markdown("**Setting A Output:**")
        if rec_a:
            for r in rec_a:
                st.write(f"• **{r['course'].split(':')[0]}** (Conf: {r['max_confidence']*100:.1f}%)")
        else:
            st.info("No recommendations met Setting A thresholds (too strict).")
    with c_out2:
        st.markdown("**Setting B Output:**")
        if rec_b:
            for r in rec_b:
                st.write(f"• **{r['course'].split(':')[0]}** (Conf: {r['max_confidence']*100:.1f}%)")
        else:
            st.info("No recommendations generated.")

# ==========================================
# TAB 7: VISUAL ANALYTICS
# ==========================================
with tabs[6]:
    st.subheader("📈 Visual Analytics & Association Graph")
    
    col_g1, col_g2 = st.columns([1.2, 1])
    with col_g1:
        st.markdown("#### Course Association Network Graph")
        st.write("Directed edges represent rules $A \\to B$. Thickness represents confidence; emerald edges indicate positive lift ($\\ge 1.0$).")
        fig_graph = plot_association_graph(df_rules, top_n=15, min_confidence=min_conf_slider)
        st.pyplot(fig_graph)
    with col_g2:
        st.markdown("#### Rule Metric Landscape")
        st.write("Scatter distribution of rule support versus confidence, with point coloring indicating lift.")
        fig_scatter = plot_rules_scatter(df_rules)
        st.pyplot(fig_scatter)

# ==========================================
# TAB 8: EVALUATION & AUDIT
# ==========================================
with tabs[7]:
    st.subheader("📝 Empirical Evaluation & Manual Rule Audit")

    st.markdown("### Part 1: Five Real-World Recommendation Test Scenarios")
    eval_scenarios = [
        {
            "Scenario": "1. Classical Humanities Scholar",
            "Input Basket": ["CB22x: The Ancient Greek Hero"],
            "Top Recommendation": "ER22x: Justice: What's the Right Thing to Do?",
            "Metrics": "Conf: 66.9% | Lift: 1.16x | Sup: 26.1%",
            "Rationale & Suitability": "Strong affinity between ancient political philosophy/mythology and Sandel's moral philosophy. Highly complementary."
        },
        {
            "Scenario": "2. Public Health Specialist",
            "Input Basket": ["PH207x: Health in Numbers: Quantitative Methods"],
            "Top Recommendation": "CS50x: Introduction to Computer Science I",
            "Metrics": "Conf: 54.7% | Lift: 1.10x | Sup: 16.0%",
            "Rationale & Suitability": "Epidemiology and biostatistics learners actively seek programming and computational data analysis skills."
        },
        {
            "Scenario": "3. Environmental Policy Learner",
            "Input Basket": ["PH278x: Human Health and Global Environmental Change"],
            "Top Recommendation": "ER22x: Justice: What's the Right Thing to Do?",
            "Metrics": "Conf: 53.5% | Lift: 0.92x | Sup: 24.3%",
            "Rationale & Suitability": "Environmental ethics, resource distribution, and climate justice naturally intersect with ethical decision-making."
        },
        {
            "Scenario": "4. Interdisciplinary Thinker",
            "Input Basket": ["CB22x: The Ancient Greek Hero", "CS50x: Introduction to Computer Science I"],
            "Top Recommendation": "ER22x: Justice: What's the Right Thing to Do?",
            "Metrics": "Conf: 45.2% | Lift: 0.78x | Sup: 6.2%",
            "Rationale & Suitability": "Broad liberal arts and computational foundation points directly to ethical frameworks."
        },
        {
            "Scenario": "5. Legal & Moral Philosophy Explorer",
            "Input Basket": ["ER22x: Justice: What's the Right Thing to Do?"],
            "Top Recommendation": "CB22x: The Ancient Greek Hero",
            "Metrics": "Conf: 45.1% | Lift: 1.16x | Sup: 26.1%",
            "Rationale & Suitability": "Reciprocal philosophical pathway exploring classical moral dilemmas, virtue ethics, and societal norms."
        }
    ]
    st.dataframe(pd.DataFrame(eval_scenarios), use_container_width=True)

    st.markdown("### Part 2: Manual Inspection & Classification of 10 Discovered Rules")
    manual_rules = [
        {
            "Rule ID": "R01",
            "Discovered Rule": "{CB22x: Greek Hero} → {ER22x: Justice}",
            "Conf": "66.9%",
            "Lift": "1.16x",
            "Classification": "Useful",
            "Domain Rationale": "Actionable pathway linking classical humanities with contemporary moral philosophy."
        },
        {
            "Rule ID": "R02",
            "Discovered Rule": "{ER22x: Justice} → {CB22x: Greek Hero}",
            "Conf": "45.1%",
            "Lift": "1.16x",
            "Classification": "Useful",
            "Domain Rationale": "Reciprocal humanities bridge for students wanting historical foundation for ethical theories."
        },
        {
            "Rule ID": "R03",
            "Discovered Rule": "{PH207x: Health in Numbers} → {CS50x: Intro to CS}",
            "Conf": "54.7%",
            "Lift": "1.10x",
            "Classification": "Useful",
            "Domain Rationale": "Meaningful interdisciplinary bridge between biostatistical quantitative methods and programming."
        },
        {
            "Rule ID": "R04",
            "Discovered Rule": "{PH278x: Environmental Health} → {ER22x: Justice}",
            "Conf": "53.5%",
            "Lift": "0.92x",
            "Classification": "Useful",
            "Domain Rationale": "Connects environmental science to distributive justice and climate ethics."
        },
        {
            "Rule ID": "R05",
            "Discovered Rule": "{ER22x: Justice} → {CS50x: Intro to CS}",
            "Conf": "37.6%",
            "Lift": "0.76x",
            "Classification": "Obvious / Trivial",
            "Domain Rationale": "CS50x is an overwhelming massive gateway course taken by ~50% of multi-course students; lift < 1 indicates no special synergy."
        },
        {
            "Rule ID": "R06",
            "Discovered Rule": "{CB22x: Greek Hero} → {CS50x: Intro to CS}",
            "Conf": "34.9%",
            "Lift": "0.70x",
            "Classification": "Obvious / Trivial",
            "Domain Rationale": "Reflects baseline global platform popularity of CS50x rather than thematic connection."
        },
        {
            "Rule ID": "R07",
            "Discovered Rule": "{PH207x: Health in Numbers} → {PH278x: Environmental Health}",
            "Conf": "40.8%",
            "Lift": "0.90x",
            "Classification": "Useful",
            "Domain Rationale": "Domain coherence within the Harvard School of Public Health curriculum."
        },
        {
            "Rule ID": "R08",
            "Discovered Rule": "{PH278x: Environmental Health} → {PH207x: Health in Numbers}",
            "Conf": "26.3%",
            "Lift": "0.90x",
            "Classification": "Weak / Uncertain",
            "Domain Rationale": "Lower confidence (26.3%) indicates learners in general environmental health don't reliably pursue heavy biostatistics."
        },
        {
            "Rule ID": "R09",
            "Discovered Rule": "{CB22x: Greek Hero, CS50x: Intro to CS} → {ER22x: Justice}",
            "Conf": "45.2%",
            "Lift": "0.78x",
            "Classification": "Unexpected",
            "Domain Rationale": "Students with eclectic mix of literature and programming converge back onto philosophy."
        },
        {
            "Rule ID": "R10",
            "Discovered Rule": "{CB22x: Greek Hero, ER22x: Justice} → {CS50x: Intro to CS}",
            "Conf": "23.6%",
            "Lift": "0.48x",
            "Classification": "Weak / Uncertain",
            "Domain Rationale": "Low confidence (23.6%) and low lift (0.48x) shows pure humanities students rarely cross into computer science."
        }
    ]
    st.table(pd.DataFrame(manual_rules))

    st.markdown("### Part 3: Critical Limitations of Association Rule Mining in Educational Recommendations")
    st.markdown("""
    1. **Inability to Model Pedagogical Progression & Prerequisite Sequences:**
       Association rule mining treats enrollment baskets as unordered static sets ($A \\cap B$). It cannot distinguish whether a learner took *Introduction to Computer Science* **before** or **after** an advanced course. In education, sequence and prerequisites are fundamental.
    
    2. **Popularity Bias & Dominant Gateway Courses:**
       High-enrollment courses (such as CS50x) naturally exhibit high support and appear in dozens of rules purely due to platform-wide scale, suppressing niche, high-quality domain relationships unless controlled using Lift or Leverage.
    
    3. **Cold-Start Problem for Newly Introduced Courses:**
       A newly launched course has zero historical co-enrollments, giving it 0 support. Standard Apriori will completely ignore it until sufficient learners enroll, requiring hybrid content-based fallbacks.
    """)
