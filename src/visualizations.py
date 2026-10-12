"""
visualizations.py
Interactive and statistical visual analytics for CourseCompass:
- Directed Association Network Graph (NetworkX + Matplotlib)
- Course enrollment frequency chart
- Learner basket size distribution
- Rule metrics scatter plot (Support vs Confidence vs Lift)
- Apriori candidate generation & pruning waterfall chart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Premium color palette
PALETTE = {
    "bg": "#0E1117",
    "card_bg": "#1E222D",
    "primary": "#4F46E5",     # Indigo
    "secondary": "#06B6D4",   # Cyan
    "accent": "#F59E0B",      # Amber
    "text": "#F3F4F6",        # Light grey
    "muted": "#9CA3AF",       # Medium grey
    "edge_positive": "#10B981", # Emerald
    "edge_high": "#EC4899"     # Pink
}

FIELD_COLORS = {
    "Computer Science & Programming": "#3B82F6",    # Blue
    "Philosophy, Law & Ethics": "#8B5CF6",          # Violet
    "Humanities & Classical Literature": "#EC4899", # Pink
    "Public Health, Biostatistics & Epidemiology": "#10B981", # Emerald
    "Public Health & Environmental Sciences": "#F59E0B"       # Amber
}

def plot_association_graph(
    rules_df: pd.DataFrame,
    top_n: int = 15,
    min_confidence: float = 0.2
) -> plt.Figure:
    """
    Constructs a directed Association Network Graph where:
    - Nodes represent courses
    - Directed edges represent association rules (A -> B)
    - Edge thickness represents confidence
    - Edge color represents lift
    """
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    if rules_df.empty:
        ax.text(0.5, 0.5, "No association rules available at current thresholds",
                color='#F3F4F6', fontsize=14, ha='center', va='center')
        ax.axis('off')
        return fig

    # Filter top rules by confidence
    filtered = rules_df[rules_df['confidence'] >= min_confidence].copy()
    if filtered.empty:
        filtered = rules_df.head(top_n)
    else:
        filtered = filtered.head(top_n)

    G = nx.DiGraph()

    for _, r in filtered.iterrows():
        ant_items = r['antecedent']
        con_items = r['consequent']
        # Add edge from each antecedent item to each consequent item
        for a in ant_items:
            for b in con_items:
                if a != b:
                    G.add_edge(a, b, weight=r['confidence'], lift=r['lift'], rule=r['rule_repr'])

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No edges to plot", color='#F3F4F6', ha='center', va='center')
        ax.axis('off')
        return fig

    # Layout using circular or spring layout for clarity
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=60)

    # Draw nodes
    node_sizes = [3000 + 300 * G.degree(n) for n in G.nodes()]
    node_colors = []
    for node in G.nodes():
        matched = False
        for field, color in FIELD_COLORS.items():
            if any(k in node for k in ["CS50", "Greek Hero", "Justice", "PH207", "PH278"]):
                if "CS50" in node:
                    node_colors.append("#3B82F6")
                elif "Justice" in node:
                    node_colors.append("#8B5CF6")
                elif "Greek Hero" in node:
                    node_colors.append("#EC4899")
                elif "PH207" in node or "Numbers" in node:
                    node_colors.append("#10B981")
                elif "PH278" in node or "Environmental" in node:
                    node_colors.append("#F59E0B")
                matched = True
                break
        if not matched:
            node_colors.append("#6366F1")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='#F3F4F6',
        linewidths=1.5,
        alpha=0.9
    )

    # Draw labels with clean wrapping
    labels = {}
    for n in G.nodes():
        # Shorten label for neat display
        parts = n.split(":")
        if len(parts) > 1:
            code = parts[0].strip()
            rest = parts[1].strip()
            labels[n] = f"{code}\n{rest[:18]}..." if len(rest) > 18 else f"{code}\n{rest}"
        else:
            labels[n] = n[:20]

    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8.5,
        font_color='#FFFFFF',
        font_weight='bold'
    )

    # Draw directed edges
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [1.0 + (w / max_w) * 4.0 for w in weights]

    lifts = [G[u][v]['lift'] for u, v in G.edges()]
    edge_colors = ['#10B981' if l >= 1.0 else '#EF4444' for l in lifts]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=widths,
        edge_color=edge_colors,
        arrowsize=18,
        arrowstyle='-|>',
        connectionstyle="arc3,rad=0.15",
        alpha=0.8
    )

    ax.set_title("Course Association Network Graph (Directed Rules: A → B)",
                 color='#F3F4F6', fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    # Add custom legend
    legend_elements = [
        mpatches.Patch(color='#10B981', label='Positive Association (Lift ≥ 1.0)'),
        mpatches.Patch(color='#EF4444', label='Negative / Sub-baseline (Lift < 1.0)'),
        mpatches.Patch(color='#F3F4F6', label='Edge width ∝ Rule Confidence')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=3, facecolor='#1E222D', edgecolor='#374151', labelcolor='#F3F4F6', fontsize=8.5)

    plt.tight_layout()
    return fig

def plot_item_frequency(df: pd.DataFrame) -> plt.Figure:
    """Plots enrollment counts across all available courses."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    counts = df['course_title'].value_counts()
    colors = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EC4899'][:len(counts)]

    # Shorten labels
    short_labels = [c.split(":")[0].strip() + (" - " + c.split(":")[1].strip()[:20] if ":" in c else "") for c in counts.index]

    bars = ax.barh(short_labels, counts.values, color=colors, edgecolor='#1E222D', height=0.6)
    ax.invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        ax.text(w + (max(counts.values) * 0.01), bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", ha='left', va='center', color='#F3F4F6', fontsize=9, fontweight='bold')

    ax.set_title("Total Enrollments per Course", color='#F3F4F6', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Enrollment Count", color='#9CA3AF', fontsize=10)
    ax.tick_params(colors='#9CA3AF', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#374151')
    ax.spines['bottom'].set_color('#374151')
    ax.grid(axis='x', color='#1F2937', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def plot_basket_size_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plots learner basket size distribution (1 to 5 courses)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    basket_sizes = df.groupby('userid_DI')['course_id'].nunique().value_counts().sort_index()

    bars = ax.bar(basket_sizes.index, basket_sizes.values, color='#6366F1', edgecolor='#4338CA', width=0.55)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (max(basket_sizes.values) * 0.015),
                f"{int(h):,}\n({h/len(df.groupby('userid_DI'))*100:.1f}%)",
                ha='center', va='bottom', color='#F3F4F6', fontsize=8.5, fontweight='bold')

    ax.set_title("Learner Basket Size Distribution (# of Enrolled Courses)", color='#F3F4F6', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Courses in Basket", color='#9CA3AF', fontsize=10)
    ax.set_ylabel("Number of Learners", color='#9CA3AF', fontsize=10)
    ax.set_xticks(basket_sizes.index)
    ax.tick_params(colors='#9CA3AF', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#374151')
    ax.spines['bottom'].set_color('#374151')
    ax.grid(axis='y', color='#1F2937', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def plot_rules_scatter(rules_df: pd.DataFrame) -> plt.Figure:
    """Plots Support vs Confidence colored by Lift."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    if rules_df.empty:
        ax.text(0.5, 0.5, "No rules available to display", color='#F3F4F6', ha='center', va='center')
        ax.axis('off')
        return fig

    sc = ax.scatter(
        rules_df['support'],
        rules_df['confidence'],
        c=rules_df['lift'],
        cmap='viridis',
        s=80,
        alpha=0.85,
        edgecolors='#F3F4F6',
        linewidth=0.8
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Lift Metric", color='#F3F4F6', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#9CA3AF')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#9CA3AF')

    ax.set_title("Association Rules: Support vs. Confidence (Colored by Lift)",
                 color='#F3F4F6', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Support", color='#9CA3AF', fontsize=10)
    ax.set_ylabel("Confidence", color='#9CA3AF', fontsize=10)
    ax.tick_params(colors='#9CA3AF', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#374151')
    ax.spines['bottom'].set_color('#374151')
    ax.grid(color='#1F2937', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def plot_apriori_pruning_flow(trace: List[Dict[str, Any]]) -> plt.Figure:
    """Visualizes candidates generated vs pruned vs surviving frequent itemsets across levels."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    if not trace:
        ax.text(0.5, 0.5, "No Apriori trace data", color='#F3F4F6', ha='center', va='center')
        ax.axis('off')
        return fig

    levels = [f"Level {t['level']} (k={t['level']})" for t in trace]
    gen = [t['candidates_generated'] for t in trace]
    surviving = [t['surviving_frequent'] for t in trace]
    pruned_sup = [t['pruned_by_min_support'] for t in trace]
    pruned_apr = [t['pruned_by_apriori'] for t in trace]

    x = np.arange(len(levels))
    width = 0.22

    ax.bar(x - width*1.5, gen, width, label='Candidates Generated ($C_k$)', color='#6366F1')
    ax.bar(x - width*0.5, pruned_apr, width, label='Pruned by Apriori Property', color='#EF4444')
    ax.bar(x + width*0.5, pruned_sup, width, label='Pruned by Min Support', color='#F59E0B')
    ax.bar(x + width*1.5, surviving, width, label='Surviving Frequent ($L_k$)', color='#10B981')

    ax.set_title("Apriori Candidate Generation & Pruning Waterfall", color='#F3F4F6', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, color='#F3F4F6', fontsize=9.5)
    ax.set_ylabel("Itemset Count", color='#9CA3AF', fontsize=10)
    ax.tick_params(colors='#9CA3AF')
    ax.legend(facecolor='#1E222D', edgecolor='#374151', labelcolor='#F3F4F6', fontsize=8.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#374151')
    ax.spines['bottom'].set_color('#374151')
    ax.grid(axis='y', color='#1F2937', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig
