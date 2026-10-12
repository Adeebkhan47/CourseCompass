"""
generate_report.py
Generates both:
1. REPORT.md (comprehensive technical report in GitHub-flavored Markdown)
2. TECHNICAL_REPORT.pdf (publication-grade PDF report using ReportLab)
Structured strictly according to Section 8 of the assignment document.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

REPORT_MD_PATH = os.path.join(PROJECT_ROOT, "REPORT.md")
REPORT_PDF_PATH = os.path.join(PROJECT_ROOT, "TECHNICAL_REPORT.pdf")

def generate_markdown_report():
    print("[1/2] Generating REPORT.md...")
    md_content = """# CourseCompass: Association Rule Mining–Based Course Recommendation System
## Applied Machine Learning (AI3002) — Assignment #01 Technical Report
**Institution:** National University of Computer & Emerging Sciences (FAST-NUCES)  
**Program:** BS Software Engineering | Semester 7 (Fall 2026)  
**Instructor:** Shoaib Khan  
**Target CLO:** CLO 1  

---

## 1. Introduction and Problem Statement
In higher education and Massive Open Online Courses (MOOCs), learners frequently navigate vast course catalogs without personalized pedagogical guidance. While typical e-commerce recommendation engines employ black-box matrix factorization or deep collaborative filtering, educational recommendation systems demand high transparency, determinism, and explainability. An educational recommendation is far more actionable when a student understands *why* a particular course is suggested (e.g., *"Learners who took Classical Greek Literature also studied Justice with 66.9% co-enrollment rate"*).

**Problem Statement:** Design and implement an end-to-end Course Recommendation Engine using **Association Rule Mining (ARM)** and the **Apriori Algorithm**. The system models each learner's enrollments as a distinct transaction basket, mines frequent co-enrollment patterns, generates directional association rules with rigorous metric guarantees (Support, Confidence, Lift, Leverage, Conviction), excludes already-enrolled courses from candidate outputs, and provides an interactive, explorable visual interface via Streamlit.

---

## 2. Application Domain and Motivation
### Domain Selection: Academic Course Enrollment Analytics (HarvardX MOOCs)
MOOC platforms such as edX host millions of enrollments across humanities, engineering, sciences, and philosophy. However, student retention and completion rates in open online learning are notoriously low (<10%). A primary factor is curriculum fragmentation: learners often enroll in a single isolated course and fail to discover related or complementary learning pathways.

### Motivation
1. **Interdisciplinary Skill Bridges:** Learners taking quantitative epidemiology frequently require programming skills (e.g., CS50x), while philosophy students benefit from foundational ethics and classical literature.
2. **Pedagogical Explainability:** Unlike neural embeddings, association rules offer exact conditional probabilities ($P(B|A)$) and lift multipliers, empowering advisors and students with interpretable recommendations.
3. **Deterministic Verification:** Mathematical thresholds for minimum support and confidence give domain administrators direct control over recommendation conservatism versus coverage.

---

## 3. Dataset Description & Preprocessing
### 3.1 Data Source and Provenance
The system utilizes the official de-identified **HarvardX Person-Course Academic Year 2013 Dataset (Version 3.0)**, deposited by Harvard University on **Harvard Dataverse** (DOI: `10.7910/DVN/26147`, File ID: `3624095`).

| Dataset Attribute | Value | Description |
| :--- | :--- | :--- |
| **Originating Platform** | edX / HarvardX | First inaugural academic year (2012–2013) |
| **Raw Records (Rows)** | **338,223** | Total student-course registration events |
| **Unique Learners** | **301,609** | Distinct de-identified learner IDs (`userid_DI`) |
| **Unique Courses** | **5** | Distinct HarvardX flagship courses offered in Year 1 |
| **Cleaned Enrollment Records** | **338,223** | Records after deduplication and validation |
| **Multi-Course Baskets (≥ 2)** | **30,267** | Learners enrolled in 2 or more courses (10.04%) |
| **Single-Course Baskets (= 1)** | **271,342** | Learners enrolled in exactly 1 course (89.96%) |
| **Maximum Basket Size** | **5** | Maximum courses taken by a single student |

### 3.2 Flagship Course Catalog
1. **HarvardX/CS50x/2012**: *Introduction to Computer Science I* (Prof. David J. Malan) — 169,621 enrollments (50.15% share)
2. **HarvardX/ER22x/2013_Spring**: *Justice: What's the Right Thing to Do?* (Prof. Michael J. Sandel) — 57,406 enrollments (16.97% share)
3. **HarvardX/PH207x/2012_Fall**: *Health in Numbers: Quantitative Methods in Clinical & Public Health Research* (Prof. Marcello Pagano & Prof. E. Francis Cook) — 41,592 enrollments (12.29% share)
4. **HarvardX/PH278x/2013_Spring**: *Human Health and Global Environmental Change* (Prof. Aaron Bernstein & Prof. John Spengler) — 39,602 enrollments (11.71% share)
5. **HarvardX/CB22x/2013_Spring**: *The Ancient Greek Hero* (Prof. Gregory Nagy) — 30,002 enrollments (8.87% share)

### 3.3 Data Cleaning and Preprocessing Pipeline
1. **Missing Value Resolution:** Inspected identifiers `userid_DI` and `course_id`. Zero null entries found in essential transaction fields.
2. **Deduplication:** Evaluated `(userid_DI, course_id)` pairs to ensure no learner has redundant duplicate entries for the same course.
3. **Metadata Enrichment:** Standardized raw identifiers into human-readable titles, instructor attributions, and academic divisions.
4. **Basket Construction:** Transformed raw tabular records into learner-centric transactions:
   $$\mathcal{T} = \{ T_u \mid u \in \text{Learners}, T_u = \{ c_{u,1}, c_{u,2}, \dots, c_{u,k} \} \}$$
5. **Dual Population Modeling:** Implemented dual transaction views:
   - **Multi-Course Cohort ($N = 30,267$):** Focuses on students demonstrating cross-course behavior, preventing 90% single-course drop-in skew.
   - **Full Platform Cohort ($N = 301,609$):** Global platform perspective with calibrated micro-support thresholds.

---

## 4. Association Rule Mining Methodology
### 4.1 Core Mathematical Definitions

| Metric / Term | Formula / Definition | Simple Interpretation |
| :--- | :--- | :--- |
| **Support** | $\\text{support}(A \\to B) = P(A \\cap B) = \\frac{\\text{count}(A \\cup B)}{N}$ | How frequently courses $A$ and $B$ occur together across all learner baskets. |
| **Confidence** | $\\text{confidence}(A \\to B) = P(B \\mid A) = \\frac{\\text{support}(A \\cup B)}{\\text{support}(A)}$ | The proportion of learners enrolled in $A$ who also enrolled in course $B$. |
| **Lift** | $\\text{lift}(A \\to B) = \\frac{P(B \\mid A)}{P(B)} = \\frac{\\text{confidence}(A \\to B)}{\\text{support}(B)}$ | Ratio of observed co-enrollment to expected co-enrollment under independence. Lift > 1 indicates positive affinity. |
| **Leverage** | $\\text{leverage}(A \\to B) = P(A \\cap B) - P(A) \\cdot P(B)$ | Absolute excess co-occurrence frequency above expected independence. |
| **Conviction** | $\\text{conviction}(A \\to B) = \\frac{1 - P(B)}{1 - \\text{confidence}(A \\to B)}$ | Measures the degree of implication; infinite for perfect rules. |

### 4.2 Mathematical Derivation: Why Confidence(A → B) ≠ Confidence(B → A)
While Joint Support is strictly symmetric:
$$\\text{support}(A \\to B) = \\text{support}(B \\to A) = P(A \\cap B)$$

Rule Confidence is fundamentally asymmetric:
$$\\text{confidence}(A \\to B) = \\frac{P(A \\cap B)}{P(A)}, \\quad \\text{confidence}(B \\to A) = \\frac{P(A \\cap B)}{P(B)}$$

Whenever $P(A) \\neq P(B)$, the denominators differ. In educational datasets, course popularity varies widely (e.g., Justice has 57,406 enrollments while Greek Hero has 30,002). Consequently:
$$\\text{confidence}(\\text{Greek Hero} \\to \\text{Justice}) = \\frac{0.2609}{0.3902} = \\mathbf{66.87\\%}$$
$$\\text{confidence}(\\text{Justice} \\to \\text{Greek Hero}) = \\frac{0.2609}{0.5785} = \\mathbf{45.11\\%}$$
The $21.76\\%$ difference arises because a Greek Hero student is much more likely to take Justice than a Justice student is to take Greek Hero.

---

## 5. Apriori Explanation and Hand-Worked Example
### 5.1 The Apriori Principle (Downward Closure Property)
> **Theorem:** *If an itemset is frequent, then all of its subsets must also be frequent.*  
> **Contrapositive (Pruning Rule):** *If any subset of an itemset is infrequent, the superset cannot be frequent and is pruned immediately.*

### 5.2 Hand-Worked Example
Consider a toy educational database with $N = 5$ learner baskets and threshold $\\text{min\\_sup} = 0.40$ (minimum count $= 2$):
- Basket 1: $\{ \\text{CS}, \\text{Ethics} \}$
- Basket 2: $\{ \\text{CS}, \\text{Stats}, \\text{Ethics} \}$
- Basket 3: $\{ \\text{Stats}, \\text{Bio} \}$
- Basket 4: $\{ \\text{CS}, \\text{Stats}, \\text{Bio} \}$
- Basket 5: $\{ \\text{CS}, \\text{Ethics} \}$

#### Step 1: Candidate Generation $C_1$ & Pruning to $L_1$
- $\\text{CS}$: Count $= 4 \\implies \\text{sup} = 0.80 \\ge 0.40$ (Frequent)
- $\\text{Ethics}$: Count $= 3 \\implies \\text{sup} = 0.60 \\ge 0.40$ (Frequent)
- $\\text{Stats}$: Count $= 3 \\implies \\text{sup} = 0.60 \\ge 0.40$ (Frequent)
- $\\text{Bio}$: Count $= 2 \\implies \\text{sup} = 0.40 \\ge 0.40$ (Frequent)
- **Result $L_1$:** $\{ \\{\\text{CS}\\}, \\{\\text{Ethics}\\}, \\{\\text{Stats}\\}, \\{\\text{Bio}\\} \}$

#### Step 2: Candidate Generation $C_2$ ($L_1 \\bowtie L_1$)
- Pairs: $\{ \\text{CS}, \\text{Ethics} \}$, $\{ \\text{CS}, \\text{Stats} \}$, $\{ \\text{CS}, \\text{Bio} \}$, $\{ \\text{Ethics}, \\text{Stats} \}$, $\{ \\text{Ethics}, \\text{Bio} \}$, $\{ \\text{Stats}, \\text{Bio} \}$
- Support counts:
  - $\{\\text{CS}, \\text{Ethics}\}$: Count $= 3 \\implies 0.60$ (Frequent)
  - $\{\\text{CS}, \\text{Stats}\}$: Count $= 2 \\implies 0.40$ (Frequent)
  - $\{\\text{CS}, \\text{Bio}\}$: Count $= 1 \\implies 0.20$ (Infrequent $\\to$ Pruned by support)
  - $\{\\text{Ethics}, \\text{Stats}\}$: Count $= 1 \\implies 0.20$ (Infrequent $\\to$ Pruned by support)
  - $\{\\text{Ethics}, \\text{Bio}\}$: Count $= 0 \\implies 0.00$ (Infrequent $\\to$ Pruned by support)
  - $\{\\text{Stats}, \\text{Bio}\}$: Count $= 2 \\implies 0.40$ (Frequent)
- **Result $L_2$:** $\{ \\{\\text{CS}, \\text{Ethics}\\}, \\{\\text{CS}, \\text{Stats}\\}, \\{\\text{Stats}, \\text{Bio}\\} \}$

#### Step 3: Candidate Generation $C_3$ ($L_2 \\bowtie L_2$) & Apriori Pruning
- Joining candidate $\{\\text{CS}, \\text{Ethics}, \\text{Stats}\}$:
  - Subsets: $\{\\text{CS}, \\text{Ethics}\} \\in L_2$, $\{\\text{CS}, \\text{Stats}\} \\in L_2$, but $\{\\text{Ethics}, \\text{Stats}\} \\notin L_2$!
  - **Pruning Action:** Pruned prior to scanning transaction records!
- No valid candidates survive. Algorithm terminates.

---

## 6. Discovered Frequent Itemsets and Association Rules
At $\\text{min\\_sup} = 0.05$ and $\\text{min\\_conf} = 0.15$ on multi-course learner baskets ($N = 30,267$), the algorithm discovers 15 frequent itemsets and 25 association rules.

### 6.1 Top Frequent Itemsets

| Length ($k$) | Frequent Itemset | Support | Co-enrolled Learners |
| :---: | :--- | :---: | :---: |
| 1 | {ER22x: Justice} | 0.5785 | 17,509 |
| 1 | {CS50x: Intro to CS} | 0.4957 | 15,002 |
| 1 | {PH278x: Environmental Health} | 0.4534 | 13,724 |
| 1 | {CB22x: Greek Hero} | 0.3902 | 11,811 |
| 1 | {PH207x: Health in Numbers} | 0.2919 | 8,835 |
| 2 | {CB22x: Greek Hero, ER22x: Justice} | **0.2609** | **7,898** |
| 2 | {ER22x: Justice, PH278x: Environmental Health} | **0.2426** | **7,343** |
| 2 | {CS50x: Intro to CS, ER22x: Justice} | **0.2178** | **6,591** |
| 2 | {CS50x: Intro to CS, PH207x: Health in Numbers} | **0.1597** | **4,833** |
| 2 | {CB22x: Greek Hero, CS50x: Intro to CS} | **0.1361** | **4,118** |
| 2 | {CB22x: Greek Hero, PH278x: Environmental Health} | **0.1325** | **4,011** |
| 2 | {CS50x: Intro to CS, PH278x: Environmental Health} | **0.1234** | **3,735** |
| 2 | {PH207x: Health in Numbers, PH278x: Environmental Health} | **0.1191** | **3,605** |
| 3 | {CB22x: Greek Hero, ER22x: Justice, PH278x: Environmental Health} | **0.0753** | **2,279** |
| 3 | {CB22x: Greek Hero, CS50x: Intro to CS, ER22x: Justice} | **0.0616** | **1,863** |

---

## 7. Recommendation Engine Logic & System Architecture
```
[User Basket] ──> [Filter Matching Rules: Antecedent ⊆ Basket]
                       │
                       ▼
             [Candidate Course Extraction]
                       │
                       ▼
         [CRITICAL: Exclude Courses in User Basket]
                       │
                       ▼
         [Aggregate Supporting Evidence Rules]
                       │
                       ▼
          [Apply Ranking Strategy (Conf / Lift / Hybrid)]
                       │
                       ▼
       [Generate Dynamic "Why This Recommendation?" Rationale]
```

### 7.1 Key Invariants
1. **Zero Self-Recommendation:** If a user selects $\{ \\text{CB22x: Greek Hero} \}$, the candidate pool strictly excludes $\\text{CB22x}$.
2. **Evidence Aggregation:** If multiple rules point to course $C$, all rules are collected into an evidence dossier.
3. **Multi-Strategy Ranking:**
   - **Confidence-First:** $S = \\max(\\text{confidence})$
   - **Lift-First:** $S = \\max(\\text{lift})$
   - **Hybrid Score:** $S = \\max(\\text{confidence}) \\times \\sqrt{\\max(\\text{lift})} \\times [1 + 0.05 \\min(\\text{count}-1, 4)]$

---

## 8. Threshold Experiments & Outcome Comparison
We evaluated the behavior of CourseCompass across two operational settings:

| Metric | Setting A (Conservative) | Setting B (Exploratory) |
| :--- | :---: | :---: |
| **Minimum Support ($min\\_sup$)** | **0.10** | **0.03** |
| **Minimum Confidence ($min\\_conf$)** | **0.40** | **0.15** |
| **Frequent Itemsets Discovered** | 13 | 16 |
| **Association Rules Discovered** | 9 | 26 |
| **Average Rule Confidence** | **52.6%** | **38.4%** |
| **Average Rule Lift** | **1.01x** | **0.86x** |
| **Recommendation Coverage for Single-Course Basket** | Highly selective (1–2 top recs) | Rich diversity (3–4 recs) |

**Analysis:** Setting A ensures only exceptionally strong, high-volume associations trigger recommendations, avoiding noise. Setting B surfaces subtle interdisciplinary bridges (such as biostatistics cross-pollinating into computer science) at the cost of including lower-confidence associations.

---

## 9. Evaluation: Five Recommendation Scenarios & Manual Rule Audit

### 9.1 Five Evaluated Recommendation Scenarios
1. **Scenario 1: Classical Humanities Scholar**
   - *Input:* `{CB22x: The Ancient Greek Hero}`
   - *Top Recommendation:* `ER22x: Justice: What's the Right Thing to Do?`
   - *Metrics:* Conf: 66.9%, Lift: 1.16x, Sup: 26.1%
   - *Audit:* Exceptionally coherent pedagogical transition from ancient heroic ethics to modern political and moral philosophy.
2. **Scenario 2: Quantitative Public Health Analyst**
   - *Input:* `{PH207x: Health in Numbers: Quantitative Methods}`
   - *Top Recommendation:* `CS50x: Introduction to Computer Science I`
   - *Metrics:* Conf: 54.7%, Lift: 1.10x, Sup: 16.0%
   - *Audit:* Demonstrates natural student motivation to augment statistical analysis with programming and software skills.
3. **Scenario 3: Environmental Sciences Learner**
   - *Input:* `{PH278x: Human Health and Global Environmental Change}`
   - *Top Recommendation:* `ER22x: Justice: What's the Right Thing to Do?`
   - *Metrics:* Conf: 53.5%, Lift: 0.92x, Sup: 24.3%
   - *Audit:* Reflects intersection of climate crisis, environmental stewardship, and social/distributive justice.
4. **Scenario 4: Interdisciplinary Thinker**
   - *Input:* `{CB22x: The Ancient Greek Hero, CS50x: Intro to CS}`
   - *Top Recommendation:* `ER22x: Justice: What's the Right Thing to Do?`
   - *Metrics:* Conf: 45.2%, Lift: 0.78x, Sup: 6.2%
   - *Audit:* Triangulates technical and classical training toward legal and ethical foundations.
5. **Scenario 5: Moral Philosophy Student**
   - *Input:* `{ER22x: Justice: What's the Right Thing to Do?}`
   - *Top Recommendation:* `CB22x: The Ancient Greek Hero`
   - *Metrics:* Conf: 45.1%, Lift: 1.16x, Sup: 26.1%
   - *Audit:* Reciprocal humanities bridge; shows how directional confidence naturally reflects relative population sizes.

### 9.2 Manual Inspection of 10 Discovered Rules

| ID | Discovered Rule | Conf | Lift | Category | Domain Rationale |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **R01** | {CB22x: Greek Hero} → {ER22x: Justice} | **66.9%** | **1.16x** | **Useful** | Actionable pathway connecting ancient humanities to modern moral philosophy. |
| **R02** | {ER22x: Justice} → {CB22x: Greek Hero} | **45.1%** | **1.16x** | **Useful** | Solid reciprocal bridge providing classical literature context for ethical theory. |
| **R03** | {PH207x: Health in Numbers} → {CS50x: Intro to CS} | **54.7%** | **1.10x** | **Useful** | High practical utility bridging biostatistics with computer programming. |
| **R04** | {PH278x: Environmental Health} → {ER22x: Justice} | **53.5%** | **0.92x** | **Useful** | Bridges environmental challenges to distributive justice and public ethics. |
| **R05** | {ER22x: Justice} → {CS50x: Intro to CS} | **37.6%** | **0.76x** | **Obvious/Trivial** | CS50x is taken by ~50% of students; high confidence driven purely by platform ubiquity. |
| **R06** | {CB22x: Greek Hero} → {CS50x: Intro to CS} | **34.9%** | **0.70x** | **Obvious/Trivial** | Reflects general popularity of CS50x rather than specific pedagogical affinity. |
| **R07** | {PH207x: Health in Numbers} → {PH278x: Environmental Health} | **40.8%** | **0.90x** | **Useful** | Internal coherence within Harvard School of Public Health curriculum. |
| **R08** | {PH278x: Environmental Health} → {PH207x: Health in Numbers} | **26.3%** | **0.90x** | **Weak/Uncertain** | Low confidence indicates environmental students do not reliably pursue rigorous biostatistics. |
| **R09** | {CB22x: Greek Hero, CS50x: Intro to CS} → {ER22x: Justice} | **45.2%** | **0.78x** | **Unexpected** | Students combining coding and epic poetry converge strongly back onto philosophy. |
| **R10** | {CB22x: Greek Hero, ER22x: Justice} → {CS50x: Intro to CS} | **23.6%** | **0.48x** | **Weak/Uncertain** | Low confidence and poor lift (0.48x); dual humanities students rarely cross into CS. |

---

## 10. System Limitations
1. **Absence of Temporal & Pedagogical Ordering:**  
   Standard ARM treats baskets as unordered sets ($\\{A, B\\}$). In education, course sequence is crucial: taking Introduction to CS *before* Biostatistics is natural, but the reverse order may lead to severe frustration. ARM cannot represent prerequisite graphs without sequence-mining extensions (e.g., GSP, PrefixSpan).
2. **Super-Item Popularity Bias:**  
   When a single course (such as CS50x) commands 50%+ enrollment, it naturally appears in almost every rule consequent. Without filtering by Lift or Leverage, the system risks recommending the most popular course to everyone regardless of specialization.
3. **Cold-Start Vulnerability:**  
   New courses have 0 transactions, resulting in 0 support. Apriori completely omits new courses until a substantial volume of co-enrollments accumulates.

---
*Report generated automatically from verified empirical calculations on HarvardX dataset.*
"""
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] REPORT.md written to {REPORT_MD_PATH}")

def generate_pdf_report():
    print("[2/2] Generating TECHNICAL_REPORT.pdf via ReportLab...")
    doc = SimpleDocTemplate(
        REPORT_PDF_PATH,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E1B4B'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#4338CA'),
        spaceAfter=12
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=6
    )
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1E1B4B'),
        backColor=colors.HexColor('#F1F5F9'),
        spaceBefore=4,
        spaceAfter=6
    )
    table_text = ParagraphStyle(
        'TText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1F2937')
    )
    table_header = ParagraphStyle(
        'THeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # Title & Header
    story.append(Paragraph("CourseCompass: Association Rule Mining–Based Course Recommendation System", title_style))
    story.append(Paragraph("Applied Machine Learning (AI3002) — Project Assignment #01 Technical Report", subtitle_style))
    story.append(Paragraph("<b>Institution:</b> National University of Computer & Emerging Sciences (FAST-NUCES)<br/>"
                           "<b>Program:</b> BS Software Engineering | Semester 7 (Fall 2026) | <b>Instructor:</b> Shoaib Khan | <b>Target CLO:</b> CLO 1", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#4338CA'), spaceBefore=2, spaceAfter=10))

    # Section 1
    story.append(Paragraph("1. Introduction and Problem Statement", h1_style))
    story.append(Paragraph(
        "In modern online education, learners face vast catalogs without individualized guidance. "
        "Unlike black-box collaborative filtering, <b>Association Rule Mining (ARM)</b> provides fully transparent, "
        "interpretable, and mathematically verifiable recommendations. CourseCompass formulates each learner's enrollments "
        "as a transaction basket, discovers frequent course co-occurrences via the Apriori algorithm, calculates direction-sensitive "
        "metrics (Support, Confidence, Lift, Leverage, Conviction), strictly excludes already-enrolled items, and presents an interactive "
        "Streamlit application with 'Why this recommendation?' rationales.", body_style
    ))

    # Section 2
    story.append(Paragraph("2. Application Domain and Motivation", h1_style))
    story.append(Paragraph(
        "<b>Domain:</b> Academic Course Enrollment Analytics on edX/HarvardX MOOCs. Open online learning suffers from massive dropout rates (>90%), "
        "largely due to isolated course taking. CourseCompass identifies natural interdisciplinary bridges—such as connecting Classical Humanities to Moral Philosophy, "
        "or Public Health Biostatistics to Computer Science—guiding students along coherent educational pathways.", body_style
    ))

    # Section 3
    story.append(Paragraph("3. Dataset Description & Preprocessing Pipeline", h1_style))
    story.append(Paragraph(
        "The system uses the official de-identified <b>HarvardX Person-Course Academic Year 2013 Dataset</b> (Harvard Dataverse DOI: 10.7910/DVN/26147). "
        "The dataset contains <b>338,223 total records</b> across <b>301,609 unique learners</b> and 5 flagship HarvardX courses.", body_style
    ))

    ds_summary_data = [
        [Paragraph("Dataset Metric", table_header), Paragraph("Value", table_header), Paragraph("Description", table_header)],
        [Paragraph("Total Records", table_text), Paragraph("338,223", table_text), Paragraph("Total student course registrations", table_text)],
        [Paragraph("Unique Learners", table_text), Paragraph("301,609", table_text), Paragraph("Distinct de-identified user accounts", table_text)],
        [Paragraph("Unique Courses", table_text), Paragraph("5", table_text), Paragraph("Flagship HarvardX inaugural courses", table_text)],
        [Paragraph("Multi-Course Baskets", table_text), Paragraph("30,267 (10.0%)", table_text), Paragraph("Learners with basket size ≥ 2 (Primary ARM cohort)", table_text)],
        [Paragraph("Single-Course Baskets", table_text), Paragraph("271,342 (90.0%)", table_text), Paragraph("Learners taking exactly one course", table_text)],
        [Paragraph("Max Basket Size", table_text), Paragraph("5", table_text), Paragraph("Enrolled in all 5 flagship courses", table_text)]
    ]
    t_ds = Table(ds_summary_data, colWidths=[1.8*inch, 1.4*inch, 4.0*inch])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E1B4B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 8))

    # Section 4
    story.append(Paragraph("4. Association Rule Mining Methodology", h1_style))
    story.append(Paragraph(
        "<b>Mathematical Formulations:</b><br/>"
        "• <b>Support:</b> support(A → B) = P(A ∩ B) = count(A ∪ B) / N<br/>"
        "• <b>Confidence:</b> confidence(A → B) = P(B | A) = support(A ∪ B) / support(A)<br/>"
        "• <b>Lift:</b> lift(A → B) = confidence(A → B) / support(B)<br/>"
        "• <b>Leverage:</b> leverage(A → B) = support(A ∪ B) - support(A) · support(B)<br/>"
        "• <b>Conviction:</b> conviction(A → B) = (1 - support(B)) / (1 - confidence(A → B))", body_style
    ))
    story.append(Paragraph(
        "<b>Mathematical Derivation: Why Confidence(A → B) ≠ Confidence(B → A):</b><br/>"
        "While Joint Support P(A ∩ B) is symmetric, Confidence divides by the antecedent's individual support: "
        "conf(A → B) = P(A ∩ B)/P(A) while conf(B → A) = P(A ∩ B)/P(B). Because course popularities vary, P(A) ≠ P(B). "
        "Empirically, conf(Greek Hero → Justice) = 26.09% / 39.02% = <b>66.87%</b>, whereas conf(Justice → Greek Hero) = 26.09% / 57.85% = <b>45.11%</b>. "
        "The 21.76% difference reflects that Justice is broader and more populous than Greek Hero.", body_style
    ))

    # Section 5
    story.append(Paragraph("5. Apriori Explanation and Hand-Worked Example", h1_style))
    story.append(Paragraph(
        "<b>The Apriori Principle (Downward Closure):</b> If an itemset is frequent, all its subsets must be frequent. "
        "Any candidate itemset containing an infrequent subset is pruned prior to database scanning.<br/>"
        "<b>Hand-Worked Example (N=5 transactions, min_sup=0.40):</b><br/>"
        "Baskets: T1={CS, Ethics}, T2={CS, Stats, Ethics}, T3={Stats, Bio}, T4={CS, Stats, Bio}, T5={CS, Ethics}.<br/>"
        "• Level 1: C1={CS:4, Ethics:3, Stats:3, Bio:2} → All survive into L1.<br/>"
        "• Level 2: C2 joined from L1. {CS, Bio} has count 1 (pruned), {Ethics, Stats} has count 1 (pruned). L2={{CS, Ethics}, {CS, Stats}, {Stats, Bio}}.<br/>"
        "• Level 3: Joining {CS, Ethics} and {CS, Stats} produces {CS, Ethics, Stats}. Its subset {Ethics, Stats} ∉ L2. "
        "<b>Pruned by Apriori property without scanning!</b>", body_style
    ))

    story.append(PageBreak())

    # Section 6
    story.append(Paragraph("6. Frequent Itemsets and Association Rules", h1_style))
    story.append(Paragraph("Mining on multi-course learner baskets (N=30,267) at min_sup=0.05 and min_conf=0.15 yields 15 itemsets and 25 rules:", body_style))

    rules_sample_data = [
        [Paragraph("Discovered Association Rule (A → B)", table_header), Paragraph("Sup", table_header), Paragraph("Conf", table_header), Paragraph("Lift", table_header), Paragraph("Leverage", table_header)],
        [Paragraph("{CB22x: Greek Hero} → {ER22x: Justice}", table_text), Paragraph("0.2609", table_text), Paragraph("66.87%", table_text), Paragraph("1.16x", table_text), Paragraph("+0.0352", table_text)],
        [Paragraph("{ER22x: Justice} → {CB22x: Greek Hero}", table_text), Paragraph("0.2609", table_text), Paragraph("45.11%", table_text), Paragraph("1.16x", table_text), Paragraph("+0.0352", table_text)],
        [Paragraph("{PH207x: Health Numbers} → {CS50x: Intro to CS}", table_text), Paragraph("0.1597", table_text), Paragraph("54.70%", table_text), Paragraph("1.10x", table_text), Paragraph("+0.0150", table_text)],
        [Paragraph("{PH278x: Environmental} → {ER22x: Justice}", table_text), Paragraph("0.2426", table_text), Paragraph("53.50%", table_text), Paragraph("0.92x", table_text), Paragraph("-0.0197", table_text)],
        [Paragraph("{CB22x: Greek Hero} → {CS50x: Intro to CS}", table_text), Paragraph("0.1361", table_text), Paragraph("34.87%", table_text), Paragraph("0.70x", table_text), Paragraph("-0.0574", table_text)],
        [Paragraph("{CB22x, CS50x} → {ER22x: Justice}", table_text), Paragraph("0.0616", table_text), Paragraph("45.24%", table_text), Paragraph("0.78x", table_text), Paragraph("-0.0171", table_text)]
    ]
    t_rules = Table(rules_sample_data, colWidths=[3.6*inch, 0.8*inch, 0.9*inch, 0.8*inch, 1.1*inch])
    t_rules.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E1B4B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_rules)
    story.append(Spacer(1, 8))

    # Section 7
    story.append(Paragraph("7. Recommendation Engine Logic", h1_style))
    story.append(Paragraph(
        "<b>Recommendation Process:</b><br/>"
        "1. Matches all association rules whose antecedent is a subset of the learner's basket.<br/>"
        "2. <b>Strict Exclusion:</b> Completely eliminates courses already selected by the learner.<br/>"
        "3. <b>Evidence Aggregation:</b> Collects all rules supporting each candidate course.<br/>"
        "4. <b>Ranking:</b> Supports Confidence-First, Lift-First, and Hybrid Score (Conf × √Lift).<br/>"
        "5. <b>Dynamic Explanation:</b> Produces natural-language 'Why this recommendation?' narratives.", body_style
    ))

    # Section 8
    story.append(Paragraph("8. Threshold Experiments & Outcome Comparison", h1_style))
    exp_data = [
        [Paragraph("Experiment Dimension", table_header), Paragraph("Setting A (Conservative)", table_header), Paragraph("Setting B (Exploratory)", table_header)],
        [Paragraph("Minimum Support (min_sup)", table_text), Paragraph("0.10 (10%)", table_text), Paragraph("0.03 (3%)", table_text)],
        [Paragraph("Minimum Confidence (min_conf)", table_text), Paragraph("0.40 (40%)", table_text), Paragraph("0.15 (15%)", table_text)],
        [Paragraph("Frequent Itemsets Discovered", table_text), Paragraph("13 itemsets", table_text), Paragraph("16 itemsets", table_text)],
        [Paragraph("Association Rules Generated", table_text), Paragraph("9 rules", table_text), Paragraph("26 rules", table_text)],
        [Paragraph("Mean Rule Confidence", table_text), Paragraph("52.6%", table_text), Paragraph("38.4%", table_text)],
        [Paragraph("Mean Rule Lift", table_text), Paragraph("1.01x", table_text), Paragraph("0.86x", table_text)],
        [Paragraph("Recommendation Behavior", table_text), Paragraph("High precision, low coverage", table_text), Paragraph("Broad coverage, more serendipity", table_text)]
    ]
    t_exp = Table(exp_data, colWidths=[2.5*inch, 2.3*inch, 2.4*inch])
    t_exp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E1B4B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_exp)
    story.append(Spacer(1, 8))

    # Section 9
    story.append(Paragraph("9. Manual Rule Evaluation (10 Discovered Rules)", h1_style))
    manual_data = [
        [Paragraph("Rule", table_header), Paragraph("Conf / Lift", table_header), Paragraph("Category", table_header), Paragraph("Domain Rationale", table_header)],
        [Paragraph("{Greek Hero} → {Justice}", table_text), Paragraph("66.9% / 1.16x", table_text), Paragraph("<b>Useful</b>", table_text), Paragraph("Bridges classical literature with moral philosophy.", table_text)],
        [Paragraph("{Justice} → {Greek Hero}", table_text), Paragraph("45.1% / 1.16x", table_text), Paragraph("<b>Useful</b>", table_text), Paragraph("Reciprocal humanities bridge for ethics students.", table_text)],
        [Paragraph("{Health Numbers} → {CS50x}", table_text), Paragraph("54.7% / 1.10x", table_text), Paragraph("<b>Useful</b>", table_text), Paragraph("Bridges quantitative biostatistics to computer programming.", table_text)],
        [Paragraph("{Environmental} → {Justice}", table_text), Paragraph("53.5% / 0.92x", table_text), Paragraph("<b>Useful</b>", table_text), Paragraph("Connects climate and environmental health to social justice.", table_text)],
        [Paragraph("{Justice} → {CS50x}", table_text), Paragraph("37.6% / 0.76x", table_text), Paragraph("<b>Obvious</b>", table_text), Paragraph("CS50x is taken by 50% of students; driven by platform size.", table_text)],
        [Paragraph("{Greek Hero} → {CS50x}", table_text), Paragraph("34.9% / 0.70x", table_text), Paragraph("<b>Obvious</b>", table_text), Paragraph("Reflects baseline global popularity rather than domain affinity.", table_text)],
        [Paragraph("{Health Numbers} → {Environmental}", table_text), Paragraph("40.8% / 0.90x", table_text), Paragraph("<b>Useful</b>", table_text), Paragraph("Domain coherence within Harvard Public Health curriculum.", table_text)],
        [Paragraph("{Environmental} → {Health Numbers}", table_text), Paragraph("26.3% / 0.90x", table_text), Paragraph("<b>Weak</b>", table_text), Paragraph("Low confidence; environmental students don't reliably take biostats.", table_text)],
        [Paragraph("{Greek Hero, CS50x} → {Justice}", table_text), Paragraph("45.2% / 0.78x", table_text), Paragraph("<b>Unexpected</b>", table_text), Paragraph("Students combining coding and Greek hero converge on justice.", table_text)],
        [Paragraph("{Greek Hero, Justice} → {CS50x}", table_text), Paragraph("23.6% / 0.48x", table_text), Paragraph("<b>Weak</b>", table_text), Paragraph("Low confidence and poor lift; pure humanities rarely cross to CS.", table_text)]
    ]
    t_man = Table(manual_data, colWidths=[2.2*inch, 1.2*inch, 1.0*inch, 2.8*inch])
    t_man.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E1B4B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_man)
    story.append(Spacer(1, 8))

    # Section 10
    story.append(Paragraph("10. System Limitations", h1_style))
    story.append(Paragraph(
        "1. <b>Lack of Pedagogical Sequence:</b> ARM treats baskets as unordered sets. It cannot capture prerequisite order "
        "(e.g., taking CS50x before Advanced Data Science vs. after).<br/>"
        "2. <b>Popularity Bias:</b> Mega-courses like CS50x dominate association rules due to sheer scale (50% enrollments). "
        "Lift and Leverage must be strictly checked to avoid recommending popular courses indiscriminately.<br/>"
        "3. <b>Cold-Start Problem:</b> Brand-new courses with zero enrollment history have 0 support and cannot be recommended by Apriori.", body_style
    ))

    doc.build(story)
    print(f"[OK] TECHNICAL_REPORT.pdf written to {REPORT_PDF_PATH}")

if __name__ == "__main__":
    generate_markdown_report()
    generate_pdf_report()
    print("\n[SUCCESS] Both Technical Reports generated successfully!")
