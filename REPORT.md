# CourseCompass: Association Rule Mining–Based Course Recommendation System
## Applied Machine Learning  — Assignment #01 Technical Report  

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
   $$\mathcal{T} = \{ T_u \mid u \in 	ext{Learners}, T_u = \{ c_{u,1}, c_{u,2}, \dots, c_{u,k} \} \}$$
5. **Dual Population Modeling:** Implemented dual transaction views:
   - **Multi-Course Cohort ($N = 30,267$):** Focuses on students demonstrating cross-course behavior, preventing 90% single-course drop-in skew.
   - **Full Platform Cohort ($N = 301,609$):** Global platform perspective with calibrated micro-support thresholds.

---

## 4. Association Rule Mining Methodology
### 4.1 Core Mathematical Definitions

| Metric / Term | Formula / Definition | Simple Interpretation |
| :--- | :--- | :--- |
| **Support** | $\text{support}(A \to B) = P(A \cap B) = \frac{\text{count}(A \cup B)}{N}$ | How frequently courses $A$ and $B$ occur together across all learner baskets. |
| **Confidence** | $\text{confidence}(A \to B) = P(B \mid A) = \frac{\text{support}(A \cup B)}{\text{support}(A)}$ | The proportion of learners enrolled in $A$ who also enrolled in course $B$. |
| **Lift** | $\text{lift}(A \to B) = \frac{P(B \mid A)}{P(B)} = \frac{\text{confidence}(A \to B)}{\text{support}(B)}$ | Ratio of observed co-enrollment to expected co-enrollment under independence. Lift > 1 indicates positive affinity. |
| **Leverage** | $\text{leverage}(A \to B) = P(A \cap B) - P(A) \cdot P(B)$ | Absolute excess co-occurrence frequency above expected independence. |
| **Conviction** | $\text{conviction}(A \to B) = \frac{1 - P(B)}{1 - \text{confidence}(A \to B)}$ | Measures the degree of implication; infinite for perfect rules. |

### 4.2 Mathematical Derivation: Why Confidence(A → B) ≠ Confidence(B → A)
While Joint Support is strictly symmetric:
$$\text{support}(A \to B) = \text{support}(B \to A) = P(A \cap B)$$

Rule Confidence is fundamentally asymmetric:
$$\text{confidence}(A \to B) = \frac{P(A \cap B)}{P(A)}, \quad \text{confidence}(B \to A) = \frac{P(A \cap B)}{P(B)}$$

Whenever $P(A) \neq P(B)$, the denominators differ. In educational datasets, course popularity varies widely (e.g., Justice has 57,406 enrollments while Greek Hero has 30,002). Consequently:
$$\text{confidence}(\text{Greek Hero} \to \text{Justice}) = \frac{0.2609}{0.3902} = \mathbf{66.87\%}$$
$$\text{confidence}(\text{Justice} \to \text{Greek Hero}) = \frac{0.2609}{0.5785} = \mathbf{45.11\%}$$
The $21.76\%$ difference arises because a Greek Hero student is much more likely to take Justice than a Justice student is to take Greek Hero.

---

## 5. Apriori Explanation and Hand-Worked Example
### 5.1 The Apriori Principle (Downward Closure Property)
> **Theorem:** *If an itemset is frequent, then all of its subsets must also be frequent.*  
> **Contrapositive (Pruning Rule):** *If any subset of an itemset is infrequent, the superset cannot be frequent and is pruned immediately.*

### 5.2 Hand-Worked Example
Consider a toy educational database with $N = 5$ learner baskets and threshold $\text{min\_sup} = 0.40$ (minimum count $= 2$):
- Basket 1: $\{ \text{CS}, \text{Ethics} \}$
- Basket 2: $\{ \text{CS}, \text{Stats}, \text{Ethics} \}$
- Basket 3: $\{ \text{Stats}, \text{Bio} \}$
- Basket 4: $\{ \text{CS}, \text{Stats}, \text{Bio} \}$
- Basket 5: $\{ \text{CS}, \text{Ethics} \}$

#### Step 1: Candidate Generation $C_1$ & Pruning to $L_1$
- $\text{CS}$: Count $= 4 \implies \text{sup} = 0.80 \ge 0.40$ (Frequent)
- $\text{Ethics}$: Count $= 3 \implies \text{sup} = 0.60 \ge 0.40$ (Frequent)
- $\text{Stats}$: Count $= 3 \implies \text{sup} = 0.60 \ge 0.40$ (Frequent)
- $\text{Bio}$: Count $= 2 \implies \text{sup} = 0.40 \ge 0.40$ (Frequent)
- **Result $L_1$:** $\{ \{\text{CS}\}, \{\text{Ethics}\}, \{\text{Stats}\}, \{\text{Bio}\} \}$

#### Step 2: Candidate Generation $C_2$ ($L_1 \bowtie L_1$)
- Pairs: $\{ \text{CS}, \text{Ethics} \}$, $\{ \text{CS}, \text{Stats} \}$, $\{ \text{CS}, \text{Bio} \}$, $\{ \text{Ethics}, \text{Stats} \}$, $\{ \text{Ethics}, \text{Bio} \}$, $\{ \text{Stats}, \text{Bio} \}$
- Support counts:
  - $\{\text{CS}, \text{Ethics}\}$: Count $= 3 \implies 0.60$ (Frequent)
  - $\{\text{CS}, \text{Stats}\}$: Count $= 2 \implies 0.40$ (Frequent)
  - $\{\text{CS}, \text{Bio}\}$: Count $= 1 \implies 0.20$ (Infrequent $\to$ Pruned by support)
  - $\{\text{Ethics}, \text{Stats}\}$: Count $= 1 \implies 0.20$ (Infrequent $\to$ Pruned by support)
  - $\{\text{Ethics}, \text{Bio}\}$: Count $= 0 \implies 0.00$ (Infrequent $\to$ Pruned by support)
  - $\{\text{Stats}, \text{Bio}\}$: Count $= 2 \implies 0.40$ (Frequent)
- **Result $L_2$:** $\{ \{\text{CS}, \text{Ethics}\}, \{\text{CS}, \text{Stats}\}, \{\text{Stats}, \text{Bio}\} \}$

#### Step 3: Candidate Generation $C_3$ ($L_2 \bowtie L_2$) & Apriori Pruning
- Joining candidate $\{\text{CS}, \text{Ethics}, \text{Stats}\}$:
  - Subsets: $\{\text{CS}, \text{Ethics}\} \in L_2$, $\{\text{CS}, \text{Stats}\} \in L_2$, but $\{\text{Ethics}, \text{Stats}\} \notin L_2$!
  - **Pruning Action:** Pruned prior to scanning transaction records!
- No valid candidates survive. Algorithm terminates.

---

## 6. Discovered Frequent Itemsets and Association Rules
At $\text{min\_sup} = 0.05$ and $\text{min\_conf} = 0.15$ on multi-course learner baskets ($N = 30,267$), the algorithm discovers 15 frequent itemsets and 25 association rules.

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
1. **Zero Self-Recommendation:** If a user selects $\{ \text{CB22x: Greek Hero} \}$, the candidate pool strictly excludes $\text{CB22x}$.
2. **Evidence Aggregation:** If multiple rules point to course $C$, all rules are collected into an evidence dossier.
3. **Multi-Strategy Ranking:**
   - **Confidence-First:** $S = \max(\text{confidence})$
   - **Lift-First:** $S = \max(\text{lift})$
   - **Hybrid Score:** $S = \max(\text{confidence}) \times \sqrt{\max(\text{lift})} \times [1 + 0.05 \min(\text{count}-1, 4)]$

---

## 8. Threshold Experiments & Outcome Comparison
We evaluated the behavior of CourseCompass across two operational settings:

| Metric | Setting A (Conservative) | Setting B (Exploratory) |
| :--- | :---: | :---: |
| **Minimum Support ($min\_sup$)** | **0.10** | **0.03** |
| **Minimum Confidence ($min\_conf$)** | **0.40** | **0.15** |
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
   Standard ARM treats baskets as unordered sets ($\{A, B\}$). In education, course sequence is crucial: taking Introduction to CS *before* Biostatistics is natural, but the reverse order may lead to severe frustration. ARM cannot represent prerequisite graphs without sequence-mining extensions (e.g., GSP, PrefixSpan).
2. **Super-Item Popularity Bias:**  
   When a single course (such as CS50x) commands 50%+ enrollment, it naturally appears in almost every rule consequent. Without filtering by Lift or Leverage, the system risks recommending the most popular course to everyone regardless of specialization.
3. **Cold-Start Vulnerability:**  
   New courses have 0 transactions, resulting in 0 support. Apriori completely omits new courses until a substantial volume of co-enrollments accumulates.

---
*Report generated automatically from verified empirical calculations on HarvardX dataset.*
