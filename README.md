# CourseCompass 🧭

## 📖 Executive Summary
**CourseCompass** is an interactive, explainable course recommendation system that applies Market Basket Analysis and Association Rule Mining to learner enrollment trajectories. Using the official de-identified **HarvardX Person-Course Academic Year 2013 Dataset**, the system models each learner's course enrollments as a distinct transaction basket, executes a transparent **Apriori Algorithm** with candidate generation and downward-closure pruning, calculates directional rule metrics (Support, Confidence, Lift, Leverage, Conviction), enforces strict exclusion of already-enrolled courses from recommendations, and provides rich dynamic "Why this recommendation?" rationales through a modern Streamlit interface.

---

## 🚀 Key Features

1. **📊 Dataset Explorer & Statistics:**
   - Real-world HarvardX dataset containing **338,223 enrollment records** across **301,609 unique learners** and 5 flagship courses.
   - Dual-cohort modeling: Focus on **Multi-Course Baskets ($N=30,267$)** to prevent single-course dilution, or explore the Full Platform cohort ($N=301,609$).
   - Full course catalog with instructors, academic divisions, descriptions, and co-enrollment rates.

2. **⚙️ Traceable Apriori Engine & Pruning:**
   - Pure Python Apriori implementation with explicit step-by-step candidate generation ($C_k$) and subset pruning ($L_k$).
   - Live visual waterfall chart of candidate generation vs. pruning.
   - Complete mathematical derivation and empirical proof of why $\text{Confidence}(A \to B) \neq \text{Confidence}(B \to A)$.

3. **🔍 Interactive Rules Explorer:**
   - Filter and search association rules by course, length, support, confidence, or lift.
   - Natural language explanations for every rule (e.g., *"Learners who enrolled in [The Ancient Greek Hero] have a 66.9% chance of taking [Justice] (1.16x baseline)"*).
   - Export discovered itemsets and rules as CSV.

4. **🎯 Smart Recommendation Basket:**
   - Interactive multi-course selection basket.
   - **Strict Invariant:** Guarantees **zero self-recommendation** (never recommends courses already in the user's basket).
   - Multi-rule evidence dossier for every suggested course.
   - Documented ranking strategies: **Confidence-First**, **Lift-First**, and **Hybrid Score**.

5. **🔄 What-If Counterfactual Simulator:**
   - Simulates the pedagogical impact when a user removes a course from their basket.
   - Displays newly unlocked courses vs. lost recommendations in real time.

6. **🎛️ Threshold Playground & Side-by-Side Comparison:**
   - Compares two distinct threshold regimes (Conservative Setting A vs. Exploratory Setting B) side-by-side.
   - Compares itemset counts, rule quantities, average confidence/lift, and recommendation outcomes on test baskets.

7. **📈 Visual Analytics & Association Graph:**
   - Directed Network Graph of Course Associations (nodes = courses, directed edges = rules, thickness = confidence, color = lift).
   - Frequency distribution, basket size histogram, and Support-Confidence-Lift scatter landscape.

8. **📝 Empirical Evaluation & Manual Rule Audit:**
   - 5 end-to-end recommendation scenarios evaluated against domain pedagogical goals.
   - 10 discovered rules manually audited and categorized into **Useful**, **Obvious/Trivial**, **Unexpected**, or **Weak/Uncertain**.
   - In-depth analysis of ARM limitations in education (pedagogical sequencing, popularity bias, cold-start).

---

## 📂 Project Directory Structure

```text
CourseCompass/
├── Dataset/
│   ├── Cleaned Data/
│   │   ├── cleaned_course_enrollments.csv  # Cleaned, standardized enrollment records (N=338,223)
│   │   └── learner_baskets.csv             # Learner-level course baskets
│   └── Raw Data/
│       ├── HXPC13_DI_v3_11-13-2019.csv     # Official HarvardX dataset (Harvard Dataverse)
│       ├── Person Course Deidentification.pdf
│       └── Person Course Documentation.pdf
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Caching, summary statistics, dual-cohort basket builder
│   ├── apriori_engine.py    # Custom Apriori algorithm, candidate generation, rule mining
│   ├── recommender.py       # Rule-based recommender, evidence aggregation, what-if simulator
│   └── visualizations.py   # NetworkX graphs, distributions, scatter plots, pruning charts
├── scripts/
│   ├── prepare_data.py      # Automated download & preprocessing script
│   ├── verify_system.py     # 9-point end-to-end automated verification test suite
│   ├── test_rules.py        # Rule exploration utility
│   └── inspect_data.py      # Dataset inspection utility
├── app.py                   # Streamlit web application
├── requirements.txt         # Project dependencies
├── REPORT.md                # Earlier Markdown draft
├── TECHNICAL_REPORT.docx   # Final Word technical report
└── README.md                # Documentation and run instructions
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### 1. Clone or Open the Repository
```bash
git clone https://github.com/Adeebkhan47/CourseCompass.git
cd CourseCompass
```

### 2. Install Dependencies
```bash
py -m pip install -r requirements.txt
```
*(If on Linux/macOS, use `pip install -r requirements.txt` or `python3 -m pip install -r requirements.txt`)*

### 3. Verify Dataset (Optional if already preprocessed)
The dataset is already organized in `Dataset/Cleaned Data/`. If you wish to re-run the cleaning pipeline or download fresh data from Harvard Dataverse:
```bash
py scripts/prepare_data.py
```

---

## 💻 Running the Application

Launch the Streamlit web app with:
```bash
py -m streamlit run app.py
```
Or directly:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Verification & Testing

An end-to-end test suite validates all 9 functional requirements, mathematical formulas, basket exclusions, ranking strategies, and chart generators:
```bash
py scripts/verify_system.py
```

Expected output:
```text
==================================================
      COURSECOMPASS SYSTEM VERIFICATION SUITE      
==================================================
[TEST 1] Loading preprocessed dataset...
  ✓ Records: 338,223
  ✓ Unique learners: 301,609
  ✓ Multi-course baskets: 30,267
[TEST 2] Verifying Basket Construction...
  ✓ Multi-course transactions: 30,267 baskets
  ✓ All-learners transactions: 301,609 baskets
[TEST 3] Verifying Apriori Candidate Generation & Pruning...
  ✓ Discovered frequent itemsets: 15
  ✓ Apriori execution levels traced: 3
[TEST 4] Verifying Association Rule Generation...
  ✓ Generated rules: 25
[TEST 5] Checking Confidence Asymmetry (A -> B != B -> A)...
  ✓ Conf(Greek Hero -> Justice): 66.87%
  ✓ Conf(Justice -> Greek Hero): 45.11%
  ✓ Asymmetry confirmed! Absolute delta: 21.76%
[TEST 6] Verifying Recommendation Engine & Basket Exclusion...
  ✓ Recommended courses strictly exclude enrolled courses
[TEST 7] Verifying Ranking Strategies (Confidence, Lift, Hybrid)...
[TEST 8] Verifying What-If Analysis...
[TEST 9] Verifying Visual Analytics Generators...
  ✓ All 5 charts generated without error!
==================================================
      ALL 9 VERIFICATION TESTS PASSED (100%)       
==================================================
```

---

## 📄 Technical Report

Per Assignment Section 8 guidelines, the final technical report is available as a Word document:
- **Word:** [`TECHNICAL_REPORT.docx`](TECHNICAL_REPORT.docx)

`REPORT.md` is the earlier Markdown draft.

---

## 🎓 Academic Attribution
- **Dataset:** HarvardX-MITx Person-Course Academic Year 2013 De-Identified Dataset, Version 3.0 (Harvard Dataverse, DOI: `10.7910/DVN/26147`).
- **Course:** AI3002 - Applied Machine Learning, National University of Computer & Emerging Sciences (FAST-NUCES).
