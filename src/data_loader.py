"""
data_loader.py
Data loading, caching, and basket transformation module for CourseCompass.
"""

import os
import pandas as pd
from typing import Tuple, Dict, List, Set

COURSE_METADATA = {
    "HarvardX/CB22x/2013_Spring": {
        "code": "CB22x",
        "title": "The Ancient Greek Hero",
        "short_title": "Ancient Greek Hero",
        "instructor": "Gregory Nagy",
        "field": "Humanities & Classical Literature",
        "description": "An exploration of classical Greek literature, epic poetry, and heroic traditions."
    },
    "HarvardX/CS50x/2012": {
        "code": "CS50x",
        "title": "Introduction to Computer Science I",
        "short_title": "CS50: Intro to Computer Science",
        "instructor": "David J. Malan",
        "field": "Computer Science & Programming",
        "description": "An introduction to the intellectual enterprises of computer science and the art of programming."
    },
    "HarvardX/ER22x/2013_Spring": {
        "code": "ER22x",
        "title": "Justice: What's the Right Thing to Do?",
        "short_title": "Justice (Ethics & Philosophy)",
        "instructor": "Michael J. Sandel",
        "field": "Philosophy, Law & Ethics",
        "description": "A celebrated introduction to moral and political philosophy and constitutional law."
    },
    "HarvardX/PH207x/2012_Fall": {
        "code": "PH207x",
        "title": "Health in Numbers: Quantitative Methods in Clinical & Public Health Research",
        "short_title": "Health in Numbers (Biostats)",
        "instructor": "Marcello Pagano & E. Francis Cook",
        "field": "Public Health, Biostatistics & Epidemiology",
        "description": "Foundational principles of biostatistics and epidemiology in public health research."
    },
    "HarvardX/PH278x/2013_Spring": {
        "code": "PH278x",
        "title": "Human Health and Global Environmental Change",
        "short_title": "Global Environmental Health",
        "instructor": "Aaron Bernstein & John Spengler",
        "field": "Public Health & Environmental Sciences",
        "description": "Understanding the direct impacts of climate, environmental degradation, and ecology on health."
    }
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Candidate paths for robust discovery
CLEANED_CANDIDATES = [
    os.path.join(BASE_DIR, "Dataset", "Cleaned Data", "cleaned_course_enrollments.csv"),
    os.path.join(BASE_DIR, "Dataset", "cleaned_course_enrollments.csv"),
    os.path.join("Dataset", "Cleaned Data", "cleaned_course_enrollments.csv"),
    os.path.join("Dataset", "cleaned_course_enrollments.csv")
]

RAW_CANDIDATES = [
    os.path.join(BASE_DIR, "Dataset", "Raw Data", "HXPC13_DI_v3_11-13-2019.csv"),
    os.path.join(BASE_DIR, "Dataset", "HXPC13_DI_v3_11-13-2019.csv"),
    os.path.join("Dataset", "Raw Data", "HXPC13_DI_v3_11-13-2019.csv"),
    os.path.join("Dataset", "HXPC13_DI_v3_11-13-2019.csv")
]

BASKET_CANDIDATES = [
    os.path.join(BASE_DIR, "Dataset", "Cleaned Data", "learner_baskets.csv"),
    os.path.join(BASE_DIR, "Dataset", "learner_baskets.csv"),
    os.path.join("Dataset", "Cleaned Data", "learner_baskets.csv"),
    os.path.join("Dataset", "learner_baskets.csv")
]

def get_course_title(course_id: str) -> str:
    meta = COURSE_METADATA.get(course_id)
    if meta:
        return f"{meta['code']}: {meta['title']}"
    return course_id

def load_preprocessed_data() -> pd.DataFrame:
    """Loads cleaned enrollments dataset. Searches all candidate locations."""
    for cleaned_path in CLEANED_CANDIDATES:
        if os.path.exists(cleaned_path):
            df = pd.read_csv(cleaned_path, low_memory=False)
            df['course_title'] = df['course_id'].map(get_course_title)
            return df
    
    # Fallback to raw file
    for raw_path in RAW_CANDIDATES:
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path, low_memory=False)
            df = df.dropna(subset=['userid_DI', 'course_id'])
            df['userid_DI'] = df['userid_DI'].astype(str).str.strip()
            df['course_id'] = df['course_id'].astype(str).str.strip()
            df = df.drop_duplicates(subset=['userid_DI', 'course_id'])
            df['course_title'] = df['course_id'].map(get_course_title)
            return df
    
    raise FileNotFoundError(f"Could not find cleaned dataset or raw dataset in candidate paths: {CLEANED_CANDIDATES + RAW_CANDIDATES}")

def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """Calculates dataset summary metrics."""
    user_counts = df.groupby('userid_DI')['course_id'].nunique()
    total_records = len(df)
    unique_learners = int(df['userid_DI'].nunique())
    unique_courses = int(df['course_id'].nunique())
    multi_course_learners = int((user_counts >= 2).sum())
    single_course_learners = int((user_counts == 1).sum())
    max_basket = int(user_counts.max()) if len(user_counts) > 0 else 0
    avg_basket = float(user_counts.mean()) if len(user_counts) > 0 else 0.0

    return {
        "total_records": total_records,
        "unique_learners": unique_learners,
        "unique_courses": unique_courses,
        "multi_course_learners": multi_course_learners,
        "single_course_learners": single_course_learners,
        "multi_course_pct": (multi_course_learners / unique_learners * 100) if unique_learners > 0 else 0,
        "max_basket": max_basket,
        "avg_basket": avg_basket
    }

def build_transactions(df: pd.DataFrame, multi_only: bool = True) -> Tuple[List[Set[str]], pd.DataFrame]:
    """
    Builds transactions for association rule mining.
    - multi_only=True: Filters to learners enrolled in >= 2 courses (recommended for MBA).
    - multi_only=False: All learners (including single-course enrollments).
    Returns:
        - transactions: list of sets of course titles
        - one_hot_df: boolean DataFrame of users x courses
    """
    # Group courses by user
    grouped = df.groupby('userid_DI')['course_title'].apply(lambda x: sorted(list(set(x))))
    
    if multi_only:
        grouped = grouped[grouped.apply(len) >= 2]

    transactions = [set(courses) for courses in grouped]
    
    # Build boolean cross-tab / one-hot representation
    df_filtered = df[df['userid_DI'].isin(grouped.index)]
    one_hot_df = pd.crosstab(df_filtered['userid_DI'], df_filtered['course_title']).astype(bool)
    
    return transactions, one_hot_df
