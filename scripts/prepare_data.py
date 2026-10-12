"""
prepare_data.py
Script to prepare and preprocess the HarvardX Person-Course dataset for CourseCompass.
"""

import os
import urllib.request
import pandas as pd

DATA_DIR = "Dataset"
RAW_FILENAME = "HXPC13_DI_v3_11-13-2019.csv"
CLEANED_FILENAME = "cleaned_course_enrollments.csv"
BASKETS_FILENAME = "learner_baskets.csv"

COURSE_METADATA = {
    "HarvardX/CB22x/2013_Spring": {
        "code": "CB22x",
        "title": "The Ancient Greek Hero",
        "instructor": "Gregory Nagy",
        "field": "Humanities & Classical Literature"
    },
    "HarvardX/CS50x/2012": {
        "code": "CS50x",
        "title": "Introduction to Computer Science I",
        "instructor": "David J. Malan",
        "field": "Computer Science & Programming"
    },
    "HarvardX/ER22x/2013_Spring": {
        "code": "ER22x",
        "title": "Justice: What's the Right Thing to Do?",
        "instructor": "Michael J. Sandel",
        "field": "Philosophy, Law & Ethics"
    },
    "HarvardX/PH207x/2012_Fall": {
        "code": "PH207x",
        "title": "Health in Numbers: Quantitative Methods in Clinical & Public Health Research",
        "instructor": "Marcello Pagano & E. Francis Cook",
        "field": "Public Health, Biostatistics & Epidemiology"
    },
    "HarvardX/PH278x/2013_Spring": {
        "code": "PH278x",
        "title": "Human Health and Global Environmental Change",
        "instructor": "Aaron Bernstein & John Spengler",
        "field": "Public Health & Environmental Sciences"
    }
}

def ensure_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, RAW_FILENAME)
    if os.path.exists(raw_path):
        print(f"[OK] Raw dataset found at: {raw_path} ({os.path.getsize(raw_path):,} bytes)")
        return raw_path

    print(f"[*] Downloading HarvardX Person-Course dataset from Harvard Dataverse...")
    print(f"    Source URL: {DATAVERSE_URL}")
    urllib.request.urlretrieve(DATAVERSE_URL, raw_path)
    print(f"[OK] Downloaded successfully: {raw_path}")
    return raw_path

def clean_and_prepare(raw_path):
    print("\n[1/3] Loading raw dataset...")
    df = pd.read_csv(raw_path, low_memory=False)
    initial_rows = len(df)
    print(f"      Initial record count: {initial_rows:,}")

    print("\n[2/3] Performing data preprocessing...")
    # Check nulls in core identifiers
    missing_users = df['userid_DI'].isnull().sum()
    missing_courses = df['course_id'].isnull().sum()
    print(f"      Missing userid_DI: {missing_users}")
    print(f"      Missing course_id: {missing_courses}")

    # Drop missing IDs if any
    df = df.dropna(subset=['userid_DI', 'course_id'])

    # Strip whitespace
    df['userid_DI'] = df['userid_DI'].astype(str).str.strip()
    df['course_id'] = df['course_id'].astype(str).str.strip()

    # Drop duplicate enrollments of the same course by the same learner
    initial_pairs = len(df)
    df = df.drop_duplicates(subset=['userid_DI', 'course_id'])
    dropped_dupes = initial_pairs - len(df)
    print(f"      Dropped duplicate (user, course) pairs: {dropped_dupes}")

    # Add friendly course metadata
    df['course_title'] = df['course_id'].map(lambda cid: COURSE_METADATA.get(cid, {}).get('title', cid))
    df['course_code'] = df['course_id'].map(lambda cid: COURSE_METADATA.get(cid, {}).get('code', cid))
    df['course_instructor'] = df['course_id'].map(lambda cid: COURSE_METADATA.get(cid, {}).get('instructor', 'HarvardX Faculty'))
    df['course_field'] = df['course_id'].map(lambda cid: COURSE_METADATA.get(cid, {}).get('field', 'General Studies'))

    cleaned_path = os.path.join(DATA_DIR, CLEANED_FILENAME)
    df.to_csv(cleaned_path, index=False)
    print(f"[OK] Cleaned records saved to: {cleaned_path} ({len(df):,} records)")

    print("\n[3/3] Creating learner-course baskets...")
    # Group courses by user
    baskets = df.groupby('userid_DI')['course_title'].apply(lambda s: sorted(list(set(s)))).reset_index()
    baskets['basket_size'] = baskets['course_title'].apply(len)
    baskets['courses_joined'] = baskets['course_title'].apply(lambda lst: " | ".join(lst))

    baskets_path = os.path.join(DATA_DIR, BASKETS_FILENAME)
    baskets.to_csv(baskets_path, index=False)
    print(f"[OK] Learner baskets saved to: {baskets_path}")
    print(f"      Total unique learners: {len(baskets):,}")
    print(f"      Multi-course learners (basket >= 2): {(baskets['basket_size'] >= 2).sum():,} ({(baskets['basket_size'] >= 2).mean()*100:.2f}%)")
    print(f"      Max courses in a single basket: {baskets['basket_size'].max()}")

    return df, baskets

if __name__ == "__main__":
    path = ensure_dataset()
    clean_and_prepare(path)
    print("\n[SUCCESS] Data preprocessing complete!")
