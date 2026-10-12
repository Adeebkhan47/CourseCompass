import pandas as pd
import os

data_path = os.path.join("Dataset", "HXPC13_DI_v3_11-13-2019.csv")
print("Loading dataset from:", data_path)
df = pd.read_csv(data_path, low_memory=False)

print(f"Total records (rows): {len(df):,}")
print(f"Total columns: {df.shape[1]}")
print("Columns:", list(df.columns))

print("\nMissing values per column:")
print(df.isnull().sum())

unique_users = df['userid_DI'].nunique()
unique_courses = df['course_id'].nunique()
print(f"\nUnique learners (userid_DI): {unique_users:,}")
print(f"Unique courses (course_id): {unique_courses}")

print("\nCourse distribution (enrollments per course):")
course_counts = df['course_id'].value_counts()
for course, count in course_counts.items():
    print(f"  {course}: {count:,}")

# Analyze user baskets
user_counts = df.groupby('userid_DI')['course_id'].nunique()
print("\nUser enrollment distribution:")
print(f"Learners with 1 course: {(user_counts == 1).sum():,} ({(user_counts == 1).mean()*100:.2f}%)")
print(f"Learners with 2 courses: {(user_counts == 2).sum():,} ({(user_counts == 2).mean()*100:.2f}%)")
print(f"Learners with 3+ courses: {(user_counts >= 3).sum():,} ({(user_counts >= 3).mean()*100:.2f}%)")
print(f"Max courses taken by a single user: {user_counts.max()}")

multi_course_users = (user_counts >= 2).sum()
print(f"\nLearners with multi-course enrollments (>=2): {multi_course_users:,}")

# Check duplicates: (userid_DI, course_id)
duplicate_pairs = df.duplicated(subset=['userid_DI', 'course_id']).sum()
print(f"\nDuplicate (userid_DI, course_id) pairs: {duplicate_pairs:,}")
