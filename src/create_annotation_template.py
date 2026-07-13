import os
import pandas as pd

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "guardian_posts_clean.csv"))

# select 50 articles per topic randomly (fixed seed for reproducibility)
frames = []
for topic in df['topic'].unique():
    sample = df[df['topic'] == topic].sample(50, random_state=42)
    frames.append(sample)

annotation = pd.concat(frames, ignore_index=True)

# keep only the columns annotators need to see
annotation = annotation[['post_id', 'topic', 'text_clean']].copy()
annotation = annotation.rename(columns={'text_clean': 'text'})

# add empty columns for each team member
# allowed values: positive / neutral / negative
annotation['member_1'] = ''
annotation['member_2'] = ''
annotation['member_3'] = ''
annotation['member_4'] = ''

print("Annotation template created:", len(annotation), "articles")
print(annotation.groupby('topic').size().to_string())

# create output folder if it does not exist
os.makedirs(os.path.join(BASE_DIR, "data", "annotated"), exist_ok=True)

# save as CSV to upload to Google Sheets
annotation.to_csv(os.path.join(BASE_DIR, "data", "annotated", "annotation_template.csv"),
                  index=False, encoding="utf-8")
print()
print("Saved: data/annotated/annotation_template.csv")
print()
print("Instructions:")
print("Each member fills only their own column independently")
print("Allowed values: positive / neutral / negative")