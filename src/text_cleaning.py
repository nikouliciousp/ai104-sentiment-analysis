import pandas as pd
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+',   ' ', text)
    text = re.sub(r'@\w+',      ' ', text)
    text = re.sub(r'#\w+',      ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+',       ' ', text).strip()

    tokens = [w for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)


def main():
    df = pd.read_csv("data/raw/guardian_posts_raw.csv")
    print("Loaded:", len(df), "articles")

    df['text_clean'] = df['text'].apply(clean_text)

    before = len(df)
    df     = df[df['text_clean'].str.len() > 50].reset_index(drop=True)
    after  = len(df)

    print("Removed:", before - after, "| Remaining:", after)
    print(df.groupby("topic")["post_id"].count())

    df.to_csv("data/clean/guardian_posts_clean.csv",
              index=False, encoding="utf-8")
    print("Saved: data/clean/guardian_posts_clean.csv")

    print("\nBefore cleaning:")
    print(df['text'].iloc[0][:200])
    print("\nAfter cleaning:")
    print(df['text_clean'].iloc[0][:200])


if __name__ == "__main__":
    main()
