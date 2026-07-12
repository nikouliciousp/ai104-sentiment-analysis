import pandas as pd
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    text = str(text)

    # convert to lowercase
    text = text.lower()

    # remove Guardian smart quotes and dashes
    text = text.replace('\u201c', ' ').replace('\u201d', ' ')
    text = text.replace('\u2018', ' ').replace('\u2019', ' ')
    text = text.replace('\u2013', ' ').replace('\u2014', ' ')

    # keep only letters and spaces, remove digits and punctuation
    text = re.sub(r'[^a-z\s]', ' ', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # remove stopwords and words shorter than 3 characters
    tokens = [
        word for word in text.split()
        if word not in STOP_WORDS and len(word) > 2
    ]

    return ' '.join(tokens)


def main():
    df = pd.read_csv("data/raw/guardian_posts_raw.csv")
    print("Loaded:", len(df), "articles")

    # apply cleaning to all articles
    df['text_clean'] = df['text'].apply(clean_text)

    # drop articles that are too short after cleaning
    before = len(df)
    df = df[df['text_clean'].str.len() > 50].reset_index(drop=True)
    after = len(df)
    print("Dropped:", before - after, "| Remaining:", after)

    print()
    print(df.groupby("topic")["post_id"].count().to_string())

    # save cleaned dataset
    df.to_csv("data/clean/guardian_posts_clean.csv",
              index=False, encoding="utf-8")
    print()
    print("Saved: data/clean/guardian_posts_clean.csv")

    # show before and after example
    print()
    print("Before:")
    print(df['text'].iloc[0][:200])
    print()
    print("After:")
    print(df['text_clean'].iloc[0][:200])


if __name__ == "__main__":
    main()
