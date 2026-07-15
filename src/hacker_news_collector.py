import os
import requests
import pandas as pd
import re
import time

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOPICS = {
    "Artificial Intelligence": [
        "chatgpt",
        "openai",
        "artificial intelligence",
        "large language model",
        "machine learning",
    ],
    "Cryptocurrency": [
        "bitcoin",
        "ethereum",
        "cryptocurrency",
        "crypto",
        "blockchain",
    ],
    "Climate Change": [
        "climate change",
        "global warming",
        "carbon emissions",
        "renewable energy",
        "fossil fuels",
    ],
    "Remote Work": [
        "remote work",
        "work from home",
        "wfh",
        "fully remote",
        "distributed team",
        "remote job",
        "async work",
    ],
}

# keywords for relevance filtering
TOPIC_KEYWORDS = {
    "Artificial Intelligence": [
        "ai", "chatgpt", "openai", "llm",
        "machine learning", "neural", "gpt"
    ],
    "Cryptocurrency": [
        "bitcoin", "crypto", "ethereum",
        "blockchain", "btc", "eth"
    ],
    "Climate Change": [
        "climate", "warming", "carbon",
        "emission", "renewable", "fossil"
    ],
    "Remote Work": [
        "remote work", "work from home", "wfh",
        "fully remote", "remote job",
        "distributed team", "telework", "async work"
    ],
}

# minimum keyword occurrences per topic
MIN_KEYWORD_COUNT = {
    "Artificial Intelligence": 1,
    "Cryptocurrency":          2,
    "Climate Change":          1,
    "Remote Work":             1,
}

# quarterly time windows from Jan 2024 to Jul 2026
# each tuple: (label, start_timestamp, end_timestamp)
TIME_WINDOWS = [
    ("Q1-2024", 1704067200, 1711929600),
    ("Q2-2024", 1711929600, 1719792000),
    ("Q3-2024", 1719792000, 1727740800),
    ("Q4-2024", 1727740800, 1735689600),
    ("Q1-2025", 1735689600, 1743465600),
    ("Q2-2025", 1743465600, 1751328000),
    ("Q3-2025", 1751328000, 1759276800),
    ("Q4-2025", 1759276800, 1767225600),
    ("Q1-2026", 1767225600, 1775001600),
    ("Q2-2026", 1775001600, 1783065600),
]


def clean_html(text):
    # remove html tags only — full text cleaning done separately
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_relevant(text, topic):
    # check if topic keywords appear in the text
    text_lower = text.lower()
    keywords   = TOPIC_KEYWORDS[topic]
    count      = sum(text_lower.count(kw) for kw in keywords)
    return count >= MIN_KEYWORD_COUNT[topic]


def fetch_window(query, topic, start_ts, end_ts, max_per_window=10):
    # collect posts within a specific time window
    rows = []
    seen = set()
    page = 0

    while len(rows) < max_per_window:
        params = {
            "query":          query,
            "tags":           "comment",
            "hitsPerPage":    50,
            "page":           page,
            "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
        }
        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search_by_date",
                params=params,
                timeout=15
            )
            if resp.status_code != 200:
                break

            hits = resp.json().get("hits", [])
            if not hits:
                break

            for h in hits:
                post_id = h.get("objectID", "")
                if post_id in seen:
                    continue

                text = clean_html(h.get("comment_text", "") or "")

                # keep only short posts (social media style)
                if len(text) < 30 or len(text) > 500:
                    continue

                # skip off-topic comments
                if not is_relevant(text, topic):
                    continue

                seen.add(post_id)
                rows.append({
                    "post_id":    post_id,
                    "text":       text,
                    "created_at": h.get("created_at", ""),
                    "url":        f"https://news.ycombinator.com/item?id={post_id}",
                    "topic":      topic,
                })

                if len(rows) >= max_per_window:
                    break

            page += 1
            if page > 5:
                break

        except Exception as e:
            print("Error:", e)
            break

        time.sleep(0.3)

    return rows


def fetch_topic(topic, queries, target=300):
    # collect posts spread across all time windows
    topic_rows = []
    seen_ids   = set()

    # calculate posts per window to spread evenly
    posts_per_window = max(5, target // len(TIME_WINDOWS))

    for query in queries:
        if len(topic_rows) >= target:
            break

        for label, start_ts, end_ts in TIME_WINDOWS:
            if len(topic_rows) >= target:
                break

            rows = fetch_window(query, topic, start_ts, end_ts,
                                max_per_window=posts_per_window)
            added = 0
            for r in rows:
                if r["post_id"] not in seen_ids:
                    seen_ids.add(r["post_id"])
                    topic_rows.append(r)
                    added += 1

            if added > 0:
                print(f"  {label} | query: {query} | added: {added} | total: {len(topic_rows)}")

            time.sleep(0.5)

    return topic_rows


def main():
    all_rows = []

    for topic, queries in TOPICS.items():
        print("Topic:", topic)
        topic_rows = fetch_topic(topic, queries, target=300)
        all_rows.extend(topic_rows)
        print("  topic total:", len(topic_rows))
        print()

    df = pd.DataFrame(all_rows)

    print(df.groupby("topic").agg(
        posts     = ("post_id",    "count"),
        date_from = ("created_at", "min"),
        date_to   = ("created_at", "max"),
        avg_len   = ("text",       lambda x: int(x.str.len().mean())),
    ).to_string())

    # create output folder if it does not exist
    os.makedirs(os.path.join(BASE_DIR, "data", "raw"), exist_ok=True)

    output_path = os.path.join(BASE_DIR, "data", "raw", "hackernews_posts_raw.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print()
    print("Saved:", output_path, "| rows:", len(df))

    # print first 10 posts per topic
    for topic in df["topic"].unique():
        print()
        print("Topic:", topic)
        for _, row in df[df["topic"] == topic].head(10).iterrows():
            print(" ", row["created_at"][:10], "|", row["text"][:150])


if __name__ == "__main__":
    main()