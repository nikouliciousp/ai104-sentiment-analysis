import os
import requests
import pandas as pd
import re
import html
import time

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# search queries per topic
TOPICS = {
    "Artificial Intelligence": [
        "chatgpt", "openai", "artificial intelligence",
        "large language model", "machine learning",
    ],
    "Cryptocurrency": [
        "bitcoin", "ethereum", "cryptocurrency",
        "crypto", "blockchain",
    ],
    "Climate Change": [
        "climate change", "global warming",
        "carbon emissions", "renewable energy", "fossil fuels",
    ],
    "Cybersecurity": [
        "cybersecurity", "ransomware", "data breach",
        "cyber attack", "malware", "hacking",
    ],
}

# keywords used to verify a post is actually about the topic
TOPIC_KEYWORDS = {
    "Artificial Intelligence": ["ai", "chatgpt", "openai", "llm", "machine learning", "neural", "gpt"],
    "Cryptocurrency":          ["bitcoin", "crypto", "ethereum", "blockchain", "btc", "eth"],
    "Climate Change":          ["climate", "warming", "carbon", "emission", "renewable", "fossil"],
    "Cybersecurity":           ["cybersecurity", "ransomware", "malware", "data breach",
                                "cyber attack", "hacker", "phishing", "vulnerability", "exploit"],
}

# minimum keyword hits for a post to be considered relevant
MIN_KEYWORD_COUNT = {
    "Artificial Intelligence": 1,
    "Cryptocurrency":          1,
    "Climate Change":          1,
    "Cybersecurity":           1,
}

# short keywords that need word boundary to avoid false matches
# e.g. "eth" matches "weather", "gpt" matches "egypt", "ai" matches "email"
# "neural" matches "neurons", "neurally"
NEEDS_BOUNDARY = {"ai", "eth", "btc", "gpt", "llm", "wfh", "neural"}

# quarterly time windows from Jan 2024 to Jul 2026
# searched first so posts are spread evenly across time
TIME_WINDOWS = [
    ("Q2-2026", 1775001600, 1783065600),
    ("Q1-2026", 1767225600, 1775001600),
    ("Q4-2025", 1759276800, 1767225600),
    ("Q3-2025", 1751328000, 1759276800),
    ("Q2-2025", 1743465600, 1751328000),
    ("Q1-2025", 1735689600, 1743465600),
    ("Q4-2024", 1727740800, 1735689600),
    ("Q3-2024", 1719792000, 1727740800),
    ("Q2-2024", 1711929600, 1719792000),
    ("Q1-2024", 1704067200, 1711929600),
    ("Q4-2023", 1696118400, 1704067200),
    ("Q3-2023", 1688169600, 1696118400),
    ("Q2-2023", 1680307200, 1688169600),
    ("Q1-2023", 1672531200, 1680307200),
]

MAX_WORDS   = 12
TARGET      = 300
MAX_PAGES   = 8
MAX_RESULTS = 8
MAX_RETRY   = 3
SLEEP_HIT   = 0.3
SLEEP_QUERY = 0.5
RETRY_WAIT  = 5


def clean_text(text):
    # remove html tags
    text = re.sub(r'<[^>]+>', ' ', str(text))
    # decode html entities such as &quot; and &#x27;
    text = html.unescape(text)
    # remove mentions (@username) completely but keep hashtag words (#bitcoin → bitcoin)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    # remove urls so they are not counted as words
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    # keep only letters, digits, spaces, apostrophes and hyphens
    text = re.sub(r"[^a-zA-Z0-9\s\'\-]", ' ', text)
    # remove hyphens or apostrophes that are left standing alone between words
    text = re.sub(r"\s[\'\-]\s", ' ', text)
    # remove leading or trailing hyphens and apostrophes
    text = text.strip("'- ")
    # collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_count(text):
    return len(text.split())


def is_relevant(text, topic):
    text_lower = text.lower()
    count = 0
    for kw in TOPIC_KEYWORDS[topic]:
        if kw in NEEDS_BOUNDARY:
            # use word boundary to avoid partial matches
            count += len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
        else:
            count += text_lower.count(kw)
    return count >= MIN_KEYWORD_COUNT[topic]


def fetch_window_query(query, topic, start_ts, end_ts):
    # collect short posts for one query within one time window
    rows        = []
    local_seen  = set()
    page        = 0
    retry_count = 0

    while len(rows) < MAX_RESULTS:
        params = {
            "query":          query,
            "tags":           "comment",
            "hitsPerPage":    100,
            "page":           page,
            "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
        }
        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search_by_date",
                params=params,
                timeout=15
            )

            # handle rate limit with limited retries to avoid infinite loop
            if resp.status_code == 429:
                retry_count += 1
                if retry_count >= MAX_RETRY:
                    print("  Max retries reached, skipping window.")
                    break
                print(f"  Rate limit hit (retry {retry_count}/{MAX_RETRY}), waiting...")
                time.sleep(RETRY_WAIT)
                continue

            retry_count = 0

            if resp.status_code != 200:
                break

            hits = resp.json().get("hits", [])
            if not hits:
                break

            for h in hits:
                post_id = h.get("objectID", "")

                # skip posts with missing id
                if not post_id:
                    continue

                if post_id in local_seen:
                    continue

                raw_text = h.get("comment_text", "") or ""
                text     = clean_text(raw_text)

                # skip posts that are empty after cleaning
                if len(text) < 5:
                    continue

                # skip posts that are too long
                if word_count(text) > MAX_WORDS:
                    continue

                # skip posts that are too short to carry meaning
                if word_count(text) < 5:
                    continue

                # skip posts not related to the topic
                if not is_relevant(text, topic):
                    continue

                local_seen.add(post_id)
                rows.append({
                    "post_id":    post_id,
                    "text_raw":   raw_text,
                    "text_clean": text,
                    "created_at": h.get("created_at", ""),
                    "url":        f"https://news.ycombinator.com/item?id={post_id}",
                    "topic":      topic,
                })

                if len(rows) >= MAX_RESULTS:
                    break

            page += 1
            if page > MAX_PAGES:
                break

        except Exception as e:
            print("Error:", e)
            break

        time.sleep(SLEEP_HIT)

    return rows


def fetch_topic(topic, queries):
    topic_rows = []
    seen_ids   = set()

    # loop over time windows first, then queries
    # this spreads results evenly across time instead of exhausting one keyword
    for label, start_ts, end_ts in TIME_WINDOWS:
        if len(topic_rows) >= TARGET:
            break

        window_added = 0
        for query in queries:
            if len(topic_rows) >= TARGET:
                break

            rows = fetch_window_query(query, topic, start_ts, end_ts)
            for r in rows:
                if r["post_id"] not in seen_ids:
                    seen_ids.add(r["post_id"])
                    topic_rows.append(r)
                    window_added += 1

            time.sleep(SLEEP_QUERY)

        if window_added > 0:
            print(f"  {label} | added: {window_added} | total: {len(topic_rows)}")

    # warn if target was not reached
    if len(topic_rows) < TARGET:
        print(f"  WARNING: only {len(topic_rows)}/{TARGET} posts collected for {topic}")

    return topic_rows


def main():
    all_rows = []

    for topic, queries in TOPICS.items():
        print("Topic:", topic)
        rows = fetch_topic(topic, queries)
        all_rows.extend(rows)
        print(f"  topic total: {len(rows)}")
        print()

    if not all_rows:
        print("No posts collected. Check your connection.")
        return

    df = pd.DataFrame(all_rows)
    df['word_count'] = df['text_clean'].str.split().str.len()

    print(df.groupby("topic").agg(
        posts     = ("post_id",    "count"),
        date_from = ("created_at", "min"),
        date_to   = ("created_at", "max"),
        avg_words = ("word_count", "mean"),
        max_words = ("word_count", "max"),
    ).round(1).to_string())

    os.makedirs(os.path.join(BASE_DIR, "data", "raw"), exist_ok=True)

    output_path = os.path.join(BASE_DIR, "data", "raw", "hackernews_posts_raw.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print()
    print("Saved:", output_path, "| rows:", len(df))

    # print 10 sample posts per topic
    for topic in df["topic"].unique():
        print()
        print("Topic:", topic)
        for _, row in df[df["topic"] == topic].head(10).iterrows():
            print(f"  [{row['word_count']} words] {row['text_clean']}")


if __name__ == "__main__":
    main()