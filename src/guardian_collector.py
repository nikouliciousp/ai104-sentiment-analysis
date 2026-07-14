import os
import requests
import pandas as pd
import time

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

API_KEY = "bfa1699d-5d22-4538-801b-34b2c8c9a0fd"

TOPICS = {
    "Artificial Intelligence": [
        "artificial intelligence",
        "ChatGPT",
        "machine learning",
        "OpenAI",
        "large language model",
    ],
    "World Cup 2026": [
        "world cup 2026",
        "FIFA 2026",
        "football world cup 2026",
        "world cup football",
    ],
    "Climate Change": [
        "climate change",
        "global warming",
        "carbon emissions",
        "climate crisis",
        "fossil fuels",
    ],
    "Cryptocurrency": [
        "bitcoin price",
        "cryptocurrency market",
        "ethereum blockchain",
        "crypto trading",
        "digital currency",
        "crypto regulation",
        "bitcoin investment",
        "digital assets",
        "crypto news",
    ],
}

# keywords per topic for relevance filtering
TOPIC_KEYWORDS = {
    "Artificial Intelligence": [
        "ai", "artificial intelligence", "chatgpt",
        "openai", "llm", "machine learning"
    ],
    "World Cup 2026": [
        "world cup", "fifa", "2026", "football", "soccer"
    ],
    "Climate Change": [
        "climate", "warming", "carbon", "emission", "fossil"
    ],
    "Cryptocurrency": [
        "bitcoin", "crypto", "ethereum", "blockchain", "btc",
        "defi", "nft", "coinbase", "binance", "digital assets",
        "web3", "token"
    ],
}

# minimum keyword occurrences — lower for Cryptocurrency
MIN_KEYWORD_COUNT = {
    "Artificial Intelligence": 3,
    "World Cup 2026":          3,
    "Climate Change":          3,
    "Cryptocurrency":          2,
}


def is_relevant(text, topic):
    # count how many times topic keywords appear in the article
    text_lower = text.lower()
    keywords   = TOPIC_KEYWORDS[topic]
    count      = sum(text_lower.count(kw) for kw in keywords)
    return count >= MIN_KEYWORD_COUNT[topic]


def fetch_guardian(query, topic):
    rows = []
    for page in range(1, 6):
        params = {
            "q":           query,
            "api-key":     API_KEY,
            "page-size":   50,
            "page":        page,
            "from-date":   "2024-07-01",
            "to-date":     "2026-07-14",
            "show-fields": "bodyText,headline",
            "lang":        "en",
        }
        try:
            resp = requests.get(
                "https://content.guardianapis.com/search",
                params=params,
                timeout=15
            )
            if resp.status_code != 200:
                break

            data    = resp.json()["response"]
            results = data.get("results", [])
            if not results:
                break

            for r in results:
                fields = r.get("fields", {})
                text   = (fields.get("headline", "") + " " +
                          fields.get("bodyText", "")).strip()

                if len(text) < 50:
                    continue

                # skip articles not relevant to the topic
                if not is_relevant(text, topic):
                    continue

                rows.append({
                    "post_id":    r.get("id", ""),
                    "text":       text[:2000],
                    "created_at": r.get("webPublicationDate", ""),
                    "url":        r.get("webUrl", ""),
                })

            if page >= data.get("pages", 1):
                break

        except Exception as e:
            print("Error:", e)
            break

        time.sleep(0.5)

    return rows


def main():
    all_rows = []

    for topic, queries in TOPICS.items():
        print("Topic:", topic)
        topic_rows = []
        seen_ids   = set()

        for query in queries:
            if len(topic_rows) >= 300:
                break
            rows = fetch_guardian(query, topic)
            for r in rows:
                if r["post_id"] not in seen_ids:
                    seen_ids.add(r["post_id"])
                    r["topic"] = topic
                    topic_rows.append(r)
            print("  query:", query, "| found:", len(rows),
                  "| total:", len(topic_rows))
            time.sleep(1)

        all_rows.extend(topic_rows)
        print("  topic total:", len(topic_rows))
        print()

    df = pd.DataFrame(all_rows)

    print(df.groupby("topic").agg(
        articles  = ("post_id",    "count"),
        date_from = ("created_at", "min"),
        date_to   = ("created_at", "max"),
    ).to_string())

    # create output folder if it does not exist
    os.makedirs(os.path.join(BASE_DIR, "data", "raw"), exist_ok=True)

    output_path = os.path.join(BASE_DIR, "data", "raw", "guardian_posts_raw.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print()
    print("Saved:", output_path, "| rows:", len(df))

    # print first 10 articles per topic
    for topic in df["topic"].unique():
        print()
        print("Topic:", topic)
        for _, row in df[df["topic"] == topic].head(10).iterrows():
            print(" ", row["created_at"][:10], "|", row["text"][:120])


if __name__ == "__main__":
    main()