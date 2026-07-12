import requests
import pandas as pd
import time

API_KEY = "YOUR_API_KEY"

TOPICS = {
    "Artificial Intelligence": [
        "artificial intelligence", "ChatGPT", "machine learning", "OpenAI"
    ],
    "World Cup 2026": [
        "world cup 2026", "FIFA 2026", "football world cup"
    ],
    "Climate Change": [
        "climate change", "global warming", "carbon emissions"
    ],
    "Cryptocurrency": [
        "bitcoin", "cryptocurrency", "ethereum", "crypto market"
    ],
}


def fetch_guardian(query):
    rows = []
    for page in range(1, 6):
        params = {
            "q":           query,
            "api-key":     API_KEY,
            "page-size":   50,
            "page":        page,
            "from-date":   "2024-07-01",
            "to-date":     "2026-07-12",
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
                          fields.get("bodyText",  "")).strip()
                if len(text) < 50:
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
            rows = fetch_guardian(query)
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

    df = pd.DataFrame(all_rows)

    print(df.groupby("topic").agg(
        articles  = ("post_id",    "count"),
        date_from = ("created_at", "min"),
        date_to   = ("created_at", "max"),
    ).to_string())

    df.to_csv("data/raw/guardian_posts_raw.csv", index=False, encoding="utf-8")
    print("Saved: data/raw/guardian_posts_raw.csv, rows:", len(df))

    for topic in df["topic"].unique():
        print("\nTopic:", topic)
        for _, row in df[df["topic"] == topic].head(10).iterrows():
            print(" ", row["created_at"][:10], "|", row["text"][:120])


if __name__ == "__main__":
    main()
