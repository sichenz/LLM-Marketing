"""
preprocess.py — EDA and preprocessing for the scraped social media dataset.

Outputs:
    - data/processed.csv        : cleaned dataset ready for fine-tuning
    - data/stats.json           : engagement statistics per post type
    - data/top_posts.csv        : top-20 posts by likes (used as few-shot examples)
"""

import json
import pandas as pd

RAW_PATH       = "/scratch/sz4972/LLM-Marketing/data/data.csv"
PROCESSED_PATH = "/scratch/sz4972/LLM-Marketing/data/processed.csv"
STATS_PATH     = "/scratch/sz4972/LLM-Marketing/data/stats.json"
TOP_POSTS_PATH = "/scratch/sz4972/LLM-Marketing/data/top_posts.csv"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Numeric likes
    df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0).astype(int)

    # Drop rows missing essential fields
    df = df.dropna(subset=["Caption", "Post Title"])
    df = df[df["Caption"].str.strip() != "N/A"]
    df = df[df["Caption"].str.len() > 50]

    # Derived features
    df["caption_len"]   = df["Caption"].str.len()
    df["hashtag_count"] = df["Caption"].str.count("#")
    df["emoji_count"]   = df["Caption"].apply(
        lambda t: sum(1 for c in t if ord(c) > 0x1F300)
    )
    df["has_video"] = df["Video URL"].notna() & (df["Video URL"] != "N/A")
    df["has_image"] = df["Images"].notna()  & (df["Images"]    != "N/A")

    # Binary high-engagement label (top 25% by likes)
    threshold = df["Likes"].quantile(0.75)
    df["high_engagement"] = (df["Likes"] >= threshold).astype(int)

    print(f"Cleaned dataset: {len(df)} rows | engagement threshold: {threshold:.0f} likes")
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {
        "total_posts":        int(len(df)),
        "avg_likes":          float(df["Likes"].mean()),
        "median_likes":       float(df["Likes"].median()),
        "max_likes":          int(df["Likes"].max()),
        "avg_caption_len":    float(df["caption_len"].mean()),
        "avg_hashtags":       float(df["hashtag_count"].mean()),
        "pct_video":          float(df["has_video"].mean()),
        "pct_high_engagement":float(df["high_engagement"].mean()),
        "top_hashtags": (
            pd.Series(
                "#".join(df["Caption"].str.cat(sep=" ")).split("#")
            )
            .str.strip()
            .value_counts()
            .head(10)
            .to_dict()
        ),
    }
    return stats


if __name__ == "__main__":
    df    = load_and_clean(RAW_PATH)
    stats = compute_stats(df)

    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    print(f"Processed data saved to: {PROCESSED_PATH}")

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats saved to: {STATS_PATH}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    top = df.nlargest(20, "Likes")[["Post Title", "Likes", "Caption", "hashtag_count"]]
    top.to_csv(TOP_POSTS_PATH, index=False, encoding="utf-8-sig")
    print(f"Top posts saved to: {TOP_POSTS_PATH}")
