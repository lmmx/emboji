# embed_emojis.py (updated - exports both parquet and JSON)
"""
Embed emoji descriptions using polars-fastembed and prepare for web search.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from polars_fastembed import register_model
from umap import UMAP

MODEL_ID = "BAAI/bge-small-en-v1.5"
HF_DATASET = "badrex/LLM-generated-emoji-descriptions"
OUTPUT_DIR = Path("output")


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) Load from HuggingFace
    print("Loading emoji dataset...")
    df = pl.read_parquet(f"hf://datasets/{HF_DATASET}@~parquet/default/train/*.parquet")
    print(f"Loaded {len(df)} emojis")

    # 2) Register model
    print(f"\nRegistering model: {MODEL_ID}")
    register_model(MODEL_ID)

    # 3) Create combined text
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("short description"),
                pl.col("tags").list.join(", "),
                pl.col("LLM description"),
            ],
            separator=" | ",
        ).alias("text_to_embed")
    )

    # 4) Embed
    print("\nEmbedding...")
    df_emb = df.fastembed.embed(
        columns="text_to_embed",
        model_name=MODEL_ID,
        output_column="embedding",
    )

    # 5) UMAP
    print("\nRunning UMAP...")
    emb_list = df_emb["embedding"].to_list()
    embeddings = np.array(emb_list, dtype=np.float32)

    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords_2d = umap.fit_transform(embeddings)

    # 6) Save parquet (for HuggingFace upload)
    df_final = df_emb.with_columns(
        pl.Series("x", coords_2d[:, 0].tolist()),
        pl.Series("y", coords_2d[:, 1].tolist()),
    ).drop("text_to_embed")

    parquet_path = OUTPUT_DIR / "emoji_embeddings.parquet"
    df_final.write_parquet(parquet_path)
    print(f"Saved parquet: {parquet_path}")

    # 7) Save JSON for simple web loading
    json_data = []
    for i, row in enumerate(df_final.iter_rows(named=True)):
        json_data.append(
            {
                "c": row["character"],
                "n": row["short description"],
                "d": row["LLM description"],
                "e": [round(float(x), 6) for x in emb_list[i]],  # embedding
                "x": round(float(coords_2d[i, 0]), 4),
                "y": round(float(coords_2d[i, 1]), 4),
            }
        )

    json_path = OUTPUT_DIR / "emojis.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f)
    print(f"Saved JSON: {json_path} ({json_path.stat().st_size / 1024 / 1024:.2f} MB)")
