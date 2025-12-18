"""
Embed emoji descriptions using polars-fastembed.
Run once to generate the parquet file.
"""

from pathlib import Path

import numpy as np
import polars as pl
from polars_fastembed import register_model
from umap import UMAP

MODEL_ID = "snowflake/snowflake-arctic-embed-xs"
HF_DATASET = "badrex/LLM-generated-emoji-descriptions"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading emoji dataset...")
    df = pl.read_parquet(f"hf://datasets/{HF_DATASET}@~parquet/default/train/*.parquet")
    print(f"Loaded {len(df)} emojis")

    print(f"\nRegistering model: {MODEL_ID}")
    register_model(MODEL_ID, providers=["CUDAExecutionProvider"])

    # Lowercase the short description for better embeddings
    df = df.with_columns(
        pl.col("short description").str.to_lowercase().alias("text_to_embed")
    )

    print("\nEmbedding...")
    df_emb = df.fastembed.embed(
        columns="text_to_embed",
        model_name=MODEL_ID,
        output_column="embedding",
    )

    print("\nRunning UMAP for 2D visualization...")
    embeddings = np.array(df_emb["embedding"].to_list(), dtype=np.float32)
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords_2d = umap.fit_transform(embeddings)

    df_final = df_emb.with_columns(
        pl.Series("x", coords_2d[:, 0].tolist()),
        pl.Series("y", coords_2d[:, 1].tolist()),
    ).drop("text_to_embed")

    parquet_path = OUTPUT_DIR / "emoji_embeddings.parquet"
    df_final.write_parquet(parquet_path)
    print(f"\nSaved: {parquet_path}")
    print(f"Size: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    run()
