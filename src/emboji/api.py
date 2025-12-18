"""
FastAPI server for emoji semantic search.
Uses polars-fastembed retrieve() for search.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from polars_fastembed import register_model

MODEL_ID = "snowflake/snowflake-arctic-embed-xs"
BASE_DIR = Path(__file__).parents[2]
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"

# Global state
df: pl.DataFrame


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df
    print(f"Registering model: {MODEL_ID}")
    register_model(MODEL_ID, providers=["CUDAExecutionProvider"])

    parquet_path = OUTPUT_DIR / "emoji_embeddings.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No embeddings found at {parquet_path}. Run 'emboji-embed' first."
        )

    print(f"Loading embeddings from {parquet_path}")
    df = pl.read_parquet(parquet_path)
    print(f"Loaded {len(df)} emojis")
    yield


app = FastAPI(title="Emboji", description="Emoji semantic search", lifespan=lifespan)


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), k: int = Query(60, ge=1, le=200)):
    """
    Semantic search over emojis.
    Returns top k results with similarity scores.
    """
    result = df.fastembed.retrieve(
        query=q,
        model_name=MODEL_ID,
        embedding_column="embedding",
        k=k,
    )
    return [
        {
            "c": row["character"],
            "n": row["short description"],
            "d": row["LLM description"],
            "s": round(row["similarity"], 4),
            "x": round(row["x"], 4),
            "y": round(row["y"], 4),
        }
        for row in result.iter_rows(named=True)
    ]


@app.get("/api/emojis")
def get_all_emojis():
    """
    Return all emojis with UMAP coordinates for the 2D scatter plot.
    No embeddings sent to client.
    """
    return [
        {
            "c": row["character"],
            "n": row["short description"],
            "x": round(row["x"], 4),
            "y": round(row["y"], 4),
        }
        for row in df.iter_rows(named=True)
    ]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run():
    uvicorn.run(
        "emboji.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
