import logging
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def load_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading model on {device}...")
    try:
        model = SentenceTransformer(model_name, device=device)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def embed_csv(model, csv_path: str, text_column: str = 'Overview', batch_size: int = 32):
    if model is None:
        logger.error("Model is not loaded")
        return None, None
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"CSV file not found at {csv_path}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return None, None
    df = df.fillna('')
    logger.info(f"CSV columns detected: {df.columns.tolist()}")
    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
        return None, None
    texts = []
    metadata = []
    for idx, row in df.iterrows():
        text = f"{row.get('Title', '')}. {row.get(text_column, '')}".strip()
        texts.append(text)
        metadata.append({
            'index': int(idx),
            'title': str(row.get('Title', '')).strip(),
            'overview': str(row.get('Overview', '')).strip(),
            'release_date': str(row.get('Release_Date', '')).strip(),
            'genre': str(row.get('Genre', '')).strip(),
            'popularity': float(row.get('Popularity', 0) or 0),
            'vote_average': float(row.get('Vote_Average', 0) or 0),
            'vote_count': int(row.get('Vote_Count', 0) or 0),
            'original_language': str(row.get('Original_Language', '')).strip(),
            'poster_url': str(row.get('Poster_Url', '')).strip()
        })
    logger.info(f"Found {len(texts)} rows, extracting embeddings...")
    try:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]
            embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(embeddings)
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        return embeddings, metadata
    except Exception as e:
        logger.error(f"Failed to extract embeddings: {e}")
        return None, None

def embed_text(model, text: str):
    if model is None:
        logger.error("Model is not loaded")
        return None
    if not text or not text.strip():
        logger.error("Text cannot be empty")
        return None
    try:
        logger.info(f"Processing query text: {text[:100]}...")
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.reshape(1, -1)
    except Exception as e:
        logger.error(f"Failed to embed text: {e}")
        return None