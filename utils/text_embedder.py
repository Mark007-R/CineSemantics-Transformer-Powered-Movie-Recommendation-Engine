import logging
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def _query_prefix():
    """Instruction prefix for the configured encoder.

    The Day-2 champion e5-base-v2 requires an instruction prefix; for the
    symmetric 'more like this' task both catalog text and queries are prefixed
    with 'query: '. Returns '' for models that need no prefix (MiniLM/MPNet/BGE),
    so this stays a no-op if TEXT_MODEL_NAME is reverted.
    """
    if 'e5' in config.TEXT_MODEL_NAME.lower():
        return getattr(config, 'TEXT_QUERY_PREFIX', 'query: ')
    return ''


def _safe_float(value, default=None):
    if default is None:
        default = config.DEFAULT_FLOAT_VALUE
    if pd.isna(value) or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to float, using default {default}")
        return default


def _safe_int(value, default=None):
    if default is None:
        default = config.DEFAULT_INT_VALUE
    if pd.isna(value) or value == '':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to int, using default {default}")
        return default


def load_model(model_name=None):
    if model_name is None:
        model_name = config.TEXT_MODEL_NAME
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading model on {device}...")
    try:
        model = SentenceTransformer(model_name, device=device)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def embed_csv(model, csv_path: str, text_column: str = None, batch_size: int = None):
    if text_column is None:
        text_column = config.DEFAULT_TEXT_COLUMN
    if batch_size is None:
        batch_size = config.DEFAULT_BATCH_SIZE
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
        title = row.get('Title', '')
        overview = row.get(text_column, '')
        genre = row.get('Genre', '')
        release_date = row.get('Release_Date', '')
        year = ''
        if release_date and len(release_date) >= 4:
            year = release_date[:4]
        text_parts = [title]
        if genre:
            text_parts.append(f"Genre: {genre}")
        if year:
            text_parts.append(f"Released: {year}")
        if overview:
            text_parts.append(overview)
        text = ". ".join(text_parts).strip()
        texts.append(_query_prefix() + text)
        metadata.append({
            'index': int(idx),
            'title': str(row.get('Title', ''))[:config.TEXT_TITLE_MAX_LENGTH].strip(),
            'overview': str(row.get('Overview', ''))[:config.TEXT_OVERVIEW_MAX_LENGTH].strip(),
            'release_date': str(row.get('Release_Date', ''))[:config.TEXT_RELEASE_DATE_MAX_LENGTH].strip(),
            'genre': str(row.get('Genre', ''))[:config.TEXT_GENRE_MAX_LENGTH].strip(),
            'popularity': _safe_float(row.get('Popularity', config.DEFAULT_FLOAT_VALUE)),
            'vote_average': _safe_float(row.get('Vote_Average', config.DEFAULT_FLOAT_VALUE)),
            'vote_count': _safe_int(row.get('Vote_Count', config.DEFAULT_INT_VALUE)),
            'original_language': str(row.get('Original_Language', ''))[:config.TEXT_ORIGINAL_LANGUAGE_MAX_LENGTH].strip(),
            'poster_url': str(row.get('Poster_Url', ''))[:config.TEXT_POSTER_URL_MAX_LENGTH].strip()
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
            _query_prefix() + text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.reshape(1, -1)
    except Exception as e:
        logger.error(f"Failed to embed text: {e}")
        return None