import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    try:
        model = SentenceTransformer(model_name, device=device)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def embed_csv(model, csv_path: str, text_column: str = 'Overview', batch_size: int = 32):
    if model is None:
        print("Error: Model is not loaded")
        return None, None
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None
    except Exception as e:
        print(f"Error: Failed to read CSV file: {e}")
        return None, None
    df = df.fillna('')
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
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
    print(f"Found {len(texts)} rows, extracting embeddings...")
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
        print(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        return embeddings, metadata
    except Exception as e:
        print(f"Error: Failed to extract embeddings: {e}")
        return None, None

def embed_text(model, text: str):
    if model is None:
        print("Error: Model is not loaded")
        return None
    if not text or not text.strip():
        print("Error: Text cannot be empty")
        return None
    try:
        print(f"Processing query text: {text[:100]}...")
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.reshape(1, -1)
    except Exception as e:
        print(f"Error: Failed to embed text: {e}")
        return None