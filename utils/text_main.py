import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import faiss
import warnings
import logging
from pymilvus import (connections, utility, FieldSchema, CollectionSchema, DataType, Collection, )

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextEmbedder:
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        logger.info(f"TextEmbedder initialized with model: {model_name}, device: {self.device}")
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
    
    def embed_csv(self, csv_path: str, text_column: str = 'Overview', batch_size: int = 32) -> Tuple[np.ndarray, List[Dict]]:
        self._load_model()
        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        logger.info(f"Loading CSV from: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            raise ValueError(f"Failed to load CSV: {e}")
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in CSV. Available columns: {df.columns.tolist()}")
            raise ValueError(f"Column '{text_column}' not found in CSV")
        df = df.fillna('')
        logger.info(f"Found {len(df)} rows in CSV")
        metadata = []
        texts = []
        for idx, row in df.iterrows():
            text = f"{row.get('Title', '')}. {row.get(text_column, '')}".strip()
            texts.append(text)
            metadata.append({
                'index': int(idx),
                'title': row.get('Title', ''),
                'overview': row.get('Overview', ''),
                'release_date': row.get('Release_Date', ''),
                'genre': row.get('Genre', ''),
                'popularity': row.get('Popularity', 0),
                'vote_average': row.get('Vote_Average', 0),
                'vote_count': row.get('Vote_Count', 0),
                'original_language': row.get('Original_Language', ''),
                'poster_url': row.get('Poster_Url', '')
            })
        logger.info("Extracting embeddings...")
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.model.encode(
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
    
    def embed_text(self, text: str) -> np.ndarray:
        self._load_model()
        if not text or not text.strip():
            logger.error("Empty text provided")
            raise ValueError("Text cannot be empty")
        logger.info(f"Processing query text: {text[:100]}...")
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.reshape(1, -1)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise ValueError(f"Failed to embed text: {e}")


class MilvusDB:
    def __init__(self, collection_name='movie_collection', host='localhost', port='19530', dimension=384):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dimension = dimension
        self.collection = None
        logger.info(f"MilvusDB initialized for collection: {collection_name}")

    def connect(self):
        try:
            connections.connect(alias="default", host=self.host, port=self.port )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise ConnectionError(f"Milvus connection failed: {e}")
        
    def save(self, embeddings: np.ndarray, metadata: List[Dict], index_name: str = 'text_index'):
        self.db_path.mkdir(parents=True, exist_ok=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        index_path = self.db_path / f'{index_name}.index'
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to: {index_path}")
        metadata_path = self.db_path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {metadata_path}")
        info = {
            'num_vectors': len(embeddings),
            'dimension': dimension,
            'index_path': str(index_path),
            'metadata_path': str(metadata_path)
        }
        info_path = self.db_path / 'db_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved {len(embeddings)} vectors to {self.db_path}")
        return info
    
    def load(self):
        info_path = self.db_path / 'db_info.json'
        if not info_path.exists():
            logger.error(f"Database not found at {self.db_path}")
            raise ValueError(f"Database not found at {self.db_path}")
        with open(info_path, 'r') as f:
            info = json.load(f)
        self.index = faiss.read_index(info['index_path'])
        with open(info['metadata_path'], 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded {len(self.metadata)} vectors from {self.db_path}")
        return info


class TextSimilaritySearch:    
    def __init__(self, db_path='../vector_db_text', model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.model_name = model_name
        self.db = VectorDB(db_path)
        self.embedder = TextEmbedder(model_name)
        try:
            self.db.load()
            logger.info("TextSimilaritySearch initialized and database loaded")
        except ValueError:
            logger.warning("No existing database found. Please build the database first.")
    
    def build_database(self, csv_path: str, text_column: str = 'Overview', batch_size: int = 32):
        logger.info(f"Building database from {csv_path}")
        embeddings, metadata = self.embedder.embed_csv(csv_path, text_column, batch_size)
        info = self.db.save(embeddings, metadata)
        self.db.load()
        logger.info("Database built successfully")
        return info
    
    def find_similar(self, query_text: str, top_k: int = 10) -> List[Dict]:
        logger.info(f"Searching for top {top_k} similar movies to: '{query_text[:100]}...'")
        query_embedding = self.embedder.embed_text(query_text)
        distances, indices = self.db.index.search(query_embedding.astype('float32'), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            metadata = self.db.metadata[idx]
            results.append({
                'index': int(idx),
                'title': metadata['title'],
                'overview': metadata['overview'],
                'release_date': metadata['release_date'],
                'genre': metadata['genre'],
                'popularity': metadata['popularity'],
                'vote_average': metadata['vote_average'],
                'vote_count': metadata['vote_count'],
                'poster_url': metadata['poster_url'],
                'similarity': float(dist),
                'similarity_percent': f"{float(dist) * 100:.2f}%"
            })
        logger.info(f"Found {len(results)} similar movies")
        return results
    
    def get_all_movies(self) -> List[Dict[str, str]]:
        return [{"title": m["title"], "overview": m["overview"], "genre": m["genre"], "poster_url": m["poster_url"]} for m in self.db.metadata]
