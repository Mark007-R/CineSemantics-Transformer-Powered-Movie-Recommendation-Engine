from email.policy import default
from importlib.metadata import metadata
from altair import value
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
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
        def safe_float(value, default=0.0):
            try:
                return float(value) if value not in [None, '', 'N/A', 'nan'] else default
            except (ValueError, TypeError):
                logger.warning(f"Could not convert '{value}' to float, using default {default}")
                return default
        def safe_int(value, default=0):
            try:
                return int(float(value)) if value not in [None, '', 'N/A', 'nan'] else default
            except (ValueError, TypeError):
                logger.warning(f"Could not convert '{value}' to int, using default {default}")
                return default
        def safe_str(value, default=''):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return default
            return str(value).strip()
        metadata = []
        texts = []
        for idx, row in df.iterrows():
            text = f"{row.get('Title', '')}. {row.get(text_column, '')}".strip()
            texts.append(text)
            metadata.append({
                'index': int(idx),
                'title': safe_str(row.get('Title', ''), ''),
                'overview': safe_str(row.get('Overview', ''), ''),
                'release_date': safe_str(row.get('Release_Date', ''), ''),
                'genre': safe_str(row.get('Genre', ''), ''),
                'popularity': safe_float(row.get('Popularity', 0), 0.0),
                'vote_average': safe_float(row.get('Vote_Average', 0), 0.0),
                'vote_count': safe_int(row.get('Vote_Count', 0), 0),
                'original_language': safe_str(row.get('Original_Language', ''), ''),
                'poster_url': safe_str(row.get('Poster_Url', ''), '')
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

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="overview", dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name="release_date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="popularity", dtype=DataType.DOUBLE),
            FieldSchema(name="vote_average", dtype=DataType.DOUBLE),
            FieldSchema(name="vote_count", dtype=DataType.INT64),
            FieldSchema(name="original_language", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="poster_url", dtype=DataType.VARCHAR, max_length=1000),
        ]
        schema = CollectionSchema(fields=fields, description="Movie similarity search collection")
        if utility.has_collection(self.collection_name):
            logger.warning(f"Collection {self.collection_name} already exists. Dropping it.")
            utility.drop_collection(self.collection_name)
        self.collection = Collection(name=self.collection_name, schema=schema)
        logger.info(f"Created collection: {self.collection_name}")

    def save(self, embeddings: np.ndarray, metadata: List[Dict]):
        try:
            self.connect()
        except ConnectionError as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
        try:
            self._create_collection()
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise RuntimeError(f"Collection creation failed: {e}")
        try:
            ids = [m['index'] for m in metadata]
            titles = [m['title'] for m in metadata]
            overviews = [m['overview'] for m in metadata]
            release_dates = [m['release_date'] for m in metadata]
            genres = [m['genre'] for m in metadata]
            popularities = [m['popularity'] for m in metadata]
            vote_averages = [m['vote_average'] for m in metadata]
            vote_counts = [m['vote_count'] for m in metadata]
            original_languages = [m['original_language'] for m in metadata]
            poster_urls = [m['poster_url'] for m in metadata]
            entities = [ids, embeddings.tolist(), titles, overviews, release_dates, genres, 
                popularities, vote_averages, vote_counts, original_languages, poster_urls
            ]
            self.collection.insert(entities)
            logger.info(f"Inserted {len(ids)} entities into collection")
        except Exception as e:
            logger.error(f"Failed to insert entities: {e}")
            try:
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped collection {self.collection_name} due to insertion failure")
            except:
                pass
            raise RuntimeError(f"Data insertion failed: {e}")
        try:
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info("Created index on embedding field")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            try:
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped collection {self.collection_name} due to indexing failure")
            except:
                pass
            raise RuntimeError(f"Index creation failed: {e}")
        try:
            self.collection.load()
            logger.info("Collection loaded into memory")
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise RuntimeError(f"Collection loading failed: {e}")
        return {
            'num_vectors': len(embeddings),
            'dimension': embeddings.shape[1],
            'collection_name': self.collection_name
        }

    def load(self):
        self.connect()
        if not utility.has_collection(self.collection_name):
            logger.error(f"Collection {self.collection_name} does not exist")
            raise ValueError(f"Collection {self.collection_name} not found")
        self.collection = Collection(self.collection_name)
        self.collection.load()
        num_entities = self.collection.num_entities
        logger.info(f"Loaded collection {self.collection_name} with {num_entities} entities")
        return {
            'num_vectors': num_entities,
            'collection_name': self.collection_name
        }

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[List[int], List[float], List[Dict]]:
        if self.collection is None:
            raise ValueError("Collection not loaded. Call load() first.")
        search_params = { "metric_type": "IP", "params": {"nprobe": 10} }
        try:
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["title", "overview", "release_date", "genre", 
                    "popularity", "vote_average", "vote_count", "poster_url"]
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Failed to search collection: {e}")
        indices = []
        distances = []
        metadata_list = []
        if not results or len(results) == 0:
            logger.warning("No results found for the query")
            return indices, distances, metadata_list
        for hits in results:
            if not hits:
                logger.warning("Empty hits in search results")
                continue
            for hit in hits:
                indices.append(hit.id)
                distances.append(hit.distance)
                metadata_list.append(hit.entity.to_dict())
        logger.info(f"Search returned {len(indices)} results")
        return indices, distances, metadata_list
    
    def get_all_entities(self) -> List[Dict]:
        if self.collection is None:
            raise ValueError("Collection not loaded. Call load() first.")
        all_entities = []
        batch_size = 16384
        offset = 0
        try:
            total_entities = self.collection.num_entities
            logger.info(f"Fetching all {total_entities} entities from collection")
            while offset < total_entities:
                query_result = self.collection.query(
                    expr="id >= 0",
                    output_fields=["title", "overview", "genre", "poster_url"],
                    limit=batch_size,
                    offset=offset
                )
                if not query_result:
                    break
                all_entities.extend(query_result)
                offset += batch_size
                logger.info(f"Fetched {len(all_entities)}/{total_entities} entities")
            logger.info(f"Successfully fetched all {len(all_entities)} entities")
            return all_entities
        except Exception as e:
            logger.error(f"Failed to fetch entities: {e}")
            raise RuntimeError(f"Failed to query all entities: {e}")
    
class TextSimilaritySearch:
    def __init__(self, collection_name='movie_collection', host='localhost', port='19530',
        model_name='sentence-transformers/all-MiniLM-L6-v2', dimension=384):
        self.collection_name = collection_name
        self.model_name = model_name
        self.db = MilvusDB(collection_name, host, port, dimension)
        self.embedder = TextEmbedder(model_name)
        try:
            self.db.load()
            logger.info("TextSimilaritySearch initialized and collection loaded")
        except ValueError:
            logger.warning("No existing collection found. Please build the database first.")

    
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
        indices, distances, metadata_list = self.db.search(query_embedding, top_k)
        results = []
        for idx, dist, metadata in zip(indices, distances, metadata_list):
            results.append({
                'index': int(idx),
                'title': metadata.get('title', ''),
                'overview': metadata.get('overview', ''),
                'release_date': metadata.get('release_date', ''),
                'genre': metadata.get('genre', ''),
                'popularity': metadata.get('popularity', 0),
                'vote_average': metadata.get('vote_average', 0),
                'vote_count': metadata.get('vote_count', 0),
                'poster_url': metadata.get('poster_url', ''),
                'similarity': float(dist),
                'similarity_percent': f"{float(dist) * 100:.2f}%"
            })
        logger.info(f"Found {len(results)} similar movies")
        return results
    
    def get_all_movies(self) -> List[Dict[str, str]]:
        entities = self.db.get_all_entities()
        return [
            {
                "title": e.get("title", ""),
                "overview": e.get("overview", ""),
                "genre": e.get("genre", ""),
                "poster_url": e.get("poster_url", "")
            }
            for e in entities
        ]