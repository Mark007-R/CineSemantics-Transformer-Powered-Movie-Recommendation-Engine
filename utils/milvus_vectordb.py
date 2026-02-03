import logging
from text_embedder import embed_text
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def milvus_connect(host='localhost', port='19530'):
    try:
        logger.info(f"Connecting to Milvus at {host}:{port}...")
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        logger.info("Connected to Milvus successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return None


def milvus_disconnect():
    try:
        logger.info("Disconnecting from Milvus...")
        connections.disconnect(alias="default")
        logger.info("Disconnected from Milvus")
    except Exception as e:
        logger.error(f"Failed to disconnect from Milvus: {e}")


def create_collection(dimension=384):
    try:
        if utility.has_collection('movie_collection'):
            logger.info("Collection already exists, loading it...")
            collection = Collection(name='movie_collection')
            collection.load()
            return collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
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
        collection = Collection(name='movie_collection', schema=schema)
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        logger.info("Collection created successfully")
        return collection
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return None


def delete_collection(collection_name='movie_collection'):
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        else:
            logger.warning(f"Collection '{collection_name}' does not exist")
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")


def save_csv_embeddings(collection, embeddings, metadata):
    try:
        if collection is None:
            logger.error("Collection is not loaded")
            return False
        if embeddings is None or metadata is None:
            logger.error("Embeddings or metadata is None")
            return False
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
        entities = [
            ids, embeddings.tolist(), titles, overviews, release_dates,
            genres, popularities, vote_averages, vote_counts,
            original_languages, poster_urls
        ]

        collection.insert(entities)
        collection.flush()
        logger.info(f"Saved {len(ids)} embeddings to collection")
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV embeddings: {e}")
        return False


def save_text_embedding(collection, id: int, text: str, embedding, metadata: dict):
    try:
        if collection is None:
            logger.error("Collection is not loaded")
            return False
        if embedding is None:
            logger.error("Embedding is None")
            return False
        entities = [
            [id],
            embedding.tolist(),
            [metadata.get('title', '')],
            [metadata.get('overview', text)],
            [metadata.get('release_date', '')],
            [metadata.get('genre', '')],
            [metadata.get('popularity', 0.0)],
            [metadata.get('vote_average', 0.0)],
            [metadata.get('vote_count', 0)],
            [metadata.get('original_language', '')],
            [metadata.get('poster_url', '')]
        ]
        collection.insert(entities)
        collection.flush()
        logger.info(f"Saved single text embedding with id {id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save text embedding: {e}")
        return False


def search(collection, model, query_text: str, top_k: int = 10):
    try:
        if collection is None:
            logger.error("Collection is not loaded")
            return []
        if model is None:
            logger.error("Model is not loaded")
            return []
        query_embedding = embed_text(model, query_text)
        if query_embedding is None:
            logger.error("Failed to get query embedding")
            return []
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            params=search_params,
            limit=top_k,
            output_fields=["title", "overview", "release_date", "genre",
                           "popularity", "vote_average", "vote_count",
                           "poster_url", "original_language"]
        )
        movies = []
        for hits in results:
            for hit in hits:
                movies.append({
                    'id': hit.id,
                    'title': hit.entity.get('title', ''),
                    'overview': hit.entity.get('overview', ''),
                    'release_date': hit.entity.get('release_date', ''),
                    'genre': hit.entity.get('genre', ''),
                    'popularity': hit.entity.get('popularity', 0),
                    'vote_average': hit.entity.get('vote_average', 0),
                    'vote_count': hit.entity.get('vote_count', 0),
                    'poster_url': hit.entity.get('poster_url', ''),
                    'original_language': hit.entity.get('original_language', ''),
                    'similarity': float(hit.distance),
                    'similarity_percent': f"{float(hit.distance) * 100:.2f}%"
                })
        logger.info(f"Found {len(movies)} similar movies")
        return movies
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []