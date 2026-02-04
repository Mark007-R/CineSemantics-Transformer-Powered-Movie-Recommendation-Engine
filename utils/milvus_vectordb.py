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
        logger.info(f"Using Milvus Lite (embedded mode)...")
        from milvus import default_server
        default_server.start()
        connections.connect(alias="default", host="127.0.0.1", port=default_server.listen_port)
        logger.info("Connected to Milvus Lite successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start Milvus Lite: {e}")
        return False


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
            FieldSchema(name="original_language", dtype=DataType.VARCHAR, max_length=200),
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
            [metadata.get('title', '')[:1000]],
            [metadata.get('overview', text)[:20000]],
            [metadata.get('release_date', '')[:50]],
            [metadata.get('genre', '')[:200]],
            [metadata.get('popularity', 0.0)],
            [metadata.get('vote_average', 0.0)],
            [metadata.get('vote_count', 0)],
            [metadata.get('original_language', '')[:200]],
            [metadata.get('poster_url', '')[:1000]]
        ]
        collection.insert(entities)
        collection.flush()
        logger.info(f"Saved single text embedding with id {id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save text embedding: {e}")
        return False


def search(collection, model, query_text: str, top_k: int = 10, min_rating: float = None,
        max_rating: float = None, min_popularity: float = None, genre_filter: str = None, 
        year_filter: int = None, min_year: int = None, max_year: int = None):
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
        filter_parts = []
        if min_rating is not None:
            filter_parts.append(f"vote_average >= {min_rating}")
        if max_rating is not None:
            filter_parts.append(f"vote_average <= {max_rating}")
        if min_popularity is not None:
            filter_parts.append(f"popularity >= {min_popularity}")
        if year_filter is not None:
            if min_year is not None or max_year is not None:
                logger.warning("Both year_filter and min_year/max_year provided, using year_filter only.")
            try:
                year_int = int(year_filter)
                if year_int < 0:
                    raise ValueError("year_filter must be non-negative")
                filter_parts.append(f'release_date like "{year_int}-%"')
            except (ValueError, TypeError):
                logger.warning(f"Ignoring invalid year_filter value: {year_filter!r}")
        else:
            if min_year is not None:
                try:
                    min_year_int = int(min_year)
                    if min_year_int < 0:
                        raise ValueError("min_year must be non-negative")
                    filter_parts.append(f'release_date >= "{min_year_int}-01-01"')
                except (ValueError, TypeError):
                    logger.warning(f"Ignoring invalid min_year value: {min_year!r}")
            if max_year is not None:
                try:
                    max_year_int = int(max_year)
                    if max_year_int < 0:
                        raise ValueError("max_year must be non-negative")
                    filter_parts.append(f'release_date <= "{max_year_int}-12-31"')
                except (ValueError, TypeError):
                    logger.warning(f"Ignoring invalid max_year value: {max_year!r}")
        filter_expr = " and ".join(filter_parts) if filter_parts else None
        if filter_expr:
            logger.info(f"Applying database filter: {filter_expr}")
        if genre_filter:
            logger.info(f"Will apply genre post-filter in Python: '{genre_filter}'")
        fetch_limit = top_k * 3 if genre_filter else top_k  
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=fetch_limit,
            expr=filter_expr,
            output_fields=["title", "overview", "release_date", "genre",
                           "popularity", "vote_average", "vote_count",
                           "poster_url", "original_language"]
        )
        movies = []
        for hits in results:
            for hit in hits:
                movie_data = {
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
                }
                if genre_filter:
                    if genre_filter.lower() in movie_data['genre'].lower():
                        movies.append(movie_data)
                else:
                    movies.append(movie_data)
                if len(movies) >= top_k:
                    break
            if len(movies) >= top_k:
                break
        movies = movies[:top_k]
        logger.info(f"Found {len(movies)} similar movies")
        return movies
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []