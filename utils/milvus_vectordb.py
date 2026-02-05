import logging
import re
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


def create_text_collection(dimension=384):
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


def delete_text_collection(collection_name='movie_collection'):
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        else:
            logger.warning(f"Collection '{collection_name}' does not exist")
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")

def create_image_collection(collection_name='movie_posters', dimension=512):
    try:
        if utility.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' already exists, loading it...")
            collection = Collection(name=collection_name)
            collection.load()
            return collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1000),
        ]
        schema = CollectionSchema(fields=fields, description="Movie poster image similarity search")
        collection = Collection(name=collection_name, schema=schema)
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        logger.info(f"Collection '{collection_name}' created successfully")
        return collection
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return None


def delete_image_collection(collection_name='movie_posters'):
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        else:
            logger.warning(f"Collection '{collection_name}' does not exist")
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")


def save_image_embeddings(collection, embeddings, metadata):
    try:
        if collection is None:
            logger.error("Collection is not loaded")
            return False
        if embeddings is None or metadata is None:
            logger.error("Embeddings or metadata is None")
            return False
        if len(embeddings) != len(metadata):
            logger.error(f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch")
            return False
        ids = list(range(len(metadata)))
        titles = [m['title'][:500] for m in metadata]
        filenames = [m['filename'][:500] for m in metadata]
        image_paths = [m['image_path'][:1000] for m in metadata]
        entities = [
            ids,
            embeddings.tolist(),
            titles,
            filenames,
            image_paths
        ]
        collection.insert(entities)
        collection.flush()
        logger.info(f"Saved {len(ids)} image embeddings to collection")
        return True
    except Exception as e:
        logger.error(f"Failed to save image embeddings: {e}")
        return False


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


def _validate_year(year_value, param_name):
    try:
        year_int = int(year_value)
        if year_int < 1800 or year_int > 2100:
            logger.warning(f"{param_name}={year_int} is outside reasonable range [1800, 2100], ignoring")
            return None
        return year_int
    except (ValueError, TypeError):
        logger.warning(f"Invalid {param_name} value: {year_value!r}, ignoring")
        return None


def _validate_rating(rating_value, param_name):
    try:
        rating_float = float(rating_value)
        if rating_float < 0 or rating_float > 10:
            logger.warning(f"{param_name}={rating_float} is outside valid range [0, 10], ignoring")
            return None
        return rating_float
    except (ValueError, TypeError):
        logger.warning(f"Invalid {param_name} value: {rating_value!r}, ignoring")
        return None


def _is_valid_date_format(date_str):
    if not date_str or not isinstance(date_str, str):
        return False
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))


def format_genre(genre, max_len=30):
    if not genre:
        return "N/A"
    return genre if len(genre) <= max_len else genre[:max_len] + "..."


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
        if top_k < 1:
            logger.warning(f"top_k must be >= 1, got {top_k}, using 1")
            top_k = 1
        query_embedding = embed_text(model, query_text)
        if query_embedding is None:
            logger.error("Failed to get query embedding")
            return []
        filter_parts = []
        if min_rating is not None:
            validated_min = _validate_rating(min_rating, "min_rating")
            if validated_min is not None:
                filter_parts.append(f"vote_average >= {validated_min}")
        if max_rating is not None:
            validated_max = _validate_rating(max_rating, "max_rating")
            if validated_max is not None:
                filter_parts.append(f"vote_average <= {validated_max}")
        if min_rating is not None and max_rating is not None:
            val_min = _validate_rating(min_rating, "min_rating")
            val_max = _validate_rating(max_rating, "max_rating")
            if val_min is not None and val_max is not None and val_min > val_max:
                logger.warning(f"min_rating ({val_min}) > max_rating ({val_max}), swapping values")
                min_rating, max_rating = max_rating, min_rating
                filter_parts = [f for f in filter_parts if not f.startswith("vote_average")]
                filter_parts.append(f"vote_average >= {max_rating}")
                filter_parts.append(f"vote_average <= {min_rating}")
        if min_popularity is not None:
            try:
                pop_float = float(min_popularity)
                if pop_float < 0:
                    logger.warning(f"min_popularity must be >= 0, got {pop_float}, ignoring")
                else:
                    filter_parts.append(f"popularity >= {pop_float}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid min_popularity value: {min_popularity!r}, ignoring")
        if year_filter is not None:
            if min_year is not None or max_year is not None:
                logger.warning("Both year_filter and min_year/max_year provided, using year_filter only")
            validated_year = _validate_year(year_filter, "year_filter")
            if validated_year is not None:
                filter_parts.append(f'release_date like "{validated_year}-%"')
        else:
            validated_min_year = None
            validated_max_year = None
            if min_year is not None:
                validated_min_year = _validate_year(min_year, "min_year")
                if validated_min_year is not None:
                    filter_parts.append(f'release_date >= "{validated_min_year}-01-01"')
            if max_year is not None:
                validated_max_year = _validate_year(max_year, "max_year")
                if validated_max_year is not None:
                    filter_parts.append(f'release_date <= "{validated_max_year}-12-31"')
            if validated_min_year is not None and validated_max_year is not None:
                if validated_min_year > validated_max_year:
                    logger.warning(f"min_year ({validated_min_year}) > max_year ({validated_max_year}), swapping values")
                    filter_parts = [f for f in filter_parts if not f.startswith("release_date")]
                    filter_parts.append(f'release_date >= "{validated_max_year}-01-01"')
                    filter_parts.append(f'release_date <= "{validated_min_year}-12-31"')
        filter_expr = " and ".join(filter_parts) if filter_parts else None
        if filter_expr:
            logger.info(f"Applying database filter: {filter_expr}")
        if genre_filter:
            logger.info(f"Will apply genre post-filter: '{genre_filter}'")
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
                release_date = hit.entity.get('release_date', '')
                if release_date and not _is_valid_date_format(release_date):
                    logger.debug(f"Movie {hit.id} has invalid date format: {release_date}")
                movie_data = {
                    'id': hit.id,
                    'title': hit.entity.get('title', ''),
                    'overview': hit.entity.get('overview', ''),
                    'release_date': release_date,
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

def search_similar_images(collection, model, processor, device, query_image_path: str, top_k: int = 10):
    try:
        if collection is None:
            logger.error("Collection is not loaded")
            return []
        from image_embedder import embed_image
        if top_k < 1:
            logger.warning(f"Invalid top_k {top_k}, using 1")
            top_k = 1
        logger.info(f"Extracting query image embedding...")
        query_embedding = embed_image(model, processor, device, query_image_path)
        if query_embedding is None:
            logger.error("Failed to get query embedding")
            return []
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "filename", "image_path"]
        )
        movies = []
        for hits in results:
            for hit in hits:
                movies.append({
                    'id': hit.id,
                    'title': hit.entity.get('title', ''),
                    'filename': hit.entity.get('filename', ''),
                    'image_path': hit.entity.get('image_path', ''),
                    'similarity': float(hit.distance),
                    'similarity_percent': f"{float(hit.distance) * 100:.2f}%"
                })
        
        logger.info(f"Found {len(movies)} similar images")
        return movies
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

def get_all_images(collection):
    try:
        if collection is None:
            logger.error("Collection is not loaded")
            return []
        results = collection.query(
            expr="id >= 0",
            output_fields=["id", "title", "filename", "image_path"],
            limit=16384
        )
        images = [
            {
                'id': r.get('id'),
                'title': r.get('title', ''),
                'filename': r.get('filename', ''),
                'image_path': r.get('image_path', '')
            }
            for r in results
        ]
        logger.info(f"Retrieved {len(images)} images from collection")
        return images
    except Exception as e:
        logger.error(f"Failed to get all images: {e}")
        return []

def get_collection_stats(collection_name):
    try:
        if not utility.has_collection(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return None
        collection = Collection(name=collection_name)
        collection.load()
        stats = {
            'name': collection_name,
            'num_entities': collection.num_entities,
            'description': collection.description
        }
        logger.info(f"Collection '{collection_name}' has {stats['num_entities']} entities")
        return stats
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return None