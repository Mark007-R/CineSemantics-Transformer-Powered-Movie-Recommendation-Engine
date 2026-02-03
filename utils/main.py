import logging
from text_embedder import embed_text
from pymilvus import Collection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_similar_movies(collection, model, query_text: str, top_k: int = 10):
    try:
        if collection is None:
            logger.error("Collection is not loaded")
            return []
        if model is None:
            logger.error("Model is not loaded")
            return []
        if not query_text or not query_text.strip():
            logger.error("Query text is empty")
            return []
        logger.info(f"Processing query: '{query_text[:100]}...'")
        query_embedding = embed_text(model, query_text)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        logger.info(f"Searching for top {top_k} similar movies...")
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
        logger.error(f"Pipeline failed: {e}")
        return []