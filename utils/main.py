import sys
import logging
from text_embedder import load_model
from milvus_vectordb import milvus_connect, milvus_disconnect, create_collection, search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_genre(genre, max_len=30):
    return genre if len(genre) <= max_len else genre[:max_len] + "..."

def main():
    try:
        logger.info("Starting Movie Recommendation System")

        if not milvus_connect():
            logger.error("Failed to connect to Milvus")
            sys.exit(1)

        model = load_model()
        if not model:
            logger.error("Failed to load model")
            milvus_disconnect()
            sys.exit(1)

        collection = create_collection()
        if not collection:
            logger.error("Failed to create/load collection")
            milvus_disconnect()
            sys.exit(1)

        logger.info("=" * 80)
        
        # Query 1: Basic search
        logger.info("Query 1: Basic search - 'space adventure'")
        results = search(collection, model, "space adventure", top_k=5)
        for i, m in enumerate(results, 1):
            logger.info(f"  {i}. {m['title']} ({m['release_date'][:4]}) - Rating: {m['vote_average']} - {m['similarity_percent']}")
        
        # Query 2: With rating filter
        logger.info("Query 2: Highly-rated space movies (min_rating=7.0)")
        results = search(collection, model, "space adventure", top_k=5, min_rating=7.0)
        for i, m in enumerate(results, 1):
            logger.info(f"  {i}. {m['title']} ({m['release_date'][:4]}) - Rating: {m['vote_average']} - {m['similarity_percent']}")
        
        # Query 3: Genre + year filters
        logger.info("Query 3: Recent action movies (2020+, genre='Action')")
        results = search(collection, model, "intense action thriller", top_k=5, genre_filter="Action", min_year=2020)
        for i, m in enumerate(results, 1):
            logger.info(f"  {i}. {m['title']} ({m['release_date'][:4]}) - {format_genre(m['genre'])} - {m['similarity_percent']}")
        
        # Query 4: Multiple filters
        logger.info("Query 4: Rom-coms (rating 6-8, popularity>=50, genre='Romance')")
        results = search(collection, model, "romantic comedy", top_k=5, min_rating=6.0, max_rating=8.0, min_popularity=50, genre_filter="Romance")
        for i, m in enumerate(results, 1):
            logger.info(f"  {i}. {m['title']} - Rating: {m['vote_average']}, Pop: {m['popularity']:.0f} - {m['similarity_percent']}")
        
        # Query 5: 90s sci-fi
        logger.info("Query 5: 90s sci-fi (1990-1999, min_rating=7.0, genre='Science Fiction')")
        results = search(collection, model, "science fiction thriller", top_k=5, min_year=1990, max_year=1999, min_rating=7.0, genre_filter="Science Fiction")
        for i, m in enumerate(results, 1):
            logger.info(f"  {i}. {m['title']} ({m['release_date'][:4]}) - Rating: {m['vote_average']} - {m['similarity_percent']}")
        
        # Query 6: Animated family movies
        logger.info("Query 6: High-rated animated (min_rating=7.5, genre='Animation')")
        results = search(collection, model, "family adventure", top_k=5, min_rating=7.5, genre_filter="Animation")
        for i, m in enumerate(results, 1):
            logger.info(f"  {i}. {m['title']} ({m['release_date'][:4]}) - Rating: {m['vote_average']} - {m['similarity_percent']}")
        
        # Query 7: Specific year
        logger.info("Query 7: Movies from 2021 (year_filter=2021)")
        results = search(collection, model, "action adventure", top_k=5, year_filter=2021)
        for i, m in enumerate(results, 1):
            logger.info(f"  {i}. {m['title']} ({m['release_date']}) - {m['similarity_percent']}")
        
        # Query 8: Classic movies
        logger.info("Query 8: Classic dramas before 1980 (max_year=1980, min_rating=7.0)")
        results = search(collection, model, "drama", top_k=5, max_year=1980, min_rating=7.0)
        for i, m in enumerate(results, 1):
            logger.info(f"  {i}. {m['title']} ({m['release_date'][:4]}) - Rating: {m['vote_average']} - {m['similarity_percent']}")
        
        logger.info("=" * 80)
        logger.info("All searches completed successfully")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        milvus_disconnect()

if __name__ == "__main__":
    main()