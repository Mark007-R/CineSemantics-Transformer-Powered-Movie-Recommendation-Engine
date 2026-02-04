import sys
import logging
from text_embedder import load_model
from milvus_vectordb import milvus_connect, milvus_disconnect, create_collection, search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting movie recommendation system...")
        
        if not milvus_connect():
            logger.error("Failed to connect to Milvus. Exiting.")
            sys.exit(1)
        
        logger.info("Loading model...")
        model = load_model()
        if model is None:
            logger.error("Failed to load model. Exiting.")
            milvus_disconnect()
            sys.exit(1)
        
        logger.info("Creating/loading collection...")
        collection = create_collection()
        if collection is None:
            logger.error("Failed to create/load collection. Exiting.")
            milvus_disconnect()
            sys.exit(1)
        
        logger.info("Running search queries...")
        
        logger.info("Query 1: 'space adventure' (top 5)")
        results = search(collection, model, "space adventure", top_k=5)
        for movie in results:
            print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
        
        logger.info("Query 2: 'space adventure' with min_rating=7.0")
        results = search(collection, model, "space adventure", top_k=5, min_rating=7.0)
        for movie in results:
            print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
        
        logger.info("Query 3: 'intense action thriller' with genre and year filters")
        results = search(collection, model, "intense action thriller", top_k=5, 
                        genre_filter="Action", min_year=2020)
        for movie in results:
            print(f"{movie['title']} ({movie['release_date'][:4]}) - Genre: {movie['genre'][:30]}... - {movie['similarity_percent']}")
        
        logger.info("Query 4: 'funny romantic comedy' with multiple filters")
        results = search(collection, model, "funny romantic comedy", top_k=5,
                        min_rating=6.0, max_rating=8.0, min_popularity=50, genre_filter="Romance")
        for movie in results:
            print(f"{movie['title']} - Rating: {movie['vote_average']}, Popularity: {movie['popularity']:.1f} - {movie['similarity_percent']}")
        
        logger.info("Query 5: 'science fiction thriller' from the 1990s")
        results = search(collection, model, "science fiction thriller", top_k=5,
                        min_year=1990, max_year=1999, min_rating=7.0, genre_filter="Science Fiction")
        for movie in results:
            print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
        
        logger.info("Query 6: 'family adventure' animation with high rating")
        results = search(collection, model, "family adventure", top_k=5,
                        min_rating=7.5, genre_filter="Animation")
        for movie in results:
            print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
        
        logger.info("Query 7: 'action adventure' from 2021")
        results = search(collection, model, "action adventure", top_k=5,
                        year_filter=2021)
        for movie in results:
            print(f"{movie['title']} ({movie['release_date']}) - {movie['similarity_percent']}")
        
        logger.info("Query 8: 'drama' before 1980 with high rating")
        results = search(collection, model, "drama", top_k=5,
                        max_year=1980, min_rating=7.0)
        for movie in results:
            print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
        
        logger.info("All searches completed successfully")
    
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        logger.info("Cleaning up and disconnecting...")
        milvus_disconnect()
        logger.info("Application finished")

if __name__ == "__main__":
    main()