import sys
import logging
from pathlib import Path
from image_embedder import load_clip_model, embed_folder
from milvus_image_vectordb import (
    milvus_connect, milvus_disconnect, create_image_collection,
    save_image_embeddings, search_similar_images, get_collection_stats,
    delete_collection
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_database(images_folder: str, collection_name: str = 'movie_posters', batch_size: int = 32):
    """
    Build Milvus vector database from a folder of images.
    
    Args:
        images_folder: Path to folder containing movie poster images
        collection_name: Name for the Milvus collection
        batch_size: Number of images to process at once
    """
    try:
        logger.info("=" * 80)
        logger.info("Building Image Vector Database with Milvus")
        logger.info("=" * 80)
        
        # Step 1: Connect to Milvus
        logger.info("[1/5] Connecting to Milvus...")
        if not milvus_connect():
            logger.error("Failed to connect to Milvus")
            sys.exit(1)
        logger.info("✓ Connected to Milvus")
        
        # Step 2: Load CLIP model
        logger.info("[2/5] Loading CLIP model...")
        model, processor, device = load_clip_model()
        logger.info("✓ Model loaded successfully")
        
        # Step 3: Extract embeddings from images
        logger.info(f"[3/5] Extracting embeddings from: {images_folder}")
        embeddings, metadata = embed_folder(model, processor, device, images_folder, batch_size)
        logger.info(f"✓ Extracted {len(embeddings)} embeddings")
        
        # Get embedding dimension
        dimension = embeddings.shape[1]
        logger.info(f"Embedding dimension: {dimension}")
        
        # Step 4: Create collection
        logger.info(f"[4/5] Creating Milvus collection: {collection_name}")
        collection = create_image_collection(collection_name, dimension)
        if collection is None:
            logger.error("Failed to create collection")
            milvus_disconnect()
            sys.exit(1)
        logger.info("✓ Collection created")
        
        # Step 5: Save embeddings
        logger.info("[5/5] Saving embeddings to Milvus...")
        if not save_image_embeddings(collection, embeddings, metadata):
            logger.error("Failed to save embeddings")
            milvus_disconnect()
            sys.exit(1)
        logger.info(f"✓ Saved {len(embeddings)} embeddings")
        
        logger.info("=" * 80)
        logger.info("✓ Database build completed successfully")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"File/folder not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        milvus_disconnect()


def search_similar_movies(query_image: str, collection_name: str = 'movie_posters', top_k: int = 10):
    """
    Search for similar movie posters.
    
    Args:
        query_image: Path to query image
        collection_name: Name of Milvus collection
        top_k: Number of similar results to return
        
    Returns:
        List of similar movie results
    """
    try:
        logger.info("=" * 80)
        logger.info("Searching for Similar Movies")
        logger.info("=" * 80)
        
        # Step 1: Connect to Milvus
        logger.info("[1/4] Connecting to Milvus...")
        if not milvus_connect():
            logger.error("Failed to connect to Milvus")
            sys.exit(1)
        logger.info("✓ Connected")
        
        # Step 2: Load model
        logger.info("[2/4] Loading CLIP model...")
        model, processor, device = load_clip_model()
        logger.info("✓ Model loaded")
        
        # Step 3: Load collection
        logger.info(f"[3/4] Loading collection: {collection_name}")
        collection = create_image_collection(collection_name)
        if collection is None:
            logger.error("Failed to load collection")
            milvus_disconnect()
            sys.exit(1)
        
        stats = get_collection_stats(collection_name)
        if stats:
            logger.info(f"✓ Collection loaded with {stats['num_entities']} images")
        
        # Step 4: Search
        logger.info(f"[4/4] Searching for similar images...")
        results = search_similar_images(collection, model, processor, device, query_image, top_k)
        
        logger.info("=" * 80)
        logger.info(f"Top {len(results)} Similar Movies:")
        logger.info("=" * 80)
        
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result['title']} - Similarity: {result['similarity_percent']}")
            logger.info(f"     File: {result['filename']}")
        
        logger.info("=" * 80)
        
        return results
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        milvus_disconnect()


def rebuild_database(images_folder: str, collection_name: str = 'movie_posters', batch_size: int = 32):
    """
    Delete existing collection and rebuild from scratch.
    
    Args:
        images_folder: Path to folder containing movie poster images
        collection_name: Name for the Milvus collection
        batch_size: Number of images to process at once
    """
    try:
        logger.info("Rebuilding database - deleting existing collection...")
        
        if not milvus_connect():
            logger.error("Failed to connect to Milvus")
            sys.exit(1)
        
        delete_collection(collection_name)
        milvus_disconnect()
        
        logger.info("Building new database...")
        build_database(images_folder, collection_name, batch_size)
        
    except Exception as e:
        logger.error(f"Failed to rebuild database: {e}", exc_info=True)
        sys.exit(1)


def main():
    """
    Main pipeline - demonstrates both building and searching.
    """
    try:
        logger.info("Image-Based Movie Recommendation System (Milvus)")
        logger.info("=" * 80)
        
        # Configuration
        IMAGES_FOLDER = "./data/movie_posters"  # Change this to your folder
        COLLECTION_NAME = "movie_posters"
        QUERY_IMAGE = "./data/query_poster.jpg"  # Change this to your query image
        TOP_K = 5
        BATCH_SIZE = 32
        
        # Check if we need to build database
        milvus_connect()
        collection_exists = get_collection_stats(COLLECTION_NAME) is not None
        milvus_disconnect()
        
        if not collection_exists:
            logger.info("Collection not found. Building new database...")
            
            # Validate images folder exists
            if not Path(IMAGES_FOLDER).exists():
                logger.error(f"Images folder not found: {IMAGES_FOLDER}")
                logger.info("Please create the folder and add movie poster images.")
                sys.exit(1)
            
            # Build database
            build_database(IMAGES_FOLDER, COLLECTION_NAME, BATCH_SIZE)
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists")
        
        # Search for similar movies
        if Path(QUERY_IMAGE).exists():
            results = search_similar_movies(QUERY_IMAGE, COLLECTION_NAME, top_k=TOP_K)
        else:
            logger.warning(f"Query image not found: {QUERY_IMAGE}")
            logger.info("Skipping search. Please provide a query image to search.")
        
        logger.info("\n✓ Pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()