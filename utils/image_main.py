import sys
import logging
from pathlib import Path
from image_embedder import load_clip_model, embed_folder
from milvus_vectordb import (
    milvus_connect, milvus_disconnect, create_image_collection,
    save_image_embeddings, search_similar_images, get_collection_stats,
    delete_image_collection,
)
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def build_database(images_folder: str, collection_name: str = None, batch_size: int = None):
    if collection_name is None:
        collection_name = config.IMAGE_COLLECTION_NAME
    if batch_size is None:
        batch_size = config.DEFAULT_BATCH_SIZE
    try:
        logger.info("=" * 80)
        logger.info("Building Image Vector Database with Milvus")
        logger.info("=" * 80)
        
        logger.info("[1/5] Connecting to Milvus...")
        if not milvus_connect():
            logger.error("Failed to connect to Milvus")
            sys.exit(1)
        logger.info("✓ Connected to Milvus")
        
        logger.info("[2/5] Loading CLIP model...")
        model, processor, device = load_clip_model()
        logger.info("✓ Model loaded successfully")
        
        logger.info(f"[3/5] Extracting embeddings from: {images_folder}")
        embeddings, metadata = embed_folder(model, processor, device, images_folder, batch_size)
        logger.info(f"✓ Extracted {len(embeddings)} embeddings")
        
        dimension = embeddings.shape[1]
        logger.info(f"Embedding dimension: {dimension}")
        
        logger.info(f"[4/5] Creating Milvus collection: {collection_name}")
        collection = create_image_collection(collection_name, dimension)
        if collection is None:
            logger.error("Failed to create collection")
            milvus_disconnect()
            sys.exit(1)
        logger.info("✓ Collection created")
        
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


def search_similar_movies(query_image: str, collection_name: str = None, top_k: int = None):
    if collection_name is None:
        collection_name = config.IMAGE_COLLECTION_NAME
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    try:
        logger.info("=" * 80)
        logger.info("Searching for Similar Movies")
        logger.info("=" * 80)
        
        logger.info("[1/4] Connecting to Milvus...")
        if not milvus_connect():
            logger.error("Failed to connect to Milvus")
            sys.exit(1)
        logger.info("✓ Connected")
        
        logger.info("[2/4] Loading CLIP model...")
        model, processor, device = load_clip_model()
        logger.info("✓ Model loaded")
        
        logger.info(f"[3/4] Loading collection: {collection_name}")
        collection = create_image_collection(collection_name)
        if collection is None:
            logger.error("Failed to load collection")
            milvus_disconnect()
            sys.exit(1)
        
        stats = get_collection_stats(collection_name)
        if stats:
            logger.info(f"✓ Collection loaded with {stats['num_entities']} images")
        
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


def rebuild_database(images_folder: str, collection_name: str = None, batch_size: int = None):
    if collection_name is None:
        collection_name = config.IMAGE_COLLECTION_NAME
    if batch_size is None:
        batch_size = config.DEFAULT_BATCH_SIZE
    try:
        logger.info("Rebuilding database - deleting existing collection...")
        
        if not milvus_connect():
            logger.error("Failed to connect to Milvus")
            sys.exit(1)
        
        delete_image_collection(collection_name)
        milvus_disconnect()
        
        logger.info("Building new database...")
        build_database(images_folder, collection_name, batch_size)
        
    except Exception as e:
        logger.error(f"Failed to rebuild database: {e}", exc_info=True)
        sys.exit(1)


def main():
    try:
        logger.info("Image-Based Movie Recommendation System (Milvus)")
        logger.info("=" * 80)
        
        IMAGES_FOLDER = config.DEFAULT_IMAGES_FOLDER
        COLLECTION_NAME = config.IMAGE_COLLECTION_NAME
        QUERY_IMAGE = "../posters/#Alive.jpg"
        TOP_K = 5
        BATCH_SIZE = config.DEFAULT_BATCH_SIZE
        
        milvus_connect()
        stats = get_collection_stats(COLLECTION_NAME)
        collection_ready = stats is not None and stats['num_entities'] > 0
        if stats is not None and stats['num_entities'] == 0:
            logger.info("Collection exists but is empty, deleting it for rebuild...")
            delete_image_collection(COLLECTION_NAME)
        milvus_disconnect()
        
        if not collection_ready:
            logger.info("Collection not found or empty. Building new database...")
            
            if not Path(IMAGES_FOLDER).exists():
                logger.error(f"Images folder not found: {IMAGES_FOLDER}")
                logger.info("Please create the folder and add movie poster images.")
                sys.exit(1)
            
            build_database(IMAGES_FOLDER, COLLECTION_NAME, BATCH_SIZE)
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists")

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