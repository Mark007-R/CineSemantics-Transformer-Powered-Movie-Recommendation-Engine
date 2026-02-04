import sys
import logging
from text_embedder import load_model, embed_csv
from milvus_vectordb import milvus_connect, milvus_disconnect, create_collection, save_csv_embeddings, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def main():
    if not milvus_connect():
        logging.error("Could not connect to Milvus. Make sure the server is running on localhost:19530.")
        sys.exit(1)

    model = load_model()
    if model is None:
        logging.error("Failed to load the embedding model.")
        milvus_disconnect()
        sys.exit(1)

    collection = create_collection()
    if collection is None:
        logging.error("Failed to create or load the Milvus collection.")
        milvus_disconnect()
        sys.exit(1)

    embeddings, metadata = embed_csv(model, "../data/9000plus.csv")
    if embeddings is None or metadata is None:
        logging.error("Failed to embed CSV data. Check the CSV path and column names.")
        milvus_disconnect()
        sys.exit(1)

    save_csv_embeddings(collection, embeddings, metadata)
    logging.info("CSV embeddings saved successfully.")

    results = search(collection, model, "classic 90s thriller", top_k=5)
    logging.info("Search completed. Top results:")
    for movie in results:
        logging.info(f"{movie['title']} - {movie['similarity_percent']}")

    milvus_disconnect()
    logging.info("Disconnected from Milvus.")


if __name__ == "__main__":
    main()