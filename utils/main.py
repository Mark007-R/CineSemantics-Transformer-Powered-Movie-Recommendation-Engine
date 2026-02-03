from text_embedder import load_model, embed_csv
from milvus_vectordb import milvus_connect, milvus_disconnect, create_collection, save_csv_embeddings, search

def main():
    milvus_connect()
    model = load_model()
    collection = create_collection()
    embeddings, metadata = embed_csv(model, "./data/9000plus.csv")
    save_csv_embeddings(collection, embeddings, metadata)
    results = search(collection, model, "a movie about space adventure", top_k=5)
    for movie in results:
        print(f"{movie['title']} - {movie['similarity_percent']}")
    milvus_disconnect()

if __name__ == "__main__":
    main()