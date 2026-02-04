from text_embedder import load_model
from milvus_vectordb import milvus_connect, milvus_disconnect, create_collection, search

def main():
    milvus_connect()
    model = load_model()
    collection = create_collection()
    
    results = search(collection, model, "space adventure", top_k=5)
    for movie in results:
        print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
    
    results = search(collection, model, "space adventure", top_k=5, min_rating=7.0)
    for movie in results:
        print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
    
    results = search(collection, model, "intense action thriller", top_k=5, 
                    genre_filter="Action", min_year=2020)
    for movie in results:
        print(f"{movie['title']} ({movie['release_date'][:4]}) - Genre: {movie['genre'][:30]}... - {movie['similarity_percent']}")
    
    results = search(collection, model, "funny romantic comedy", top_k=5,
                    min_rating=6.0, max_rating=8.0, min_popularity=50, genre_filter="Romance")
    for movie in results:
        print(f"{movie['title']} - Rating: {movie['vote_average']}, Popularity: {movie['popularity']:.1f} - {movie['similarity_percent']}")
    
    results = search(collection, model, "science fiction thriller", top_k=5,
                    min_year=1990, max_year=1999, min_rating=7.0, genre_filter="Science Fiction")
    for movie in results:
        print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
    
    results = search(collection, model, "family adventure", top_k=5,
                    min_rating=7.5, genre_filter="Animation")
    for movie in results:
        print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
    
    results = search(collection, model, "action adventure", top_k=5,
                    year_filter=2021)
    for movie in results:
        print(f"{movie['title']} ({movie['release_date']}) - {movie['similarity_percent']}")
    
    results = search(collection, model, "drama", top_k=5,
                    max_year=1980, min_rating=7.0)
    for movie in results:
        print(f"{movie['title']} ({movie['release_date'][:4]}) - Rating: {movie['vote_average']} - {movie['similarity_percent']}")
    
    milvus_disconnect()

if __name__ == "__main__":
    main()