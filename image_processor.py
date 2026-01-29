"""
Movie Poster Similarity Search using CLIP
==========================================

This script extracts visual features from movie posters and enables 
finding similar movies based on visual similarity using IMAGE INPUT.

Usage:
    # Extract embeddings and create database
    python image_processor.py --extract --folder posters
    
    # Search for similar movies using an IMAGE
    python image_processor.py --search_image query_poster.jpg --top_k 10
    
    # Search by movie name (fallback)
    python image_processor.py --search "Avengers" --top_k 10
    
    # List all movies
    python image_processor.py --list
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Union
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import json
import faiss
import warnings
import argparse

warnings.filterwarnings('ignore')


# Configuration
CONFIG = {
    'image_folder': 'posters',
    'model_name': 'openai/clip-vit-base-patch32',  # Change to clip-vit-large-patch14 for better quality
    'batch_size': 32,
    'db_path': './vector_db',
    'index_name': 'movie_index',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def extract_clip_embeddings(
    image_folder: str,
    model_name: str,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Extract embeddings from images using CLIP.
    
    Args:
        image_folder: Path to folder containing images
        model_name: HuggingFace model name
        batch_size: Number of images to process at once
        device: 'cuda' or 'cpu'
    
    Returns:
        embeddings: Numpy array of shape (n_images, embedding_dim)
        metadata: List of dicts containing image info
    """
    print(f"Loading model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    # Find all image files (case-insensitive, avoiding duplicates)
    img_folder = Path(image_folder)
    if not img_folder.exists():
        raise ValueError(f"Folder not found: {image_folder}")
    
    # Use resolved absolute paths to avoid duplicates on case-insensitive filesystems
    image_files = set()
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        for pattern in [f'*.{ext}', f'*.{ext.upper()}']:
            for path in img_folder.glob(pattern):
                # Resolve to absolute path to ensure uniqueness
                image_files.add(path.resolve())
    
    image_files = sorted(list(image_files))  # Sort for consistency
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"Found {len(image_files)} unique images")
    
    # Debug: Check for any potential duplicates by filename
    filenames = [f.name for f in image_files]
    if len(filenames) != len(set(filenames)):
        print("WARNING: Duplicate filenames detected (case sensitivity issue)")
        # Remove duplicates by keeping only unique lowercase filenames
        seen = set()
        unique_files = []
        for f in image_files:
            if f.name.lower() not in seen:
                seen.add(f.name.lower())
                unique_files.append(f)
        image_files = unique_files
        print(f"After deduplication: {len(image_files)} unique images")
    
    # Create metadata
    metadata = []
    for img_path in image_files:
        metadata.append({
            'image_path': str(img_path),
            'filename': img_path.name,
            'title': img_path.stem.replace('_', ' ').replace('-', ' ')
        })
    
    # Extract embeddings in batches
    all_embeddings = []
    print("Extracting embeddings...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(metadata), batch_size)):
            batch_data = metadata[i:i+batch_size]
            
            # Load and convert images
            images = []
            for item in batch_data:
                try:
                    img = Image.open(item['image_path']).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {item['image_path']}: {e}")
                    continue
            
            if len(images) == 0:
                continue
            
            # Process batch
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
            
            # Normalize embeddings (important for cosine similarity)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(image_features.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    print(f"Extracted embeddings with shape: {embeddings.shape}")
    
    return embeddings, metadata


def extract_single_image_embedding(
    image_path: str,
    model_name: str = None,
    device: str = None
) -> np.ndarray:
    """
    Extract embedding from a single query image.
    
    Args:
        image_path: Path to the query image
        model_name: HuggingFace model name (uses CONFIG if None)
        device: 'cuda' or 'cpu' (uses CONFIG if None)
    
    Returns:
        embedding: Numpy array of shape (1, embedding_dim)
    """
    if model_name is None:
        model_name = CONFIG['model_name']
    if device is None:
        device = CONFIG['device']
    
    print(f"Loading model for query image: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    # Load and process image
    print(f"Processing query image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        
        # Normalize embedding
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()


def save_to_vector_db(
    embeddings: np.ndarray,
    metadata: List[Dict],
    db_path: str = './vector_db',
    index_name: str = 'movie_index'
) -> Dict:
    """
    Save embeddings to FAISS vector database.
    
    Args:
        embeddings: Numpy array of embeddings
        metadata: List of metadata dicts
        db_path: Path to save database
        index_name: Name for the index file
    
    Returns:
        db_info: Dictionary with database information
    """
    Path(db_path).mkdir(parents=True, exist_ok=True)
    
    # Ensure embeddings are normalized for cosine similarity
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index (Inner Product = Cosine Similarity for normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
    index.add(embeddings_normalized.astype('float32'))
    
    # Save index
    index_path = Path(db_path) / f'{index_name}.index'
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to: {index_path}")
    
    # Save metadata
    metadata_path = Path(db_path) / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {metadata_path}")
    
    # Save database info
    db_info = {
        'index_path': str(index_path),
        'metadata_path': str(metadata_path),
        'num_vectors': len(embeddings),
        'dimension': dimension,
        'index_type': 'IndexFlatIP',
        'model_name': CONFIG['model_name']
    }
    
    info_path = Path(db_path) / 'db_info.json'
    with open(info_path, 'w') as f:
        json.dump(db_info, f, indent=2)
    print(f"Saved database info to: {info_path}")
    
    return db_info


def search_by_image(
    query_image_path: str,
    db_path: str = './vector_db',
    top_k: int = 10,
    model_name: str = None
) -> List[Dict]:
    """
    Find similar movies by uploading a query image.
    
    Args:
        query_image_path: Path to the query image
        db_path: Path to vector database
        top_k: Number of similar movies to return
        model_name: Model name (uses saved model if None)
    
    Returns:
        List of similar movies with similarity scores
    """
    # Load database info and metadata
    with open(Path(db_path) / 'db_info.json', 'r') as f:
        db_info = json.load(f)
    
    with open(Path(db_path) / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Use the same model that was used to create the database
    if model_name is None:
        model_name = db_info.get('model_name', CONFIG['model_name'])
    
    print(f"Using model: {model_name}")
    
    # Extract embedding from query image
    query_embedding = extract_single_image_embedding(
        query_image_path,
        model_name=model_name,
        device=CONFIG['device']
    )
    
    # Load FAISS index
    index = faiss.read_index(db_info['index_path'])
    
    # Search for similar movies
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # Build results
    similar = []
    for idx, dist in zip(indices[0], distances[0]):
        similar.append({
            'index': int(idx),
            'title': metadata[idx]['title'],
            'filename': metadata[idx]['filename'],
            'similarity': float(dist),
            'similarity_percent': f"{float(dist) * 100:.2f}%",
            'image_path': metadata[idx]['image_path']
        })
    
    return similar


def find_similar_movies(
    query: Union[str, int],
    db_path: str = './vector_db',
    top_k: int = 10
) -> List[Dict]:
    """
    Find movies similar to the query movie (by name or index).
    
    Args:
        query: Movie title (str) or index (int)
        db_path: Path to vector database
        top_k: Number of similar movies to return
    
    Returns:
        List of similar movies with similarity scores
    """
    # Load database info and metadata
    with open(Path(db_path) / 'db_info.json', 'r') as f:
        db_info = json.load(f)
    
    with open(Path(db_path) / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Find query index
    if isinstance(query, str):
        query_lower = query.lower()
        query_idx = None
        
        # Try exact match first
        for i, m in enumerate(metadata):
            if query_lower == m['title'].lower():
                query_idx = i
                break
        
        # Try partial match
        if query_idx is None:
            for i, m in enumerate(metadata):
                if query_lower in m['title'].lower():
                    query_idx = i
                    break
        
        if query_idx is None:
            available = [m['title'] for m in metadata[:10]]
            raise ValueError(f"Movie '{query}' not found. Available movies (first 10): {available}")
        
        print(f"Found query movie: {metadata[query_idx]['title']}")
    else:
        query_idx = query
        if query_idx >= len(metadata):
            raise ValueError(f"Index {query_idx} out of range (max: {len(metadata)-1})")
    
    # Load index
    index = faiss.read_index(db_info['index_path'])
    
    # Reconstruct query embedding
    query_embedding = np.zeros((1, index.d), dtype='float32')
    index.reconstruct(query_idx, query_embedding[0])
    
    # Search for similar movies (top_k + 1 to exclude query itself)
    distances, indices = index.search(query_embedding, top_k + 1)
    
    # Build results (excluding the query movie itself)
    similar = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != query_idx:  # Skip the query movie
            similar.append({
                'index': int(idx),
                'title': metadata[idx]['title'],
                'filename': metadata[idx]['filename'],
                'similarity': float(dist),
                'similarity_percent': f"{float(dist) * 100:.2f}%",
                'image_path': metadata[idx]['image_path']
            })
    
    return similar[:top_k]


def get_movie_image(
    movie_id: Union[str, int],
    db_path: str = './vector_db'
) -> Image.Image:
    """
    Get the poster image for a movie.
    
    Args:
        movie_id: Movie title (str) or index (int)
        db_path: Path to vector database
    
    Returns:
        PIL Image object
    """
    with open(Path(db_path) / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if isinstance(movie_id, str):
        movie = next((m for m in metadata if movie_id.lower() in m['title'].lower()), None)
        if movie is None:
            raise ValueError(f"Movie '{movie_id}' not found")
    else:
        movie = metadata[movie_id]
    
    return Image.open(movie['image_path'])


def list_all_movies(db_path: str = './vector_db') -> List[str]:
    """
    List all movies in the database.
    
    Args:
        db_path: Path to vector database
    
    Returns:
        List of movie titles
    """
    with open(Path(db_path) / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return [m['title'] for m in metadata]


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Movie Poster Similarity Search')
    parser.add_argument('--extract', action='store_true', help='Extract embeddings from images')
    parser.add_argument('--search_image', type=str, help='Search using an IMAGE file (e.g., query_poster.jpg)')
    parser.add_argument('--search', type=str, help='Search for similar movies by name')
    parser.add_argument('--list', action='store_true', help='List all movies in database')
    parser.add_argument('--folder', type=str, default='posters', help='Image folder path')
    parser.add_argument('--top_k', type=int, default=10, help='Number of similar movies to return')
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32', 
                       help='Model to use for embeddings')
    parser.add_argument('--db_path', type=str, default='./vector_db', help='Database path')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG['image_folder'] = args.folder
    CONFIG['model_name'] = args.model
    CONFIG['db_path'] = args.db_path
    
    if args.extract:
        # Extract embeddings and create database
        print("=" * 50)
        print("EXTRACTING EMBEDDINGS FROM IMAGES")
        print("=" * 50)
        
        embeddings, metadata = extract_clip_embeddings(
            image_folder=CONFIG['image_folder'],
            model_name=CONFIG['model_name'],
            batch_size=CONFIG['batch_size'],
            device=CONFIG['device']
        )
        
        print(f"\nSuccessfully processed {len(metadata)} movies")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Save to database
        print("\n" + "=" * 50)
        print("SAVING TO VECTOR DATABASE")
        print("=" * 50)
        
        db_info = save_to_vector_db(
            embeddings=embeddings,
            metadata=metadata,
            db_path=CONFIG['db_path'],
            index_name=CONFIG['index_name']
        )
        
        print(f"\nDatabase created successfully!")
        print(f"Location: {CONFIG['db_path']}")
        print(f"Total vectors: {db_info['num_vectors']}")
        
    elif args.search_image:
        # Search for similar movies using an IMAGE
        print("=" * 50)
        print(f"SEARCHING BY IMAGE: '{args.search_image}'")
        print("=" * 50)
        
        results = search_by_image(
            query_image_path=args.search_image,
            db_path=CONFIG['db_path'],
            top_k=args.top_k,
            model_name=args.model
        )
        
        print(f"\nTop {args.top_k} similar movies:\n")
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']}")
            print(f"   Similarity: {movie['similarity_percent']} ({movie['similarity']:.4f})")
            print(f"   File: {movie['filename']}")
            print(f"   Path: {movie['image_path']}\n")
        
    elif args.search:
        # Search for similar movies by name
        print("=" * 50)
        print(f"SEARCHING FOR MOVIES SIMILAR TO: '{args.search}'")
        print("=" * 50)
        
        results = find_similar_movies(
            query=args.search,
            db_path=CONFIG['db_path'],
            top_k=args.top_k
        )
        
        print(f"\nTop {args.top_k} similar movies:\n")
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']}")
            print(f"   Similarity: {movie['similarity_percent']} ({movie['similarity']:.4f})")
            print(f"   File: {movie['filename']}\n")
    
    elif args.list:
        # List all movies
        print("=" * 50)
        print("ALL MOVIES IN DATABASE")
        print("=" * 50)
        
        movies = list_all_movies(CONFIG['db_path'])
        print(f"\nTotal movies: {len(movies)}\n")
        
        for i, title in enumerate(movies, 1):
            print(f"{i}. {title}")
    
    else:
        # No arguments provided, run default workflow
        print("=" * 50)
        print("MOVIE POSTER SIMILARITY SEARCH")
        print("=" * 50)
        print(f"Using device: {CONFIG['device']}")
        print(f"Model: {CONFIG['model_name']}")
        print()
        
        # Step 1: Extract embeddings
        print("Step 1: Extracting embeddings from images...")
        embeddings, metadata = extract_clip_embeddings(
            image_folder=CONFIG['image_folder'],
            model_name=CONFIG['model_name'],
            batch_size=CONFIG['batch_size'],
            device=CONFIG['device']
        )
        
        print(f"\nSuccessfully processed {len(metadata)} movies")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Step 2: Save to database
        print("\nStep 2: Saving to vector database...")
        db_info = save_to_vector_db(
            embeddings=embeddings,
            metadata=metadata,
            db_path=CONFIG['db_path'],
            index_name=CONFIG['index_name']
        )
        
        print(f"\nDatabase created successfully!")
        print(f"Location: {CONFIG['db_path']}")
        print(f"Total vectors: {db_info['num_vectors']}")
        
        print("\n" + "=" * 50)
        print("DONE!")
        print("=" * 50)
        print("\nUsage examples:")
        print("  # Search by IMAGE (recommended for webapp)")
        print("  python image_processor.py --search_image my_poster.jpg --top_k 10")
        print()
        print("  # Search by movie name (fallback)")
        print("  python image_processor.py --search 'Avengers' --top_k 10")
        print()
        print("  # List all movies")
        print("  python image_processor.py --list")
        print()
        print("  # Re-extract with better model")
        print("  python image_processor.py --extract --model openai/clip-vit-large-patch14")


if __name__ == "__main__":
    main()