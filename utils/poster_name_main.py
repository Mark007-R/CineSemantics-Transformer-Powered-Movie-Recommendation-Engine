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
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

warnings.filterwarnings('ignore')
CONFIG = {
    'image_folder': 'posters',
    'model_name': 'openai/clip-vit-base-patch32',
    'batch_size': 32,
    'db_path': './vector_db',
    'index_name': 'movie_index',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def extract_clip_embeddings(image_folder: str,model_name: str,batch_size: int = 32,device: str = 'cuda') -> Tuple[np.ndarray, List[Dict]]:
    logging.info(f"Loading model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    img_folder = Path(image_folder)
    if not img_folder.exists():
        raise ValueError(f"Folder not found: {image_folder}")
    
    image_files = set()
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        for pattern in [f'*.{ext}', f'*.{ext.upper()}']:
            for path in img_folder.glob(pattern):
                image_files.add(path.resolve())
    
    image_files = sorted(list(image_files))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    logging.info(f"Found {len(image_files)} unique images")
    filenames = [f.name for f in image_files]
    if len(filenames) != len(set(filenames)):
        logging.warning("Duplicate filenames detected (case sensitivity issue)")
        seen = set()
        unique_files = []
        for f in image_files:
            if f.name.lower() not in seen:
                seen.add(f.name.lower())
                unique_files.append(f)
        image_files = unique_files
        logging.info(f"After deduplication: {len(image_files)} unique images")
    
    metadata = []
    for img_path in image_files:
        metadata.append({
            'image_path': str(img_path),
            'filename': img_path.name,
            'title': img_path.stem.replace('_', ' ').replace('-', ' ')
        })
    all_embeddings = []
    logging.info("Extracting embeddings...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(metadata), batch_size)):
            batch_data = metadata[i:i+batch_size]
            images = []
            for item in batch_data:
                try:
                    img = Image.open(item['image_path']).convert('RGB')
                    images.append(img)
                except Exception as e:
                    logging.error(f"Error loading {item['image_path']}: {e}")
                    continue
            
            if len(images) == 0:
                continue
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy())
    embeddings = np.vstack(all_embeddings)
    logging.info(f"Extracted embeddings with shape: {embeddings.shape}")
    return embeddings, metadata

def save_to_vector_db(embeddings: np.ndarray,metadata: List[Dict],db_path: str = './vector_db',index_name: str = 'movie_index') -> Dict:

    Path(db_path).mkdir(parents=True, exist_ok=True)
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_normalized.astype('float32'))
    
    index_path = Path(db_path) / f'{index_name}.index'
    faiss.write_index(index, str(index_path))
    logging.info(f"Saved FAISS index to: {index_path}")
    
    metadata_path = Path(db_path) / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved metadata to: {metadata_path}")
    
    db_info = {
        'index_path': str(index_path),
        'metadata_path': str(metadata_path),
        'num_vectors': len(embeddings),
        'dimension': dimension,
        'index_type': 'IndexFlatIP'
    }
    info_path = Path(db_path) / 'db_info.json'
    with open(info_path, 'w') as f:
        json.dump(db_info, f, indent=2)
    logging.info(f"Saved database info to: {info_path}")
    return db_info

def find_similar_movies(query: Union[str, int],db_path: str = './vector_db',top_k: int = 10) -> List[Dict]:

    with open(Path(db_path) / 'db_info.json', 'r') as f:
        db_info = json.load(f)
    
    with open(Path(db_path) / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if isinstance(query, str):
        query_lower = query.lower()
        query_idx = None
        for i, m in enumerate(metadata):
            if query_lower == m['title'].lower():
                query_idx = i
                break
        
        if query_idx is None:
            for i, m in enumerate(metadata):
                if query_lower in m['title'].lower():
                    query_idx = i
                    break
        
        if query_idx is None:
            available = [m['title'] for m in metadata[:10]]
            raise ValueError(f"Movie '{query}' not found. Available movies (first 10): {available}")
        
        logging.info(f"Found query movie: {metadata[query_idx]['title']}")
    else:
        query_idx = query
        if query_idx >= len(metadata):
            raise ValueError(f"Index {query_idx} out of range (max: {len(metadata)-1})")
    
    index = faiss.read_index(db_info['index_path'])
    
    query_embedding = np.zeros((1, index.d), dtype='float32')
    index.reconstruct(query_idx, query_embedding[0])
    distances, indices = index.search(query_embedding, top_k + 1)
    similar = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != query_idx:
            similar.append({
                'index': int(idx),
                'title': metadata[idx]['title'],
                'filename': metadata[idx]['filename'],
                'similarity': float(dist),
                'similarity_percent': f"{float(dist) * 100:.2f}%",
                'image_path': metadata[idx]['image_path']
            })
    return similar[:top_k]

def get_movie_image(movie_id: Union[str, int],db_path: str = './vector_db') -> Image.Image:
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
    with open(Path(db_path) / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return [m['title'] for m in metadata]
