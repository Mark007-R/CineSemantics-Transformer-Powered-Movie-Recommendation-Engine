import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import json
import faiss
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageEmbedder:
    
    def __init__(self, model_name='openai/clip-vit-base-patch32', device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        logger.info(f"ImageEmbedder initialized with model: {model_name}, device: {self.device}")
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            try:
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                self.model.eval()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
    
    def embed_folder(self, folder_path: str, batch_size: int = 32) -> Tuple[np.ndarray, List[Dict]]:
        self._load_model()
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            raise ValueError(f"Folder not found: {folder_path}")
        image_files = set()
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            for pattern in [f'*.{ext}', f'*.{ext.upper()}']:
                for path in folder.glob(pattern):
                    image_files.add(path.resolve())
        image_files = sorted(list(image_files))
        if len(image_files) == 0:
            logger.error(f"No images found in {folder_path}")
            raise ValueError(f"No images found in {folder_path}")
        logger.info(f"Found {len(image_files)} images")
        metadata = []
        for img_path in image_files:
            metadata.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'title': img_path.stem.replace('_', ' ').replace('-', ' ')
            })
        all_embeddings = []
        logger.info("Extracting embeddings...")
        with torch.no_grad():
            for i in tqdm(range(0, len(metadata), batch_size), desc="Processing batches"):
                batch_data = metadata[i:i+batch_size]
                images = []
                for item in batch_data:
                    try:
                        img = Image.open(item['image_path']).convert('RGB')
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Error loading {item['image_path']}: {e}")
                        continue
                if len(images) == 0:
                    continue
                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        return embeddings, metadata
    
    def embed_image(self, image_path: str) -> np.ndarray:
        self._load_model()
        img_path = Path(image_path)
        if not img_path.exists():
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        logger.info(f"Processing query image: {image_path}")
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error opening image {image_path}: {e}")
            raise ValueError(f"Failed to open image: {e}")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


class VectorDB:
    
    def __init__(self, db_path='./vector_db'):
        self.db_path = Path(db_path)
        self.index = None
        self.metadata = None
        logger.info(f"VectorDB initialized at: {db_path}")
        
    def save(self, embeddings: np.ndarray, metadata: List[Dict], index_name: str = 'movie_index'):
        self.db_path.mkdir(parents=True, exist_ok=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        index_path = self.db_path / f'{index_name}.index'
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to: {index_path}")
        metadata_path = self.db_path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {metadata_path}")
        info = {
            'num_vectors': len(embeddings),
            'dimension': dimension,
            'index_path': str(index_path),
            'metadata_path': str(metadata_path)
        }
        info_path = self.db_path / 'db_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved {len(embeddings)} vectors to {self.db_path}")
        return info
    
    def load(self):
        info_path = self.db_path / 'db_info.json'
        if not info_path.exists():
            logger.error(f"Database not found at {self.db_path}")
            raise ValueError(f"Database not found at {self.db_path}")
        with open(info_path, 'r') as f:
            info = json.load(f)
        self.index = faiss.read_index(info['index_path'])
        with open(info['metadata_path'], 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded {len(self.metadata)} vectors from {self.db_path}")
        return info


class SimilaritySearch:
    
    def __init__(self, db_path='./vector_db', model_name='openai/clip-vit-base-patch32'):
        self.db_path = db_path
        self.model_name = model_name
        self.db = VectorDB(db_path)
        self.embedder = ImageEmbedder(model_name)
        self.db.load()
        logger.info("SimilaritySearch initialized and database loaded")
    
    def find_similar(self, query_image_path: str, top_k: int = 10) -> List[Dict]:
        logger.info(f"Searching for top {top_k} similar images to: {query_image_path}")
        query_embedding = self.embedder.embed_image(query_image_path)
        distances, indices = self.db.index.search(query_embedding.astype('float32'), top_k)  
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            metadata = self.db.metadata[idx]
            results.append({
                'index': int(idx),
                'title': metadata['title'],
                'filename': metadata['filename'],
                'image_path': metadata['image_path'],
                'similarity': float(dist),
                'similarity_percent': f"{float(dist) * 100:.2f}%"
            })
        logger.info(f"Found {len(results)} similar images")
        return results
    
    def get_all_movies(self) -> List[str]:
        return [m['title'] for m in self.db.metadata]