"""
Configuration file for CineSemantics Movie Recommendation Engine
Contains all configurable parameters for text and image embeddings, 
database connections, and search functionality.
"""
# MODEL CONFIGURATIONS
# Day-2 bake-off champion: e5-base-v2 (768-dim) beat the shipped MiniLM-L6-v2
# (384-dim) on content-retrieval NDCG@10 0.0482 vs 0.0295 (+63%). E5 needs an
# instruction prefix; for symmetric movie<->movie/query similarity we use
# "query: " on every text (the scheme the Day-2/4 numbers were measured under).
TEXT_MODEL_NAME = 'intfloat/e5-base-v2'
TEXT_EMBEDDING_DIMENSION = 768
TEXT_MODEL_PREFIX = 'query: '

IMAGE_MODEL_NAME = 'openai/clip-vit-base-patch32'
IMAGE_EMBEDDING_DIMENSION = 512

# DATABASE CONFIGURATIONS
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
MILVUS_DEFAULT_SERVER_HOST = '127.0.0.1'

TEXT_COLLECTION_NAME = 'movie_collection'
IMAGE_COLLECTION_NAME = 'movie_posters'

# Day-4 ANN sweep champion: HNSW matched exact NDCG@10 at ~3.9x lower p95 latency
# than the previously-shipped IVF_FLAT default. IVF params kept for fallback.
INDEX_METRIC_TYPE = 'IP'
INDEX_TYPE = 'HNSW'
INDEX_NLIST = 128            # IVF_FLAT fallback
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

SEARCH_METRIC_TYPE = 'IP'
SEARCH_NPROBE = 10          # IVF_FLAT fallback
SEARCH_EF = 64             # HNSW efSearch (Day-4 champion operating point)
SEARCH_MULTIPLIER = 3


def build_index_params():
    """Index params matching the active INDEX_TYPE (HNSW champion, else IVF_FLAT)."""
    if INDEX_TYPE == 'HNSW':
        return {"metric_type": INDEX_METRIC_TYPE, "index_type": "HNSW",
                "params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION}}
    return {"metric_type": INDEX_METRIC_TYPE, "index_type": INDEX_TYPE,
            "params": {"nlist": INDEX_NLIST}}


def build_search_params():
    """Search params matching the active INDEX_TYPE."""
    if INDEX_TYPE == 'HNSW':
        return {"metric_type": SEARCH_METRIC_TYPE, "params": {"ef": SEARCH_EF}}
    return {"metric_type": SEARCH_METRIC_TYPE, "params": {"nprobe": SEARCH_NPROBE}}

# DATA PROCESSING CONFIGURATIONS
DEFAULT_BATCH_SIZE = 32
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 128
INSERT_BATCH_SIZE = 100

DEFAULT_TEXT_COLUMN = 'Overview'

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

# FIELD LENGTH LIMITS
TEXT_TITLE_MAX_LENGTH = 1000
TEXT_OVERVIEW_MAX_LENGTH = 20000
TEXT_RELEASE_DATE_MAX_LENGTH = 50
TEXT_GENRE_MAX_LENGTH = 200
TEXT_ORIGINAL_LANGUAGE_MAX_LENGTH = 200
TEXT_POSTER_URL_MAX_LENGTH = 1000

IMAGE_TITLE_MAX_LENGTH = 500
IMAGE_FILENAME_MAX_LENGTH = 500
IMAGE_PATH_MAX_LENGTH = 1000

GENRE_DISPLAY_MAX_LENGTH = 30

# VALIDATION RANGES
MIN_YEAR = 1800
MAX_YEAR = 2100
DATE_FORMAT_PATTERN = r'^\d{4}-\d{2}-\d{2}$'

MIN_RATING = 0.0
MAX_RATING = 10.0

MIN_POPULARITY = 0.0

MAX_QUERY_LIMIT = 16384

# DEFAULT VALUES
DEFAULT_FLOAT_VALUE = 0.0
DEFAULT_INT_VALUE = 0
DEFAULT_STRING_VALUE = ''

DEFAULT_TOP_K = 10

# FILE PATHS (for main scripts)
DEFAULT_IMAGES_FOLDER = '../posters'
DEFAULT_CSV_PATH = '../data/9000plus.csv'

# LOGGING CONFIGURATION
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'