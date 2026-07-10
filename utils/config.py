"""
Configuration file for CineSemantics Movie Recommendation Engine
Contains all configurable parameters for text and image embeddings, 
database connections, and search functionality.
"""
# MODEL CONFIGURATIONS
# Day-2 embedding bake-off champion: e5-base-v2 (768d) beat the original
# MiniLM-L6-v2 (384d) on the held-out co-rating eval by +63% NDCG@10.
# e5 expects an instruction prefix; for the symmetric "more like this" task both
# sides use "query: " (see utils/text_embedder.py + src/eval/embedding_comparison.py).
TEXT_MODEL_NAME = 'intfloat/e5-base-v2'
TEXT_EMBEDDING_DIMENSION = 768
TEXT_QUERY_PREFIX = 'query: '

IMAGE_MODEL_NAME = 'openai/clip-vit-base-patch32'
IMAGE_EMBEDDING_DIMENSION = 512

# DATABASE CONFIGURATIONS
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
MILVUS_DEFAULT_SERVER_HOST = '127.0.0.1'

TEXT_COLLECTION_NAME = 'movie_collection'
IMAGE_COLLECTION_NAME = 'movie_posters'

INDEX_METRIC_TYPE = 'IP'
# Day-4 ANN sweep champion: HNSW Pareto-dominated the original IVF_FLAT — exact
# NDCG@10 at ~4x lower p95 latency. IVF params kept below as a fallback.
INDEX_TYPE = 'HNSW'
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
INDEX_NLIST = 128          # IVF_FLAT fallback only

SEARCH_METRIC_TYPE = 'IP'
SEARCH_EF = 64             # HNSW efSearch (Day-4 champion)
SEARCH_NPROBE = 10         # IVF_FLAT fallback only
SEARCH_MULTIPLIER = 3

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