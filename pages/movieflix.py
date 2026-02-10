import streamlit as st
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path
import logging
import random
import json
from datetime import datetime

utils_dir = Path(__file__).resolve().parent.parent / 'utils'
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

try:
    import config
    from text_embedder import load_model
    from image_embedder import load_clip_model
    from milvus_vectordb import (milvus_connect, milvus_disconnect,
        create_text_collection, create_image_collection,
        search_similar_movies, search_similar_images,
        get_collection_stats
    )
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="MovieFlix - AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 40%, #0d0d2b 100%);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        padding: 8px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: rgba(255, 255, 255, 0.5);
        font-weight: 600;
        font-size: 0.95rem;
        padding: 14px 28px;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.8);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e50914 0%, #b20710 100%) !important;
        color: #fff !important;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.4);
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(20, 20, 50, 0.95) 50%, rgba(229, 9, 20, 0.05) 100%);
        padding: 50px 40px;
        border-radius: 28px;
        margin-bottom: 40px;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(229, 9, 20, 0.5), transparent);
    }
    
    .logo-text {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e50914 0%, #ff6b6b 40%, #ffd93d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -2px;
        text-shadow: 0 0 80px rgba(229, 9, 20, 0.5);
    }
    
    .tagline {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.3rem;
        margin-top: 15px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .search-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 24px;
        padding: 35px;
        margin-bottom: 35px;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    .search-box::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 24px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.3), transparent, rgba(229, 9, 20, 0.1));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }
    
    .movie-card {
        background: linear-gradient(145deg, rgba(30, 30, 60, 0.8) 0%, rgba(20, 20, 40, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 24px;
        margin: 24px 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 100%;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, transparent 50%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .movie-card:hover {
        transform: translateY(-12px) scale(1.01);
        box-shadow: 0 24px 60px rgba(229, 9, 20, 0.25), 0 8px 32px rgba(0, 0, 0, 0.4);
        border-color: rgba(229, 9, 20, 0.4);
    }
    
    .movie-card:hover::before {
        opacity: 1;
    }
    
    .movie-poster {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
        position: relative;
    }
    
    .movie-poster::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 50%;
        background: linear-gradient(transparent, rgba(0, 0, 0, 0.8));
        pointer-events: none;
    }
    
    .movie-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 14px;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    
    .movie-meta {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 18px 0;
    }
    
    .badge {
        padding: 10px 18px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: scale(1.05);
    }
    
    .badge-year {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0.1) 100%);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .badge-rating {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.25) 0%, rgba(245, 158, 11, 0.15) 100%);
        color: #fcd34d;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .badge-similarity {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.25) 0%, rgba(229, 9, 20, 0.1) 100%);
        color: #ff6b6b;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }
    
    .badge-popularity {
        background: linear-gradient(135deg, rgba(251, 146, 60, 0.25) 0%, rgba(251, 146, 60, 0.1) 100%);
        color: #fdba74;
        border: 1px solid rgba(251, 146, 60, 0.3);
    }
    
    .genre-tag {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(34, 211, 238, 0.08) 100%);
        color: #67e8f9;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin: 4px;
        border: 1px solid rgba(34, 211, 238, 0.25);
        transition: all 0.3s ease;
    }
    
    .genre-tag:hover {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.3) 0%, rgba(34, 211, 238, 0.15) 100%);
        transform: translateY(-2px);
    }
    
    .overview-text {
        color: rgba(255, 255, 255, 0.65);
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    .filter-section {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.08) 0%, rgba(229, 9, 20, 0.02) 100%);
        border: 1px solid rgba(229, 9, 20, 0.15);
        border-radius: 24px;
        padding: 30px;
        margin: 25px 0;
        backdrop-filter: blur(15px);
    }
    
    .list-item {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-left: 5px solid #e50914;
        border-radius: 16px;
        padding: 24px;
        margin: 14px 0;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .list-item:hover {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.12) 0%, rgba(229, 9, 20, 0.04) 100%);
        transform: translateX(12px);
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.15);
    }
    
    .list-item h4 {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0 0 10px 0;
    }
    
    .list-item p {
        color: rgba(255, 255, 255, 0.55);
        margin: 0;
        font-size: 0.9rem;
    }
    
    .stat-card {
        background: linear-gradient(145deg, rgba(30, 30, 60, 0.6) 0%, rgba(20, 20, 40, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px 24px;
        text-align: center;
        backdrop-filter: blur(15px);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e50914, #ff6b6b, #e50914);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: 200% 0; }
        50% { background-position: 0 0; }
    }
    
    .stat-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(229, 9, 20, 0.2);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e50914 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.85rem;
        margin-top: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .star-rating {
        color: #fcd34d;
        font-size: 1.1rem;
        letter-spacing: 2px;
    }
    
    .quick-search-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.7);
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 4px;
        display: inline-block;
    }
    
    .quick-search-btn:hover {
        background: rgba(229, 9, 20, 0.2);
        border-color: rgba(229, 9, 20, 0.4);
        color: #fff;
        transform: translateY(-2px);
    }
    
    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .history-item:hover {
        background: rgba(229, 9, 20, 0.1);
        border-color: rgba(229, 9, 20, 0.2);
    }
    
    .surprise-btn {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
    }
    
    .surprise-btn:hover {
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.5) !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff3b3b 0%, #e50914 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 24px rgba(229, 9, 20, 0.35);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(229, 9, 20, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 2px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        color: white !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(229, 9, 20, 0.5) !important;
        box-shadow: 0 0 0 4px rgba(229, 9, 20, 0.1), 0 8px 24px rgba(0, 0, 0, 0.2) !important;
        background: rgba(255, 255, 255, 0.06) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.04);
        border: 2px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        color: white;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #e50914 0%, #ff6b6b 100%);
    }
    
    .note-input {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.9rem !important;
    }
    
    .export-btn {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.3) !important;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(229, 9, 20, 0.5), transparent);
    }
    </style>
""", unsafe_allow_html=True)

if 'text_model' not in st.session_state:
    st.session_state.text_model = None
if 'image_model' not in st.session_state:
    st.session_state.image_model = None
if 'image_processor' not in st.session_state:
    st.session_state.image_processor = None
if 'image_device' not in st.session_state:
    st.session_state.image_device = None
if 'milvus_connected' not in st.session_state:
    st.session_state.milvus_connected = False
if 'text_collection' not in st.session_state:
    st.session_state.text_collection = None
if 'image_collection' not in st.session_state:
    st.session_state.image_collection = None
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'show_filters' not in st.session_state:
    st.session_state.show_filters = False
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'home'
if 'movie_notes' not in st.session_state:
    st.session_state.movie_notes = {}
if 'personal_ratings' not in st.session_state:
    st.session_state.personal_ratings = {}
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Quick search suggestions
QUICK_SEARCHES = [
    "Mind-bending sci-fi thriller",
    "Heartwarming family adventure",
    "Epic fantasy journey",
    "Romantic comedy",
    "Gripping crime drama",
    "Animated masterpiece",
    "Intense action movie",
    "Psychological horror"
]

# Random movie prompts for "Surprise Me"
SURPRISE_PROMPTS = [
    "hidden gem underrated masterpiece",
    "cult classic unique film",
    "critically acclaimed drama",
    "visually stunning cinematography",
    "award winning performance",
    "thought provoking deep meaning",
    "exciting adventure exploration",
    "emotional touching story"
]

@st.cache_resource
def initialize_milvus():
    try:
        if milvus_connect():
            logger.info("Connected to Milvus")
            return True
        return False
    except Exception as e:
        logger.error(f"Milvus connection error: {e}")
        return False

@st.cache_resource
def initialize_text_model():
    try:
        model = load_model()
        logger.info("Text model loaded")
        return model
    except Exception as e:
        logger.error(f"Text model loading error: {e}")
        return None

@st.cache_resource
def initialize_image_model():
    try:
        model, processor, device = load_clip_model()
        logger.info("Image model loaded")
        return model, processor, device
    except Exception as e:
        logger.error(f"Image model loading error: {e}")
        return None, None, None

def add_to_watchlist(movie):
    movie_id = f"{movie['title']}_{movie.get('release_date', '')}"
    existing_ids = [f"{m['title']}_{m.get('release_date', '')}" for m in st.session_state.watchlist]
    if movie_id not in existing_ids:
        movie_copy = movie.copy()
        movie_copy['added_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.watchlist.append(movie_copy)
        return True
    return False

def add_to_favorites(movie):
    movie_id = f"{movie['title']}_{movie.get('release_date', '')}"
    existing_ids = [f"{m['title']}_{m.get('release_date', '')}" for m in st.session_state.favorites]
    if movie_id not in existing_ids:
        movie_copy = movie.copy()
        movie_copy['added_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.favorites.append(movie_copy)
        return True
    return False

def remove_from_watchlist(index):
    if 0 <= index < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(index)

def remove_from_favorites(index):
    if 0 <= index < len(st.session_state.favorites):
        st.session_state.favorites.pop(index)

def add_to_search_history(query):
    """Add search to history with timestamp"""
    if query and query.strip():
        history_item = {
            'query': query.strip(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        # Remove duplicate if exists
        st.session_state.search_history = [
            h for h in st.session_state.search_history 
            if h['query'].lower() != query.lower()
        ]
        st.session_state.search_history.insert(0, history_item)
        # Keep only last 10 searches
        st.session_state.search_history = st.session_state.search_history[:10]

def get_star_rating(rating):
    """Convert numeric rating to star display"""
    if not rating:
        return "☆☆☆☆☆"
    stars = int(float(rating) / 2)
    half_star = (float(rating) / 2) % 1 >= 0.5
    full_stars = "★" * stars
    half = "½" if half_star and stars < 5 else ""
    empty = "☆" * (5 - stars - (1 if half_star else 0))
    return full_stars + half + empty

def export_list_to_json(list_data, list_name):
    """Export watchlist or favorites to JSON"""
    export_data = {
        'exported_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'list_name': list_name,
        'movies': list_data
    }
    return json.dumps(export_data, indent=2)

def get_movie_id(movie):
    """Generate unique movie ID"""
    return f"{movie.get('title', '')}_{movie.get('release_date', '')}"

def save_movie_note(movie_id, note):
    """Save a note for a movie"""
    st.session_state.movie_notes[movie_id] = note

def save_personal_rating(movie_id, rating):
    """Save personal rating for a movie"""
    st.session_state.personal_ratings[movie_id] = rating

def display_movie_card(movie, card_key="", show_actions=True):
    movie_id = get_movie_id(movie)
    with st.container():
        st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("<div class='movie-poster'>", unsafe_allow_html=True)
            if movie.get('poster_url'):
                try:
                    st.image(movie['poster_url'], use_container_width=True)
                except:
                    st.markdown("<div style='font-size: 80px; text-align: center; padding: 40px; background: linear-gradient(135deg, #1a1a3e, #0a0a1a); border-radius: 12px;'>🎬</div>", unsafe_allow_html=True)
            elif movie.get('image_path') and os.path.exists(movie['image_path']):
                try:
                    st.image(movie['image_path'], use_container_width=True)
                except:
                    st.markdown("<div style='font-size: 80px; text-align: center; padding: 40px; background: linear-gradient(135deg, #1a1a3e, #0a0a1a); border-radius: 12px;'>🎬</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 80px; text-align: center; padding: 40px; background: linear-gradient(135deg, #1a1a3e, #0a0a1a); border-radius: 12px;'>🎬</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='movie-title'>{movie.get('title', 'Unknown')}</div>", unsafe_allow_html=True)
            
            # Star rating display
            if movie.get('vote_average'):
                stars = get_star_rating(movie['vote_average'])
                st.markdown(f"<div class='star-rating'>{stars}</div>", unsafe_allow_html=True)
            
            badges_html = "<div class='movie-meta'>"
            if movie.get('release_date'):
                year = movie['release_date'][:4] if len(movie['release_date']) >= 4 else movie['release_date']
                badges_html += f"<span class='badge badge-year'>📅 {year}</span>"
            if movie.get('vote_average'):
                badges_html += f"<span class='badge badge-rating'>⭐ {movie['vote_average']}/10</span>"
            if movie.get('similarity_percent'):
                badges_html += f"<span class='badge badge-similarity'>🎯 {movie['similarity_percent']}</span>"
            if movie.get('popularity'):
                badges_html += f"<span class='badge badge-popularity'>🔥 {movie['popularity']:.0f}</span>"
            badges_html += "</div>"
            st.markdown(badges_html, unsafe_allow_html=True)
            
            if movie.get('genre'):
                genres = movie['genre'].split(',') if ',' in movie['genre'] else [movie['genre']]
                genre_html = "<div style='margin: 15px 0;'>"
                for g in genres[:4]:
                    genre_html += f"<span class='genre-tag'>{g.strip()}</span>"
                genre_html += "</div>"
                st.markdown(genre_html, unsafe_allow_html=True)
            
            if movie.get('overview'):
                with st.expander("📖 Read Synopsis", expanded=False):
                    st.markdown(f"<div class='overview-text'>{movie['overview']}</div>", unsafe_allow_html=True)
            
            # Personal note preview
            if movie_id in st.session_state.movie_notes and st.session_state.movie_notes[movie_id]:
                st.markdown(f"<div style='color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 8px;'>📝 {st.session_state.movie_notes[movie_id][:50]}...</div>", unsafe_allow_html=True)
            
            if show_actions:
                col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])
                with col_a:
                    btn_key = f"watchlist_{card_key}_{hash(str(movie.get('title', '')))}"
                    if st.button("📋 List", key=btn_key, use_container_width=True, help="Add to Watchlist"):
                        if add_to_watchlist(movie):
                            st.toast("✅ Added to watchlist!", icon="📋")
                        else:
                            st.toast("Already in watchlist", icon="ℹ️")
                        st.rerun()
                with col_b:
                    fav_key = f"favorite_{card_key}_{hash(str(movie.get('title', '')))}"
                    if st.button("❤️ Fave", key=fav_key, use_container_width=True, help="Add to Favorites"):
                        if add_to_favorites(movie):
                            st.toast("✅ Added to favorites!", icon="❤️")
                        else:
                            st.toast("Already in favorites", icon="ℹ️")
                        st.rerun()
                with col_c:
                    note_key = f"note_{card_key}_{hash(str(movie.get('title', '')))}"
                    with st.popover("📝 Note"):
                        current_note = st.session_state.movie_notes.get(movie_id, "")
                        new_note = st.text_area("Your notes:", value=current_note, key=f"note_input_{note_key}", height=100)
                        if st.button("Save", key=f"save_note_{note_key}"):
                            save_movie_note(movie_id, new_note)
                            st.toast("Note saved!", icon="📝")
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    if not st.session_state.milvus_connected:
        with st.spinner("🔗 Connecting to database..."):
            st.session_state.milvus_connected = initialize_milvus()
            if not st.session_state.milvus_connected:
                st.error("❌ Failed to connect to database")
                st.stop()
    
    st.markdown("""
        <div class='main-header'>
            <h1 class='logo-text'>🎬 MovieFlix</h1>
            <p class='tagline'>AI-Powered Movie Discovery • Find Your Next Favorite Film</p>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Discover",
        "🔍 Smart Search",
        "🖼️ Visual Search",
        "📋 Watchlist",
        "❤️ Favorites"
    ])
    
    with tab1:
        # Stats dashboard
        col1, col2, col3, col4 = st.columns(4)
        try:
            text_stats = get_collection_stats(config.TEXT_COLLECTION_NAME)
            image_stats = get_collection_stats(config.IMAGE_COLLECTION_NAME)
            with col1:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-number'>{text_stats['num_entities'] if text_stats else 0:,}</div>
                        <div class='stat-label'>Movies in Database</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-number'>{len(st.session_state.watchlist)}</div>
                        <div class='stat-label'>In Watchlist</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-number'>{len(st.session_state.favorites)}</div>
                        <div class='stat-label'>Favorites</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-number'>{image_stats['num_entities'] if image_stats else 0:,}</div>
                        <div class='stat-label'>Movie Posters</div>
                    </div>
                """, unsafe_allow_html=True)
        except:
            pass
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Surprise Me Feature
        st.markdown("<div class='section-title'>🎲 Feeling Lucky?</div>", unsafe_allow_html=True)
        col_surprise, col_info = st.columns([1, 3])
        with col_surprise:
            if st.button("🎰 Surprise Me!", type="primary", use_container_width=True, help="Get random movie recommendations"):
                if st.session_state.text_model is None:
                    with st.spinner("Loading AI model..."):
                        st.session_state.text_model = initialize_text_model()
                if st.session_state.text_collection is None:
                    st.session_state.text_collection = create_text_collection()
                
                if st.session_state.text_model and st.session_state.text_collection:
                    random_prompt = random.choice(SURPRISE_PROMPTS)
                    with st.spinner("🎲 Finding something special..."):
                        try:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                random_prompt,
                                top_k=5, min_rating=6.5
                            )
                            if results:
                                st.session_state['surprise_results'] = results
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        with col_info:
            st.markdown("<p style='color: rgba(255,255,255,0.5); margin-top: 10px;'>Let AI pick random movies based on quality, uniqueness, and hidden gems!</p>", unsafe_allow_html=True)
        
        if 'surprise_results' in st.session_state and st.session_state.surprise_results:
            st.markdown("<br>", unsafe_allow_html=True)
            st.success(f"🎲 Found {len(st.session_state.surprise_results)} surprise picks!")
            for idx, movie in enumerate(st.session_state.surprise_results):
                display_movie_card(movie, card_key=f"surprise_{idx}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Browse by Category
        st.markdown("<div class='section-title'>📚 Browse by Category</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            category = st.selectbox(
                "Choose a category",
                ["🏆 Top Rated Movies", "🔥 Most Popular", "🆕 Recent Releases",
                 "🎞️ Classic Films", "🎭 By Genre"],
                label_visibility="collapsed"
            )
        with col2:
            limit = st.slider("Results", 5, 20, 10, key="discover_limit")
        
        if "By Genre" in category:
            genre = st.selectbox(
                "Select Genre",
                ["Action", "Adventure", "Animation", "Comedy", "Drama",
                 "Horror", "Romance", "Science Fiction", "Thriller", "Documentary", "Mystery"]
            )
        
        if st.button("🎬 Load Movies", type="primary", use_container_width=True):
            if st.session_state.text_model is None:
                with st.spinner("🤖 Loading AI model..."):
                    st.session_state.text_model = initialize_text_model()
            if st.session_state.text_collection is None:
                st.session_state.text_collection = create_text_collection()
            
            if st.session_state.text_model and st.session_state.text_collection:
                with st.spinner("🔍 Loading movies..."):
                    try:
                        if "Top Rated" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                "highly rated acclaimed masterpiece",
                                top_k=limit, min_rating=7.5
                            )
                        elif "Popular" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                "popular trending blockbuster",
                                top_k=limit, min_popularity=100
                            )
                        elif "Recent" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                "new recent latest releases",
                                top_k=limit, min_year=2020
                            )
                        elif "Classic" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                "classic legendary iconic",
                                top_k=limit, max_year=1990, min_rating=7.0
                            )
                        else:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                f"best {genre} movies",
                                top_k=limit, genre_filter=genre
                            )
                        
                        if results:
                            st.success(f"🎉 Found {len(results)} movies!")
                            for idx, movie in enumerate(results):
                                display_movie_card(movie, card_key=f"discover_{idx}")
                        else:
                            st.warning("No movies found in this category")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab2:
        st.markdown("<div class='search-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔍 Smart Movie Search</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: rgba(255,255,255,0.6);'>Describe what you're looking for — our AI understands natural language!</p>", unsafe_allow_html=True)
        
        query = st.text_area(
            "Search",
            placeholder="e.g., 'a gripping thriller with unexpected twists', 'feel-good comedy for family night', 'visually stunning sci-fi adventure'...",
            height=100,
            label_visibility="collapsed"
        )
        
        # Quick search suggestions
        st.markdown("<p style='color: rgba(255,255,255,0.5); font-size: 0.85rem; margin: 15px 0 10px 0;'>💡 Quick searches:</p>", unsafe_allow_html=True)
        quick_cols = st.columns(4)
        for idx, suggestion in enumerate(QUICK_SEARCHES):
            with quick_cols[idx % 4]:
                if st.button(suggestion, key=f"quick_{idx}", use_container_width=True):
                    st.session_state['quick_search'] = suggestion
                    st.rerun()
        
        # Auto-fill from quick search
        if 'quick_search' in st.session_state:
            query = st.session_state.quick_search
            del st.session_state.quick_search
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("⚙️ Advanced Filters", use_container_width=True):
                st.session_state.show_filters = not st.session_state.show_filters
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Search History
        if st.session_state.search_history:
            with st.expander("📜 Recent Searches", expanded=False):
                for hist_idx, hist_item in enumerate(st.session_state.search_history[:5]):
                    col_hist, col_time, col_btn = st.columns([3, 1, 1])
                    with col_hist:
                        st.markdown(f"<div style='color: rgba(255,255,255,0.7);'>{hist_item['query'][:50]}...</div>" if len(hist_item['query']) > 50 else f"<div style='color: rgba(255,255,255,0.7);'>{hist_item['query']}</div>", unsafe_allow_html=True)
                    with col_time:
                        st.markdown(f"<div style='color: rgba(255,255,255,0.4); font-size: 0.8rem;'>{hist_item['timestamp']}</div>", unsafe_allow_html=True)
                    with col_btn:
                        if st.button("Re-search", key=f"hist_{hist_idx}", use_container_width=True):
                            st.session_state['quick_search'] = hist_item['query']
                            st.rerun()
        
        if st.session_state.show_filters:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.markdown("#### 🎚️ Advanced Filters")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                min_year = st.number_input("📅 From Year", 1900, 2030, 1990, 1)
                max_year = st.number_input("📅 To Year", 1900, 2030, 2024, 1)
            with col2:
                min_rating = st.slider("⭐ Min Rating", 0.0, 10.0, 6.0, 0.1)
                max_rating = st.slider("⭐ Max Rating", 0.0, 10.0, 10.0, 0.1)
            with col3:
                min_pop = st.number_input("🔥 Min Popularity", 0.0, 1000.0, 0.0, 10.0)
            with col4:
                genre_filter = st.selectbox("🎭 Genre", ["All", "Action", "Adventure", "Animation", "Comedy", "Drama", "Horror", "Romance", "Science Fiction", "Thriller", "Documentary", "Mystery"])
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            min_year, max_year = 1990, 2024
            min_rating, max_rating = 0.0, 10.0
            min_pop = 0.0
            genre_filter = "All"
        
        col1, col2 = st.columns([1, 3])
        with col1:
            top_k = st.slider("📊 Max Results", 1, 20, 8)
        with col2:
            search_btn = st.button("🔍 Search Movies", type="primary", use_container_width=True)
        
        if search_btn and query:
            add_to_search_history(query)
            
            if st.session_state.text_model is None:
                st.session_state.text_model = initialize_text_model()
            if st.session_state.text_collection is None:
                st.session_state.text_collection = create_text_collection()
            
            if st.session_state.text_model and st.session_state.text_collection:
                with st.spinner("🔍 Searching with AI..."):
                    try:
                        kwargs = {'top_k': top_k}
                        if st.session_state.show_filters:
                            kwargs.update({
                                'min_year': min_year, 'max_year': max_year,
                                'min_rating': min_rating, 'max_rating': max_rating
                            })
                            if min_pop > 0:
                                kwargs['min_popularity'] = min_pop
                            if genre_filter != "All":
                                kwargs['genre_filter'] = genre_filter
                        
                        results = search_similar_movies(
                            st.session_state.text_collection,
                            st.session_state.text_model,
                            query,
                            **kwargs
                        )
                        
                        if results:
                            st.success(f"🎉 Found {len(results)} matches for \"{query[:30]}{'...' if len(query) > 30 else ''}\"")
                            for idx, movie in enumerate(results):
                                display_movie_card(movie, card_key=f"search_{idx}")
                        else:
                            st.warning("🔍 No matches found. Try different keywords!")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab3:
        st.markdown("<div class='section-title'>🖼️ Visual Movie Discovery</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: rgba(255,255,255,0.6);'>Upload a movie poster or any image — our AI will find visually similar films!</p>", unsafe_allow_html=True)
        
        col_upload, col_info = st.columns([2, 1])
        with col_upload:
            uploaded = st.file_uploader(
                "Drop an image here", 
                type=["jpg", "jpeg", "png", "webp"], 
                label_visibility="collapsed",
                help="Upload a movie poster or scene to find similar movies"
            )
        with col_info:
            st.markdown("""
                <div style='background: rgba(255,255,255,0.03); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);'>
                    <p style='color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0;'>
                        <strong>💡 Tips:</strong><br>
                        • Upload movie posters for best results<br>
                        • Higher quality images = better matches<br>
                        • Works with screenshots too!
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        if uploaded:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("<div style='background: rgba(255,255,255,0.03); border-radius: 20px; padding: 20px;'>", unsafe_allow_html=True)
                img = Image.open(uploaded)
                st.image(img, use_container_width=True, caption="📷 Your uploaded image")
                
                img_k = st.slider("🎯 Number of results", 1, 20, 6, key="img_k")
                
                search_visual = st.button("🔎 Find Similar Movies", type="primary", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                if search_visual:
                    if st.session_state.image_model is None:
                        with st.spinner("🤖 Loading visual AI model..."):
                            model, proc, dev = initialize_image_model()
                            st.session_state.image_model = model
                            st.session_state.image_processor = proc
                            st.session_state.image_device = dev
                    
                    if st.session_state.image_collection is None:
                        st.session_state.image_collection = create_image_collection()
                    
                    if st.session_state.image_model and st.session_state.image_collection:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            img.save(tmp.name)
                            path = tmp.name
                        
                        with st.spinner("🖼️ Analyzing image and finding matches..."):
                            try:
                                results = search_similar_images(
                                    st.session_state.image_collection,
                                    st.session_state.image_model,
                                    st.session_state.image_processor,
                                    st.session_state.image_device,
                                    path, top_k=img_k
                                )
                                
                                if results:
                                    st.success(f"🎉 Found {len(results)} visually similar movies!")
                                    for idx, m in enumerate(results):
                                        display_movie_card(m, card_key=f"img_{idx}")
                                else:
                                    st.warning("🔍 No visual matches found. Try a different image!")
                            except Exception as e:
                                st.error(f"Error: {e}")
                            finally:
                                if os.path.exists(path):
                                    os.remove(path)
                else:
                    st.markdown("""
                        <div style='text-align: center; padding: 60px 20px; color: rgba(255,255,255,0.4);'>
                            <p style='font-size: 3rem; margin-bottom: 15px;'>🎬</p>
                            <p>Click "Find Similar Movies" to start visual search</p>
                        </div>
                    """, unsafe_allow_html=True)

    with tab4:
        st.markdown("<div class='section-title'>📋 My Watchlist</div>", unsafe_allow_html=True)
        
        if st.session_state.watchlist:
            # Stats and actions row
            col_stats, col_export, col_clear = st.columns([3, 1, 1])
            with col_stats:
                st.markdown(f"""
                    <p style='color: rgba(255,255,255,0.7);'>
                        📊 <strong>{len(st.session_state.watchlist)}</strong> movies to watch 
                        • Estimated watch time: ~<strong>{len(st.session_state.watchlist) * 2}</strong> hours
                    </p>
                """, unsafe_allow_html=True)
            with col_export:
                export_data = export_list_to_json(st.session_state.watchlist, "Watchlist")
                st.download_button(
                    label="📥 Export",
                    data=export_data,
                    file_name="movieflix_watchlist.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_clear:
                if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                    st.session_state.watchlist = []
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            for idx, movie in enumerate(st.session_state.watchlist):
                movie_id = get_movie_id(movie)
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    stars = get_star_rating(movie.get('vote_average'))
                    note_preview = ""
                    if movie_id in st.session_state.movie_notes and st.session_state.movie_notes[movie_id]:
                        note_preview = f"<br><span style='color: rgba(255,255,255,0.4); font-size: 0.8rem;'>📝 {st.session_state.movie_notes[movie_id][:40]}...</span>"
                    
                    st.markdown(f"""
                        <div class='list-item'>
                            <h4>{movie.get('title', 'Unknown')} <span style='color: #fcd34d; font-size: 0.9rem;'>{stars}</span></h4>
                            <p>
                                📅 {movie.get('release_date', 'N/A')[:4] if movie.get('release_date') else 'N/A'} • 
                                ⭐ {movie.get('vote_average', 'N/A')}/10 • 
                                🎭 {movie.get('genre', 'N/A')[:30] if movie.get('genre') else 'N/A'}
                                {note_preview}
                            </p>
                            <p style='font-size: 0.75rem; color: rgba(255,255,255,0.3); margin-top: 8px;'>
                                Added: {movie.get('added_date', 'Unknown')}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div style='display: flex; flex-direction: column; gap: 8px; padding-top: 10px;'>", unsafe_allow_html=True)
                    if st.button("❌", key=f"rm_w_{idx}", use_container_width=True, help="Remove from watchlist"):
                        remove_from_watchlist(idx)
                        st.rerun()
                    if st.button("❤️", key=f"move_fav_{idx}", use_container_width=True, help="Move to favorites"):
                        if add_to_favorites(movie):
                            remove_from_watchlist(idx)
                            st.toast("Moved to favorites!", icon="❤️")
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='text-align: center; padding: 80px 20px; color: rgba(255,255,255,0.4);'>
                    <p style='font-size: 4rem; margin-bottom: 20px;'>📋</p>
                    <p style='font-size: 1.2rem;'>Your watchlist is empty</p>
                    <p style='font-size: 0.9rem; margin-top: 10px;'>Go to Discover or Search to add movies!</p>
                </div>
            """, unsafe_allow_html=True)

    with tab5:
        st.markdown("<div class='section-title'>❤️ My Favorite Movies</div>", unsafe_allow_html=True)
        
        if st.session_state.favorites:
            # Stats and actions row
            col_stats, col_export, col_clear = st.columns([3, 1, 1])
            with col_stats:
                avg_rating = sum(float(m.get('vote_average', 0)) for m in st.session_state.favorites) / len(st.session_state.favorites)
                st.markdown(f"""
                    <p style='color: rgba(255,255,255,0.7);'>
                        💎 <strong>{len(st.session_state.favorites)}</strong> favorite movies 
                        • Average rating: <strong>{avg_rating:.1f}</strong>/10
                    </p>
                """, unsafe_allow_html=True)
            with col_export:
                export_data = export_list_to_json(st.session_state.favorites, "Favorites")
                st.download_button(
                    label="📥 Export",
                    data=export_data,
                    file_name="movieflix_favorites.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_clear:
                if st.button("🗑️ Clear All", type="secondary", key="clear_fav", use_container_width=True):
                    st.session_state.favorites = []
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            for idx, movie in enumerate(st.session_state.favorites):
                movie_id = get_movie_id(movie)
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    stars = get_star_rating(movie.get('vote_average'))
                    note_preview = ""
                    if movie_id in st.session_state.movie_notes and st.session_state.movie_notes[movie_id]:
                        note_preview = f"<br><span style='color: rgba(255,255,255,0.4); font-size: 0.8rem;'>📝 {st.session_state.movie_notes[movie_id][:40]}...</span>"
                    
                    st.markdown(f"""
                        <div class='list-item' style='border-left-color: #ec4899;'>
                            <h4>❤️ {movie.get('title', 'Unknown')} <span style='color: #fcd34d; font-size: 0.9rem;'>{stars}</span></h4>
                            <p>
                                📅 {movie.get('release_date', 'N/A')[:4] if movie.get('release_date') else 'N/A'} • 
                                ⭐ {movie.get('vote_average', 'N/A')}/10 • 
                                🎭 {movie.get('genre', 'N/A')[:30] if movie.get('genre') else 'N/A'}
                                {note_preview}
                            </p>
                            <p style='font-size: 0.75rem; color: rgba(255,255,255,0.3); margin-top: 8px;'>
                                Added: {movie.get('added_date', 'Unknown')}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div style='display: flex; flex-direction: column; gap: 8px; padding-top: 10px;'>", unsafe_allow_html=True)
                    if st.button("❌", key=f"rm_f_{idx}", use_container_width=True, help="Remove from favorites"):
                        remove_from_favorites(idx)
                        st.rerun()
                    with st.popover("📝"):
                        current_note = st.session_state.movie_notes.get(movie_id, "")
                        new_note = st.text_area("Your notes:", value=current_note, key=f"fav_note_{idx}", height=80)
                        if st.button("Save", key=f"save_fav_note_{idx}"):
                            save_movie_note(movie_id, new_note)
                            st.toast("Note saved!", icon="📝")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='text-align: center; padding: 80px 20px; color: rgba(255,255,255,0.4);'>
                    <p style='font-size: 4rem; margin-bottom: 20px;'>❤️</p>
                    <p style='font-size: 1.2rem;'>No favorites yet</p>
                    <p style='font-size: 0.9rem; margin-top: 10px;'>Mark movies as favorites to save them here!</p>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='
            text-align: center; 
            color: rgba(255,255,255,0.4); 
            padding: 50px 20px; 
            font-size: 0.9rem;
            background: linear-gradient(180deg, transparent 0%, rgba(229, 9, 20, 0.05) 100%);
            border-top: 1px solid rgba(255,255,255,0.05);
            margin-top: 40px;
        '>
            <p style='font-size: 1.8rem; font-weight: 800; margin-bottom: 15px; background: linear-gradient(135deg, #e50914, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🎬 MovieFlix</p>
            <p style='color: rgba(255,255,255,0.5);'>Powered by AI Vector Search • Built with Streamlit & Milvus</p>
            <p style='font-size: 0.8rem; margin-top: 15px; color: rgba(255,255,255,0.3);'>
                Discover • Explore • Enjoy • Your perfect movie is just a search away
            </p>
            <div style='margin-top: 20px; display: flex; justify-content: center; gap: 20px;'>
                <span style='color: rgba(255,255,255,0.4);'>🤖 AI-Powered</span>
                <span style='color: rgba(255,255,255,0.4);'>🔍 Semantic Search</span>
                <span style='color: rgba(255,255,255,0.4);'>🖼️ Visual Discovery</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()