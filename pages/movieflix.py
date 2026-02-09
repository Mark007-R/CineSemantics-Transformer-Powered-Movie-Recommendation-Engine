import streamlit as st
from PIL import Image
import tempfile
import os
import sys
import pandas as pd
from pathlib import Path
import logging
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
    page_title="MovieFlix - Discover Movies",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 24px;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(255, 107, 107, 0.2) 100%);
        color: #fff;
        border-bottom: 3px solid #FF4B4B;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .logo-text {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 50%, #FFD93D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .tagline {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.2rem;
        margin-top: 10px;
    }
    
    /* Search container */
    .search-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 75, 75, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Movie card - horizontal card design */
    .movie-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .movie-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(255, 75, 75, 0.3);
        border: 1px solid rgba(255, 75, 75, 0.4);
    }
    
    .movie-poster {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .movie-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        line-height: 1.3;
    }
    
    .movie-meta {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        margin: 15px 0;
    }
    
    .badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-year {
        background: linear-gradient(135deg, rgba(100, 100, 255, 0.3) 0%, rgba(100, 100, 255, 0.1) 100%);
        color: #a0a0ff;
        border: 1px solid rgba(100, 100, 255, 0.3);
    }
    
    .badge-rating {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.3) 0%, rgba(255, 165, 0, 0.2) 100%);
        color: #FFD700;
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    .badge-similarity {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.3) 0%, rgba(255, 107, 107, 0.2) 100%);
        color: #FF6B6B;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    
    .badge-popularity {
        background: linear-gradient(135deg, rgba(255, 100, 100, 0.3) 0%, rgba(255, 150, 100, 0.2) 100%);
        color: #FF9A76;
        border: 1px solid rgba(255, 100, 100, 0.3);
    }
    
    .genre-tag {
        background: linear-gradient(135deg, rgba(75, 192, 192, 0.3) 0%, rgba(75, 192, 192, 0.1) 100%);
        color: #4BCFCF;
        padding: 6px 14px;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 4px;
        border: 1px solid rgba(75, 192, 192, 0.3);
    }
    
    .overview-text {
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Filter section */
    .filter-section {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 107, 107, 0.05) 100%);
        border: 2px solid rgba(255, 75, 75, 0.2);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Watchlist/Favorites items */
    .list-item {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
        border-left: 4px solid #FF4B4B;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .list-item:hover {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.15) 0%, rgba(255, 75, 75, 0.05) 100%);
        transform: translateX(8px);
    }
    
    .list-item h4 {
        color: #ffffff;
        font-size: 1.3rem;
        margin: 0 0 8px 0;
    }
    
    .list-item p {
        color: rgba(255, 255, 255, 0.6);
        margin: 0;
    }
    
    /* Category buttons */
    .category-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.8);
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .category-btn:hover {
        background: rgba(255, 75, 75, 0.2);
        border-color: rgba(255, 75, 75, 0.5);
        color: #fff;
    }
    
    /* Stats display */
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    /* Browse grid */
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(48, 43, 99, 0.3) 100%);
        border-radius: 25px;
        padding: 40px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8B8B 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    
    /* Input fields */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(255, 75, 75, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.1) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
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
        st.session_state.watchlist.append(movie.copy())
        return True
    return False

def add_to_favorites(movie):
    movie_id = f"{movie['title']}_{movie.get('release_date', '')}"
    existing_ids = [f"{m['title']}_{m.get('release_date', '')}" for m in st.session_state.favorites]
    if movie_id not in existing_ids:
        st.session_state.favorites.append(movie.copy())
        return True
    return False

def remove_from_watchlist(index):
    if 0 <= index < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(index)

def remove_from_favorites(index):
    if 0 <= index < len(st.session_state.favorites):
        st.session_state.favorites.pop(index)

def display_movie_card(movie, card_key=""):
    with st.container():
        st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("<div class='movie-poster'>", unsafe_allow_html=True)
            if movie.get('poster_url'):
                try:
                    st.image(movie['poster_url'], use_container_width=True)
                except:
                    st.markdown("<div style='font-size: 100px; text-align: center; padding: 40px;'></div>", unsafe_allow_html=True)
            elif movie.get('image_path') and os.path.exists(movie['image_path']):
                try:
                    st.image(movie['image_path'], use_container_width=True)
                except:
                    st.markdown("<div style='font-size: 100px; text-align: center; padding: 40px;'></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 100px; text-align: center; padding: 40px;'></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='movie-title'>{movie.get('title', 'Unknown')}</div>", unsafe_allow_html=True)
            badges_html = "<div class='movie-meta'>"
            if movie.get('release_date'):
                year = movie['release_date'][:4] if len(movie['release_date']) >= 4 else movie['release_date']
                badges_html += f"<span class='badge badge-year'>{year}</span>"
            if movie.get('vote_average'):
                badges_html += f"<span class='badge badge-rating'>{movie['vote_average']}/10</span>"
            if movie.get('similarity_percent'):
                badges_html += f"<span class='badge badge-similarity'>{movie['similarity_percent']}</span>"
            if movie.get('popularity'):
                badges_html += f"<span class='badge badge-popularity'>{movie['popularity']:.0f}</span>"
            badges_html += "</div>"
            st.markdown(badges_html, unsafe_allow_html=True)
            if movie.get('genre'):
                genres = movie['genre'].split(',') if ',' in movie['genre'] else [movie['genre']]
                genre_html = "<div style='margin: 15px 0;'>"
                for g in genres[:3]:
                    genre_html += f"<span class='genre-tag'>{g.strip()}</span>"
                genre_html += "</div>"
                st.markdown(genre_html, unsafe_allow_html=True)
            if movie.get('overview'):
                with st.expander("Read Synopsis"):
                    st.markdown(f"<div class='overview-text'>{movie['overview']}</div>", unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns([1, 1, 3])
            with col_a:
                btn_key = f"watchlist_{card_key}_{hash(str(movie.get('title', '')))}"
                if st.button("Watchlist", key=btn_key, use_container_width=True):
                    if add_to_watchlist(movie):
                        st.toast("Added to watchlist!", icon="")
                    else:
                        st.toast("Already in watchlist", icon="")
                    st.rerun()
            with col_b:
                fav_key = f"favorite_{card_key}_{hash(str(movie.get('title', '')))}"
                if st.button("Favorite", key=fav_key, use_container_width=True):
                    if add_to_favorites(movie):
                        st.toast("Added to favorites!", icon="")
                    else:
                        st.toast("Already in favorites", icon="")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    if not st.session_state.milvus_connected:
        with st.spinner("Connecting to database..."):
            st.session_state.milvus_connected = initialize_milvus()
            if not st.session_state.milvus_connected:
                st.error("Failed to connect to database")
                st.stop()
    st.markdown("""
        <div class='main-header'>
            <h1 class='logo-text'>MovieFlix</h1>
            <p class='tagline'>Discover Your Next Favorite Movie</p>
        </div>
    """, unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Discover",
        "Search",
        "Visual Search",
        "My Watchlist",
        "Favorites"
    ])
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        try:
            text_stats = get_collection_stats(config.TEXT_COLLECTION_NAME)
            image_stats = get_collection_stats(config.IMAGE_COLLECTION_NAME)
            with col1:
                st.markdown(f"""
                    <div class='stat-card'>
                        <div class='stat-number'>{text_stats['num_entities'] if text_stats else 0}</div>
                        <div class='stat-label'>Movies Available</div>
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
                        <div class='stat-number'>{image_stats['num_entities'] if image_stats else 0}</div>
                        <div class='stat-label'>Posters</div>
                    </div>
                """, unsafe_allow_html=True)
        except:
            pass
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Browse by Category")
        col1, col2 = st.columns([3, 1])
        with col1:
            category = st.selectbox(
                "Choose a category",
                ["Top Rated Movies", "Most Popular", "Recent Releases",
                 "Classic Films", "By Genre"],
                label_visibility="collapsed"
            )
        with col2:
            limit = st.slider("Movies", 5, 20, 10, key="discover_limit")
        if category == "By Genre":
            genre = st.selectbox(
                "Select Genre",
                ["Action", "Adventure", "Animation", "Comedy", "Drama",
                 "Horror", "Romance", "Science Fiction", "Thriller"]
            )
        if st.button("Load Movies", type="primary", use_container_width=True):
            if st.session_state.text_model is None:
                with st.spinner("Loading AI model..."):
                    st.session_state.text_model = initialize_text_model()
            if st.session_state.text_collection is None:
                st.session_state.text_collection = create_text_collection()
            if st.session_state.text_model and st.session_state.text_collection:
                with st.spinner("Loading movies..."):
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
                            st.success(f"Found {len(results)} movies!")
                            for idx, movie in enumerate(results):
                                display_movie_card(movie, card_key=f"discover_{idx}")
                        else:
                            st.warning("No movies found in this category")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab2:
        st.markdown("<div class='search-box'>", unsafe_allow_html=True)
        st.markdown("### Smart Movie Search")
        st.markdown("<p style='color: rgba(255,255,255,0.6);'>Describe what you're looking for and let AI find the perfect match</p>", unsafe_allow_html=True)
        query = st.text_area(
            "Search",
            placeholder="e.g., 'mind-bending sci-fi thriller', 'heartwarming family drama', 'epic adventure'",
            height=100,
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Show Advanced Filters", use_container_width=True):
                st.session_state.show_filters = not st.session_state.show_filters
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state.show_filters:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.markdown("#### Advanced Filters")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                min_year = st.number_input("From Year", 1900, 2030, 1990, 1)
                max_year = st.number_input("To Year", 1900, 2030, 2024, 1)
            with col2:
                min_rating = st.slider("Min Rating", 0.0, 10.0, 6.0, 0.1)
                max_rating = st.slider("Max Rating", 0.0, 10.0, 10.0, 0.1)
            with col3:
                min_pop = st.number_input("Min Popularity", 0.0, 1000.0, 0.0, 10.0)
            with col4:
                genre_filter = st.selectbox("Genre", ["All", "Action", "Adventure", "Animation", "Comedy", "Drama", "Horror", "Romance", "Science Fiction", "Thriller"])
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            min_year, max_year = 1990, 2024
            min_rating, max_rating = 0.0, 10.0
            min_pop = 0.0
            genre_filter = "All"
        col1, col2 = st.columns([1, 3])
        with col1:
            top_k = st.slider("Results", 1, 20, 5)
        with col2:
            search_btn = st.button("Search Movies", type="primary", use_container_width=True)
        if search_btn and query:
            if st.session_state.text_model is None:
                st.session_state.text_model = initialize_text_model()
            if st.session_state.text_collection is None:
                st.session_state.text_collection = create_text_collection()
            if st.session_state.text_model and st.session_state.text_collection:
                with st.spinner("Searching..."):
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
                            st.success(f"Found {len(results)} matches!")
                            for idx, movie in enumerate(results):
                                display_movie_card(movie, card_key=f"search_{idx}")
                        else:
                            st.warning("No matches found")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab3:
        st.markdown("### Visual Movie Discovery")
        st.markdown("<p style='color: rgba(255,255,255,0.6);'>Upload a movie poster to find visually similar films</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop poster here", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if uploaded:
            col1, col2 = st.columns([1, 2])
            with col1:
                img = Image.open(uploaded)
                st.image(img, use_container_width=True)
                img_k = st.slider("Results", 1, 20, 5, key="img_k")
                if st.button("Find Similar", type="primary", use_container_width=True):
                    if st.session_state.image_model is None:
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
                        with st.spinner("Analyzing..."):
                            try:
                                results = search_similar_images(
                                    st.session_state.image_collection,
                                    st.session_state.image_model,
                                    st.session_state.image_processor,
                                    st.session_state.image_device,
                                    path, top_k=img_k
                                )
                                if results:
                                    with col2:
                                        st.success(f"{len(results)} similar movies")
                                        for idx, m in enumerate(results):
                                            display_movie_card(m, card_key=f"img_{idx}")
                                else:
                                    st.warning("No matches")
                            except Exception as e:
                                st.error(f"Error: {e}")
                            finally:
                                if os.path.exists(path):
                                    os.remove(path)

    with tab4:
        st.markdown("### My Watchlist")
        if st.session_state.watchlist:
            st.markdown(f"<p style='color: rgba(255,255,255,0.7);'>{len(st.session_state.watchlist)} movies to watch</p>", unsafe_allow_html=True)
            for idx, movie in enumerate(st.session_state.watchlist):
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(f"""
                        <div class='list-item'>
                            <h4>{movie.get('title', 'Unknown')}</h4>
                            <p>{movie.get('release_date', 'N/A')[:4]} • {movie.get('vote_average', 'N/A')}/10 • {movie.get('genre', 'N/A')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("Remove", key=f"rm_w_{idx}", use_container_width=True):
                        remove_from_watchlist(idx)
                        st.rerun()
            if st.button("Clear All", type="secondary"):
                st.session_state.watchlist = []
                st.rerun()
        else:
            st.info("Your watchlist is empty")

    with tab5:
        st.markdown("### My Favorite Movies")
        if st.session_state.favorites:
            st.markdown(f"<p style='color: rgba(255,255,255,0.7);'>{len(st.session_state.favorites)} favorite movies</p>", unsafe_allow_html=True)
            for idx, movie in enumerate(st.session_state.favorites):
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(f"""
                        <div class='list-item'>
                            <h4>{movie.get('title', 'Unknown')}</h4>
                            <p>{movie.get('release_date', 'N/A')[:4]} • {movie.get('vote_average', 'N/A')}/10 • {movie.get('genre', 'N/A')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("Remove", key=f"rm_f_{idx}", use_container_width=True):
                        remove_from_favorites(idx)
                        st.rerun()
            if st.button("Clear All", type="secondary"):
                st.session_state.favorites = []
                st.rerun()
        else:
            st.info("No favorites yet")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: rgba(255,255,255,0.4); padding: 40px 20px; font-size: 0.9rem;'>
            <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 10px;'>MovieFlix</p>
            <p>Powered by AI Vector Search • Built with Streamlit</p>
            <p style='font-size: 0.8rem; margin-top: 10px;'>Discover, explore, and enjoy movies like never before</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()