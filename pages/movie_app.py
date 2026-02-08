import streamlit as st
from PIL import Image
import tempfile
import os
import sys
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime

# Setup paths
utils_dir = Path(__file__).resolve().parent.parent / 'utils'
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

try:
    import config
    from text_embedder import load_model
    from image_embedder import load_clip_model
    from milvus_vectordb import (
        milvus_connect, milvus_disconnect, 
        create_text_collection, create_image_collection,
        search_similar_movies, search_similar_images,
        get_collection_stats
    )
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MovieFinder Pro - Discover Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .movie-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .movie-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1a1a1a;
        margin-bottom: 10px;
    }
    .similarity-badge {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .rating-badge {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .genre-tag {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    .filter-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border: 2px solid #d0d0d0;
    }
    .stat-box {
        background-color: rgba(255, 75, 75, 0.1);
        border: 2px solid #FF4B4B;
        border-left: 4px solid #FF4B4B;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stat-box h4 {
        color: #FF4B4B;
        margin: 0 0 8px 0;
    }
    .stat-box p {
        color: inherit;
        margin: 0;
    }
    .watchlist-item {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .favorite-item {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
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
    """Add movie to watchlist"""
    # Create a unique identifier for the movie
    movie_id = f"{movie['title']}_{movie.get('release_date', '')}"
    existing_ids = [f"{m['title']}_{m.get('release_date', '')}" for m in st.session_state.watchlist]
    
    if movie_id not in existing_ids:
        st.session_state.watchlist.append(movie.copy())
        return True
    return False

def add_to_favorites(movie):
    """Add movie to favorites"""
    movie_id = f"{movie['title']}_{movie.get('release_date', '')}"
    existing_ids = [f"{m['title']}_{m.get('release_date', '')}" for m in st.session_state.favorites]
    
    if movie_id not in existing_ids:
        st.session_state.favorites.append(movie.copy())
        return True
    return False

def remove_from_watchlist(index):
    """Remove movie from watchlist by index"""
    if 0 <= index < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(index)

def remove_from_favorites(index):
    """Remove movie from favorites by index"""
    if 0 <= index < len(st.session_state.favorites):
        st.session_state.favorites.pop(index)

def add_to_search_history(query, results_count):
    """Add search to history"""
    st.session_state.search_history.insert(0, {
        'query': query,
        'results': results_count,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    st.session_state.search_history = st.session_state.search_history[:10]

def display_movie_card(movie, show_actions=True, card_key=""):
    """Display a movie card with all information"""
    with st.container():
        st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if movie.get('poster_url'):
                try:
                    st.image(movie['poster_url'], width=150)
                except:
                    st.write("🎬")
            elif movie.get('image_path') and os.path.exists(movie['image_path']):
                try:
                    st.image(movie['image_path'], width=150)
                except:
                    st.write("🎬")
            else:
                st.markdown("<div style='font-size: 80px; text-align: center;'>🎬</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='movie-title'>{movie.get('title', 'Unknown')}</div>", unsafe_allow_html=True)
            
            # Badges row
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                if movie.get('release_date'):
                    year = movie['release_date'][:4] if len(movie['release_date']) >= 4 else movie['release_date']
                    st.write(f"📅 {year}")
            with col_b:
                if movie.get('vote_average'):
                    st.markdown(f"<span class='rating-badge'>⭐ {movie['vote_average']}/10</span>", unsafe_allow_html=True)
            with col_c:
                if movie.get('similarity_percent'):
                    st.markdown(f"<span class='similarity-badge'>🎯 {movie['similarity_percent']}</span>", unsafe_allow_html=True)
            with col_d:
                if movie.get('popularity'):
                    st.write(f"🔥 {movie['popularity']:.0f}")
            
            # Genre tags
            if movie.get('genre'):
                genres = movie['genre'].split(',') if ',' in movie['genre'] else [movie['genre']]
                genre_html = "".join([f"<span class='genre-tag'>{g.strip()}</span>" for g in genres[:3]])
                st.markdown(genre_html, unsafe_allow_html=True)
            
            # Overview
            if movie.get('overview'):
                with st.expander("📖 Synopsis"):
                    st.write(movie['overview'])
            
            # Action buttons
            if show_actions:
                col_x, col_y, col_z = st.columns([1, 1, 2])
                with col_x:
                    # Generate unique key for button
                    btn_key = f"watchlist_{card_key}_{movie.get('title', '')}_{hash(str(movie.get('overview', ''))[:50])}"
                    if st.button("➕ Watchlist", key=btn_key, use_container_width=True):
                        if add_to_watchlist(movie):
                            st.success("✅ Added to watchlist!", icon="✅")
                        else:
                            st.info("Already in watchlist", icon="ℹ️")
                        st.rerun()
                
                with col_y:
                    # Generate unique key for button
                    fav_key = f"favorite_{card_key}_{movie.get('title', '')}_{hash(str(movie.get('overview', ''))[:50])}"
                    if st.button("❤️ Favorite", key=fav_key, use_container_width=True):
                        if add_to_favorites(movie):
                            st.success("❤️ Added to favorites!", icon="❤️")
                        else:
                            st.info("Already in favorites", icon="ℹ️")
                        st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("<div class='main-header'>🎬 MovieFinder Pro</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Your AI-Powered Movie Discovery Platform</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Dashboard")
        
        # Initialize connections
        if not st.session_state.milvus_connected:
            with st.spinner("Connecting to database..."):
                st.session_state.milvus_connected = initialize_milvus()
                if st.session_state.milvus_connected:
                    st.success("✅ Database Connected")
                else:
                    st.error("❌ Database Connection Failed")
                    st.stop()
        
        # Database stats
        st.markdown("---")
        st.subheader("💾 Database Info")
        try:
            text_stats = get_collection_stats(config.TEXT_COLLECTION_NAME)
            image_stats = get_collection_stats(config.IMAGE_COLLECTION_NAME)
            
            if text_stats:
                st.metric(
                    label="📚 Text Database",
                    value=f"{text_stats['num_entities']} movies"
                )
            
            if image_stats:
                st.metric(
                    label="🖼️ Image Database", 
                    value=f"{image_stats['num_entities']} posters"
                )
        except:
            st.info("Database stats unavailable")
        
        # User stats
        st.markdown("---")
        st.subheader("📊 Your Stats")
        st.metric("Watchlist", len(st.session_state.watchlist))
        st.metric("Favorites", len(st.session_state.favorites))
        st.metric("Searches", len(st.session_state.search_history))
        
        # About
        st.markdown("---")
        with st.expander("ℹ️ About MovieFinder"):
            st.markdown("""
            **Features:**
            - 🔍 Semantic text search
            - 🖼️ Visual similarity search
            - 🎯 Advanced filtering
            - ⭐ Watchlist & favorites
            - 📊 Browse by category
            
            **Powered by:**
            - Milvus Vector DB
            - CLIP & Transformers
            - Streamlit
            """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Search Movies", 
        "🖼️ Image Search", 
        "📊 Browse", 
        "📋 Watchlist", 
        "❤️ Favorites"
    ])
    
    # Tab 1: Unified Search with Filters
    with tab1:
        # Main search area
        st.subheader("🔍 Search Movies")
        query_text = st.text_area(
            "What kind of movie are you looking for?",
            placeholder="e.g., 'mind-bending sci-fi with time travel', 'heartwarming family drama', 'intense action thriller'",
            height=100,
            key="main_search"
        )
        
        # Toggle filters button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("🎯 Filters", use_container_width=True):
                st.session_state.show_filters = not st.session_state.show_filters
        
        # Collapsible filters section
        if st.session_state.show_filters:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.subheader("🎛️ Advanced Filters")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write("**📅 Year Range**")
                min_year = st.number_input("From", 1900, 2030, 1990, 1, key="filter_min_year")
                max_year = st.number_input("To", 1900, 2030, 2024, 1, key="filter_max_year")
            
            with col2:
                st.write("**⭐ Rating Range**")
                min_rating = st.slider("Min Rating", 0.0, 10.0, 6.0, 0.1, key="filter_min_rating")
                max_rating = st.slider("Max Rating", 0.0, 10.0, 10.0, 0.1, key="filter_max_rating")
            
            with col3:
                st.write("**🔥 Popularity**")
                min_pop = st.number_input("Minimum", 0.0, 1000.0, 0.0, 10.0, key="filter_min_pop")
            
            with col4:
                st.write("**🎭 Genre**")
                genres = ["All", "Action", "Adventure", "Animation", "Comedy", "Crime", 
                         "Documentary", "Drama", "Family", "Fantasy", "Horror", "Romance", 
                         "Science Fiction", "Thriller"]
                genre = st.selectbox("Select Genre", genres, key="filter_genre")
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Default values when filters are hidden
            min_year = 1990
            max_year = 2024
            min_rating = 0.0
            max_rating = 10.0
            min_pop = 0.0
            genre = "All"
        
        # Number of results
        col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
        with col_res1:
            top_k = st.slider("Results", 1, 20, 5, key="search_top_k")
        
        # Search button
        with col_res2:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Execute search
        if search_clicked:
            if not query_text.strip():
                st.warning("⚠️ Please enter a search query")
            else:
                # Initialize models
                if st.session_state.text_model is None:
                    with st.spinner("Loading AI model..."):
                        st.session_state.text_model = initialize_text_model()
                
                if st.session_state.text_collection is None:
                    try:
                        st.session_state.text_collection = create_text_collection()
                    except Exception as e:
                        st.error(f"Database error: {e}")
                
                if st.session_state.text_model and st.session_state.text_collection:
                    with st.spinner("🔍 Searching through thousands of movies..."):
                        try:
                            # Build search parameters
                            kwargs = {'top_k': top_k}
                            
                            # Apply filters if shown
                            if st.session_state.show_filters:
                                kwargs['min_year'] = min_year
                                kwargs['max_year'] = max_year
                                kwargs['min_rating'] = min_rating
                                kwargs['max_rating'] = max_rating
                                if min_pop > 0:
                                    kwargs['min_popularity'] = min_pop
                                if genre != "All":
                                    kwargs['genre_filter'] = genre
                            
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                query_text,
                                **kwargs
                            )
                            
                            if results:
                                add_to_search_history(query_text, len(results))
                                st.success(f"🎉 Found {len(results)} matching movies!")
                                
                                # Display results
                                for idx, movie in enumerate(results):
                                    display_movie_card(movie, show_actions=True, card_key=f"search_{idx}")
                            else:
                                st.warning("No matches found. Try different keywords or relax the filters.")
                        except Exception as e:
                            st.error(f"Search error: {e}")
    
    # Tab 2: Image Search
    with tab2:
        st.header("🖼️ Visual Movie Discovery")
        st.write("Upload a poster and find visually similar movies")
        
        uploaded_file = st.file_uploader(
            "Drop a movie poster here",
            type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
            key="image_uploader"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Poster", use_container_width=True)
                img_top_k = st.slider("Number of Results", 1, 20, 5, key="image_top_k")
                
                if st.button("🔍 Find Similar Movies", type="primary", use_container_width=True):
                    # Initialize models
                    if st.session_state.image_model is None:
                        with st.spinner("Loading vision model..."):
                            model, processor, device = initialize_image_model()
                            st.session_state.image_model = model
                            st.session_state.image_processor = processor
                            st.session_state.image_device = device
                    
                    if st.session_state.image_collection is None:
                        try:
                            st.session_state.image_collection = create_image_collection()
                        except Exception as e:
                            st.error(f"Database error: {e}")
                    
                    if st.session_state.image_model and st.session_state.image_collection:
                        # Save uploaded image temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            image.save(tmp_file.name)
                            temp_path = tmp_file.name
                        
                        with st.spinner("🎨 Analyzing visual patterns..."):
                            try:
                                results = search_similar_images(
                                    st.session_state.image_collection,
                                    st.session_state.image_model,
                                    st.session_state.image_processor,
                                    st.session_state.image_device,
                                    temp_path,
                                    top_k=img_top_k
                                )
                                
                                if results:
                                    with col2:
                                        st.success(f"🎨 Found {len(results)} visually similar movies")
                                        for idx, movie in enumerate(results):
                                            display_movie_card(movie, show_actions=True, card_key=f"image_{idx}")
                                else:
                                    st.warning("No similar posters found")
                            except Exception as e:
                                st.error(f"Error: {e}")
                            finally:
                                # Clean up temporary file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
    
    # Tab 3: Browse Categories
    with tab3:
        st.header("📊 Browse Movies by Category")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            categories = [
                "🏆 Top Rated Movies", 
                "🔥 Most Popular", 
                "🆕 Recent Releases", 
                "🎞️ Classic Films", 
                "🎭 By Genre"
            ]
            category = st.selectbox("Choose Category", categories, key="browse_category")
        
        with col2:
            browse_limit = st.slider("Number of Movies", 5, 30, 10, key="browse_limit")
        
        # Genre selection for "By Genre" category
        if category == "🎭 By Genre":
            browse_genre = st.selectbox(
                "Select Genre", 
                ["Action", "Adventure", "Animation", "Comedy", "Drama", 
                 "Horror", "Romance", "Science Fiction", "Thriller"],
                key="browse_genre"
            )
        
        if st.button("📊 Load Movies", type="primary", use_container_width=True):
            # Initialize models
            if st.session_state.text_model is None:
                with st.spinner("Loading model..."):
                    st.session_state.text_model = initialize_text_model()
            
            if st.session_state.text_collection is None:
                try:
                    st.session_state.text_collection = create_text_collection()
                except Exception as e:
                    st.error(f"Database error: {e}")
            
            if st.session_state.text_model and st.session_state.text_collection:
                with st.spinner("Loading movies..."):
                    try:
                        # Execute search based on category
                        if "Top Rated" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                "highly rated acclaimed movies masterpiece",
                                top_k=browse_limit,
                                min_rating=7.5
                            )
                        elif "Popular" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                "popular trending blockbuster movies",
                                top_k=browse_limit,
                                min_popularity=100
                            )
                        elif "Recent" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                "new recent latest movies releases",
                                top_k=browse_limit,
                                min_year=2020
                            )
                        elif "Classic" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                "classic legendary iconic films",
                                top_k=browse_limit,
                                max_year=1990,
                                min_rating=7.0
                            )
                        elif "By Genre" in category:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                f"best {browse_genre} movies",
                                top_k=browse_limit,
                                genre_filter=browse_genre
                            )
                        
                        if results:
                            st.success(f"📊 Showing {len(results)} movies")
                            for idx, movie in enumerate(results):
                                display_movie_card(movie, show_actions=True, card_key=f"browse_{idx}")
                        else:
                            st.warning("No movies found in this category")
                    except Exception as e:
                        st.error(f"Error loading movies: {e}")
    
    # Tab 4: Watchlist
    with tab4:
        st.header("📋 My Watchlist")
        
        if st.session_state.watchlist:
            st.write(f"**{len(st.session_state.watchlist)} movies** to watch")
            
            # Display watchlist items
            for idx, movie in enumerate(st.session_state.watchlist):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"""
                    <div class='watchlist-item'>
                        <h4>🎬 {movie.get('title', 'Unknown')}</h4>
                        <p>📅 {movie.get('release_date', 'N/A')[:4]} • ⭐ {movie.get('vote_average', 'N/A')}/10 • 🎭 {movie.get('genre', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("❌ Remove", key=f"remove_watch_{idx}", use_container_width=True):
                        remove_from_watchlist(idx)
                        st.rerun()
            
            # Clear all button
            if st.button("🗑️ Clear All Watchlist", type="secondary"):
                st.session_state.watchlist = []
                st.rerun()
        else:
            st.info("📋 Your watchlist is empty. Start adding movies from the search results!")
    
    # Tab 5: Favorites
    with tab5:
        st.header("❤️ My Favorite Movies")
        
        if st.session_state.favorites:
            st.write(f"**{len(st.session_state.favorites)} favorite movies**")
            
            # Display favorite items
            for idx, movie in enumerate(st.session_state.favorites):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"""
                    <div class='favorite-item'>
                        <h4>❤️ {movie.get('title', 'Unknown')}</h4>
                        <p>📅 {movie.get('release_date', 'N/A')[:4]} • ⭐ {movie.get('vote_average', 'N/A')}/10 • 🎭 {movie.get('genre', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("❌ Remove", key=f"remove_fav_{idx}", use_container_width=True):
                        remove_from_favorites(idx)
                        st.rerun()
            
            # Clear all button
            if st.button("🗑️ Clear All Favorites", type="secondary"):
                st.session_state.favorites = []
                st.rerun()
        else:
            st.info("❤️ No favorites yet. Start liking movies from the search results!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 30px;'>
        <h3>🎬 MovieFinder Pro</h3>
        <p>Discover, explore, and enjoy movies like never before</p>
        <p style='font-size: 0.9rem;'>Powered by AI Vector Search • Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()