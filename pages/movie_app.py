import streamlit as st
from PIL import Image
import tempfile
import os
import sys
import pandas as pd
from pathlib import Path
import logging

# Setup paths
utils_dir = Path(__file__).resolve().parent.parent / 'utils'
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

try:
    import config
    from text_embedder import load_model, embed_text
    from image_embedder import load_clip_model, embed_image
    from milvus_vectordb import (
        milvus_connect, milvus_disconnect, 
        create_text_collection, create_image_collection,
        search_similar_movies, search_similar_images,
        get_collection_stats
    )
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MovieFinder - Your Movie Recommendation Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
    }
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .movie-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
    }
    .movie-info {
        font-size: 0.9rem;
        color: #666;
    }
    .similarity-badge {
        background-color: #FF4B4B;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .filter-section {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
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

# Initialize connections and models
@st.cache_resource
def initialize_milvus():
    """Initialize Milvus connection"""
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
    """Initialize text embedding model"""
    try:
        model = load_model()
        logger.info("Text model loaded")
        return model
    except Exception as e:
        logger.error(f"Text model loading error: {e}")
        return None

@st.cache_resource
def initialize_image_model():
    """Initialize image embedding model"""
    try:
        model, processor, device = load_clip_model()
        logger.info("Image model loaded")
        return model, processor, device
    except Exception as e:
        logger.error(f"Image model loading error: {e}")
        return None, None, None

def display_movie_card(movie, show_poster=True):
    """Display a movie card with all information"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if show_poster and movie.get('poster_url'):
                try:
                    st.image(movie['poster_url'], width=150)
                except:
                    st.write("🎬")
            elif show_poster and movie.get('image_path') and os.path.exists(movie['image_path']):
                try:
                    st.image(movie['image_path'], width=150)
                except:
                    st.write("🎬")
            else:
                st.write("🎬")
        
        with col2:
            st.markdown(f"<div class='movie-title'>{movie.get('title', 'Unknown')}</div>", unsafe_allow_html=True)
            
            # Movie metadata
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if movie.get('release_date'):
                    year = movie['release_date'][:4] if len(movie['release_date']) >= 4 else movie['release_date']
                    st.write(f"📅 **Year:** {year}")
            with col_b:
                if movie.get('vote_average'):
                    st.write(f"⭐ **Rating:** {movie['vote_average']}/10")
            with col_c:
                if movie.get('similarity_percent'):
                    st.markdown(f"<span class='similarity-badge'>{movie['similarity_percent']}</span>", unsafe_allow_html=True)
            
            # Genre
            if movie.get('genre'):
                st.write(f"🎭 **Genre:** {movie['genre']}")
            
            # Overview
            if movie.get('overview'):
                with st.expander("📖 Overview"):
                    st.write(movie['overview'])
            
            # Additional info
            col_x, col_y = st.columns(2)
            with col_x:
                if movie.get('popularity'):
                    st.write(f"📊 **Popularity:** {movie['popularity']:.1f}")
            with col_y:
                if movie.get('vote_count'):
                    st.write(f"🗳️ **Votes:** {movie['vote_count']}")
        
        st.markdown("---")

def main():
    # Header
    st.markdown("<div class='main-header'>🎬 MovieFinder</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Discover your next favorite movie</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Initialize connections
        if not st.session_state.milvus_connected:
            with st.spinner("Connecting to database..."):
                st.session_state.milvus_connected = initialize_milvus()
                if st.session_state.milvus_connected:
                    st.success("✅ Connected to database")
                else:
                    st.error("❌ Failed to connect to database")
                    st.stop()
        
        # Check collections
        try:
            text_stats = get_collection_stats(config.TEXT_COLLECTION_NAME)
            image_stats = get_collection_stats(config.IMAGE_COLLECTION_NAME)
            
            if text_stats:
                st.info(f"📚 Text DB: {text_stats['num_entities']} movies")
            if image_stats:
                st.info(f"🖼️ Image DB: {image_stats['num_entities']} posters")
        except:
            pass
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        MovieFinder helps you discover movies through:
        - 🔍 Text-based search
        - 🖼️ Image similarity
        - 🎯 Advanced filters
        - ⭐ Rating-based recommendations
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search by Text", "🖼️ Search by Image", "🎯 Advanced Filters", "📊 Browse All"])
    
    # Tab 1: Text Search
    with tab1:
        st.header("Search Movies by Description")
        st.write("Describe the type of movie you're looking for")
        
        # Search input
        query_text = st.text_area(
            "What are you looking for?",
            placeholder="e.g., 'action-packed superhero movie', 'romantic comedy with witty dialogue', 'dark thriller'",
            height=100
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            top_k = st.slider("Number of results", 1, 20, 5)
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if not query_text.strip():
                st.warning("Please enter a search query")
            else:
                # Load text model
                if st.session_state.text_model is None:
                    with st.spinner("Loading text model..."):
                        st.session_state.text_model = initialize_text_model()
                
                if st.session_state.text_model is None:
                    st.error("Failed to load text model")
                else:
                    # Load collection
                    if st.session_state.text_collection is None:
                        try:
                            st.session_state.text_collection = create_text_collection()
                        except Exception as e:
                            st.error(f"Failed to load collection: {e}")
                    
                    if st.session_state.text_collection:
                        with st.spinner("Searching..."):
                            try:
                                results = search_similar_movies(
                                    st.session_state.text_collection,
                                    st.session_state.text_model,
                                    query_text,
                                    top_k=top_k
                                )
                                
                                if results:
                                    st.success(f"Found {len(results)} matching movies")
                                    for movie in results:
                                        display_movie_card(movie)
                                else:
                                    st.warning("No movies found matching your query")
                            except Exception as e:
                                st.error(f"Search error: {e}")
    
    # Tab 2: Image Search
    with tab2:
        st.header("Search Movies by Poster Image")
        st.write("Upload a movie poster to find similar movies")
        
        uploaded_file = st.file_uploader(
            "Choose a movie poster",
            type=["jpg", "jpeg", "png", "webp", "bmp", "gif"]
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            img_top_k = st.slider("Number of results", 1, 20, 5, key="img_topk")
        
        if uploaded_file is not None:
            col_img1, col_img2 = st.columns([1, 2])
            with col_img1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Poster", use_container_width=True)
            
            if st.button("🔍 Find Similar", type="primary", use_container_width=True):
                # Load image model
                if st.session_state.image_model is None:
                    with st.spinner("Loading image model..."):
                        model, processor, device = initialize_image_model()
                        st.session_state.image_model = model
                        st.session_state.image_processor = processor
                        st.session_state.image_device = device
                
                if st.session_state.image_model is None:
                    st.error("Failed to load image model")
                else:
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        image.save(tmp_file.name)
                        temp_image_path = tmp_file.name
                    
                    # Load collection
                    if st.session_state.image_collection is None:
                        try:
                            st.session_state.image_collection = create_image_collection()
                        except Exception as e:
                            st.error(f"Failed to load collection: {e}")
                    
                    if st.session_state.image_collection:
                        with st.spinner("Searching for similar posters..."):
                            try:
                                results = search_similar_images(
                                    st.session_state.image_collection,
                                    st.session_state.image_model,
                                    st.session_state.image_processor,
                                    st.session_state.image_device,
                                    temp_image_path,
                                    top_k=img_top_k
                                )
                                
                                if results:
                                    st.success(f"Found {len(results)} similar movies")
                                    for movie in results:
                                        display_movie_card(movie, show_poster=True)
                                else:
                                    st.warning("No similar movies found")
                            except Exception as e:
                                st.error(f"Search error: {e}")
                            finally:
                                os.remove(temp_image_path)
    
    # Tab 3: Advanced Filters
    with tab3:
        st.header("Advanced Search with Filters")
        st.write("Combine text search with advanced filters")
        
        # Query input
        adv_query = st.text_area(
            "Search query",
            placeholder="e.g., 'science fiction adventure'",
            height=80,
            key="adv_query"
        )
        
        # Filters section
        st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
        st.subheader("🎯 Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Year Range**")
            min_year = st.number_input("From", min_value=1900, max_value=2030, value=1990, step=1)
            max_year = st.number_input("To", min_value=1900, max_value=2030, value=2024, step=1)
        
        with col2:
            st.write("**Rating Range**")
            min_rating = st.slider("Minimum Rating", 0.0, 10.0, 6.0, 0.1)
            max_rating = st.slider("Maximum Rating", 0.0, 10.0, 10.0, 0.1)
        
        with col3:
            st.write("**Popularity**")
            min_popularity = st.number_input("Minimum Popularity", min_value=0.0, value=0.0, step=10.0)
        
        # Genre filter
        genre_options = [
            "All", "Action", "Adventure", "Animation", "Comedy", "Crime", 
            "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
            "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"
        ]
        genre_filter = st.selectbox("Genre", genre_options)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            adv_top_k = st.slider("Number of results", 1, 20, 5, key="adv_topk")
        
        if st.button("🔍 Search with Filters", type="primary", use_container_width=True):
            if not adv_query.strip():
                st.warning("Please enter a search query")
            else:
                # Load text model
                if st.session_state.text_model is None:
                    with st.spinner("Loading model..."):
                        st.session_state.text_model = initialize_text_model()
                
                if st.session_state.text_model is None:
                    st.error("Failed to load model")
                else:
                    # Load collection
                    if st.session_state.text_collection is None:
                        try:
                            st.session_state.text_collection = create_text_collection()
                        except Exception as e:
                            st.error(f"Failed to load collection: {e}")
                    
                    if st.session_state.text_collection:
                        with st.spinner("Searching with filters..."):
                            try:
                                # Prepare filters
                                kwargs = {
                                    'top_k': adv_top_k,
                                    'min_year': min_year,
                                    'max_year': max_year,
                                    'min_rating': min_rating,
                                    'max_rating': max_rating,
                                    'min_popularity': min_popularity if min_popularity > 0 else None,
                                }
                                
                                if genre_filter != "All":
                                    kwargs['genre_filter'] = genre_filter
                                
                                results = search_similar_movies(
                                    st.session_state.text_collection,
                                    st.session_state.text_model,
                                    adv_query,
                                    **kwargs
                                )
                                
                                if results:
                                    st.success(f"Found {len(results)} movies matching your criteria")
                                    for movie in results:
                                        display_movie_card(movie)
                                else:
                                    st.warning("No movies found matching your criteria. Try adjusting the filters.")
                            except Exception as e:
                                st.error(f"Search error: {e}")
    
    # Tab 4: Browse All
    with tab4:
        st.header("Browse Movies Database")
        st.write("Explore all movies in the database")
        
        # Category-based browsing
        browse_category = st.selectbox(
            "Browse by category",
            ["Top Rated", "Most Popular", "Recent Releases", "Classic Movies", "By Genre"]
        )
        
        if browse_category == "By Genre":
            browse_genre = st.selectbox(
                "Select Genre",
                ["Action", "Adventure", "Animation", "Comedy", "Crime", 
                 "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
                 "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"]
            )
        
        browse_limit = st.slider("Number of movies to show", 5, 50, 10, key="browse_limit")
        
        if st.button("📊 Browse", type="primary", use_container_width=True):
            # Load text model
            if st.session_state.text_model is None:
                with st.spinner("Loading model..."):
                    st.session_state.text_model = initialize_text_model()
            
            if st.session_state.text_model is None:
                st.error("Failed to load model")
            else:
                # Load collection
                if st.session_state.text_collection is None:
                    try:
                        st.session_state.text_collection = create_text_collection()
                    except Exception as e:
                        st.error(f"Failed to load collection: {e}")
                
                if st.session_state.text_collection:
                    with st.spinner("Loading movies..."):
                        try:
                            # Create query based on category
                            if browse_category == "Top Rated":
                                results = search_similar_movies(
                                    st.session_state.text_collection,
                                    st.session_state.text_model,
                                    "highly rated popular movies",
                                    top_k=browse_limit,
                                    min_rating=7.5
                                )
                            elif browse_category == "Most Popular":
                                results = search_similar_movies(
                                    st.session_state.text_collection,
                                    st.session_state.text_model,
                                    "popular trending movies",
                                    top_k=browse_limit,
                                    min_popularity=100
                                )
                            elif browse_category == "Recent Releases":
                                results = search_similar_movies(
                                    st.session_state.text_collection,
                                    st.session_state.text_model,
                                    "new recent movies",
                                    top_k=browse_limit,
                                    min_year=2020
                                )
                            elif browse_category == "Classic Movies":
                                results = search_similar_movies(
                                    st.session_state.text_collection,
                                    st.session_state.text_model,
                                    "classic legendary movies",
                                    top_k=browse_limit,
                                    max_year=1990,
                                    min_rating=7.0
                                )
                            elif browse_category == "By Genre":
                                results = search_similar_movies(
                                    st.session_state.text_collection,
                                    st.session_state.text_model,
                                    f"{browse_genre} movies",
                                    top_k=browse_limit,
                                    genre_filter=browse_genre
                                )
                            
                            if results:
                                st.success(f"Showing {len(results)} movies")
                                for movie in results:
                                    display_movie_card(movie)
                            else:
                                st.warning("No movies found in this category")
                        except Exception as e:
                            st.error(f"Browse error: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎬 MovieFinder - Powered by AI Vector Search</p>
        <p style='font-size: 0.8rem;'>Find your next favorite movie using advanced semantic search</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()