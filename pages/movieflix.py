import streamlit as st
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path
import logging
import random
import atexit

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

from helpers import (
    QUICK_SEARCHES, SURPRISE_PROMPTS, GENRES, CATEGORIES, TAB_NAMES,
    SUPPORTED_IMAGE_TYPES, DEFAULT_DISCOVER_LIMIT,
    DEFAULT_IMAGE_RESULTS, DEFAULT_MIN_YEAR, DEFAULT_MAX_YEAR,
    init_session_state, get_movie_id, get_star_rating, export_list_to_json,
    add_to_watchlist, add_to_favorites, remove_from_watchlist, remove_from_favorites,
    add_to_search_history, save_movie_note, save_personal_rating, get_note_preview,
    calculate_watch_time, calculate_average_rating, get_category_search_params,
    build_search_kwargs
)

atexit.register(milvus_disconnect)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="MovieFlix - AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

init_session_state(st.session_state)

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

def display_movie_card(movie, card_key="", show_actions=True):
    movie_id = get_movie_id(movie)
    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("<div class='movie-poster'>", unsafe_allow_html=True)
        if movie.get('poster_url'):
            try:
                st.image(movie['poster_url'], use_container_width=True)
            except:
                st.markdown("<div style='font-size: 80px; text-align: center; padding: 40px; background: linear-gradient(135deg, #1a1a3e, #0a0a1a); border-radius: 12px;'>MOVIE</div>", unsafe_allow_html=True)
        elif movie.get('image_path') and os.path.exists(movie['image_path']):
            try:
                st.image(movie['image_path'], use_container_width=True)
            except:
                st.markdown("<div style='font-size: 80px; text-align: center; padding: 40px; background: linear-gradient(135deg, #1a1a3e, #0a0a1a); border-radius: 12px;'>MOVIE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 80px; text-align: center; padding: 40px; background: linear-gradient(135deg, #1a1a3e, #0a0a1a); border-radius: 12px;'>MOVIE</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='movie-title'>{movie.get('title', 'Unknown')}</div>", unsafe_allow_html=True)
        
        if movie.get('vote_average'):
            stars = get_star_rating(movie['vote_average'])
            st.markdown(f"<div class='star-rating'>{stars}</div>", unsafe_allow_html=True)
        
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
            for g in genres[:4]:
                genre_html += f"<span class='genre-tag'>{g.strip()}</span>"
            genre_html += "</div>"
            st.markdown(genre_html, unsafe_allow_html=True)
        
        if movie.get('overview'):
            st.markdown(f"<div class='overview-text' style='margin-top: 15px;'>{movie['overview']}</div>", unsafe_allow_html=True)
        
        if movie_id in st.session_state.movie_notes and st.session_state.movie_notes[movie_id]:
            st.markdown(f"<div style='color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 8px;'>{st.session_state.movie_notes[movie_id][:50]}...</div>", unsafe_allow_html=True)
        
        if show_actions:
            col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])
            with col_a:
                btn_key = f"watchlist_{card_key}_{hash(str(movie.get('title', '')))}"
                if st.button("List", key=btn_key, use_container_width=True, help="Add to Watchlist"):
                    if add_to_watchlist(st.session_state, movie):
                        st.toast("Added to watchlist!")
                    else:
                        st.toast("Already in watchlist")
                    st.rerun()
            with col_b:
                fav_key = f"favorite_{card_key}_{hash(str(movie.get('title', '')))}"
                if st.button("Fav", key=fav_key, use_container_width=True, help="Add to Favorites"):
                    if add_to_favorites(st.session_state, movie):
                        st.toast("Added to favorites!")
                    else:
                        st.toast("Already in favorites")
                    st.rerun()
            with col_c:
                with st.popover("Note"):
                    current_note = st.session_state.movie_notes.get(movie_id, "")
                    new_note = st.text_area("Your notes:", value=current_note, key=f"note_{card_key}", height=100, max_chars=200)
                    if st.button("Save Note", key=f"save_note_{card_key}"):
                        save_movie_note(st.session_state, movie_id, new_note)
                        st.toast("Note saved!")
                        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.markdown("""
        <div class='main-header'>
            <h1 class='logo-text'>MovieFlix</h1>
            <p class='tagline'>AI-Powered Movie Discovery Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.milvus_connected:
        st.session_state.milvus_connected = initialize_milvus()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_NAMES)
    
    with tab1:
        st.markdown("<div class='section-title'>Your Stats</div>", unsafe_allow_html=True)
        
        try:
            text_stats = get_collection_stats(create_text_collection())
            image_stats = get_collection_stats(create_image_collection())
            
            col1, col2, col3, col4 = st.columns(4)
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
        
        st.markdown("<div class='section-title'>Feeling Lucky?</div>", unsafe_allow_html=True)
        col_surprise, col_info = st.columns([1, 3])
        with col_surprise:
            if st.button("Surprise Me!", type="primary", use_container_width=True, help="Get random movie recommendations"):
                if st.session_state.text_model is None:
                    with st.spinner("Loading AI model..."):
                        st.session_state.text_model = initialize_text_model()
                if st.session_state.text_collection is None:
                    st.session_state.text_collection = create_text_collection()
                
                if st.session_state.text_model and st.session_state.text_collection:
                    random_prompt = random.choice(SURPRISE_PROMPTS)
                    random_seed = random.randint(0, 1000000)
                    with st.spinner("Finding something special..."):
                        try:
                            results = search_similar_movies(
                                st.session_state.text_collection,
                                st.session_state.text_model,
                                random_prompt,
                                top_k=5, min_rating=6.5
                            )
                            if results:
                                random.shuffle(results)
                                st.session_state['surprise_results'] = results
                                st.session_state['surprise_seed'] = random_seed
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        with col_info:
            st.markdown("<p style='color: rgba(255,255,255,0.5); margin-top: 10px;'>Let AI pick random movies based on quality, uniqueness, and hidden gems!</p>", unsafe_allow_html=True)
        
        if 'surprise_results' in st.session_state and st.session_state.surprise_results:
            st.markdown("<br>", unsafe_allow_html=True)
            for idx, movie in enumerate(st.session_state.surprise_results):
                display_movie_card(movie, card_key=f"surprise_{idx}_{st.session_state.get('surprise_seed', 0)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>Browse by Category</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            category = st.selectbox(
                "Choose a category",
                CATEGORIES,
                label_visibility="collapsed"
            )
        with col2:
            limit = st.slider("Results", 5, 20, DEFAULT_DISCOVER_LIMIT, key="discover_limit")
        
        if "By Genre" in category:
            genre = st.selectbox(
                "Select Genre",
                GENRES
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
        st.markdown("<div class='section-title'>Smart Movie Search</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: rgba(255,255,255,0.6);'>Describe what you're looking for - our AI understands natural language!</p>", unsafe_allow_html=True)
        
        query = st.text_area(
            "Search",
            placeholder="e.g., 'a gripping thriller with unexpected twists', 'feel-good comedy for family night', 'visually stunning sci-fi adventure'...",
            height=100,
            label_visibility="collapsed"
        )
        
        st.markdown("<p style='color: rgba(255,255,255,0.5); font-size: 0.85rem; margin: 15px 0 10px 0;'>Quick searches:</p>", unsafe_allow_html=True)
        quick_cols = st.columns(4)
        for idx, qs in enumerate(QUICK_SEARCHES):
            with quick_cols[idx % 4]:
                if st.button(qs, key=f"quick_{idx}", use_container_width=True):
                    st.session_state['quick_search_query'] = qs
                    st.rerun()
        
        if 'quick_search_query' in st.session_state:
            query = st.session_state['quick_search_query']
            del st.session_state['quick_search_query']
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_filters, col_search = st.columns([1, 4])
        with col_filters:
            if st.button("Filters", use_container_width=True):
                st.session_state.show_filters = not st.session_state.show_filters
        with col_search:
            col_btn, col_results = st.columns([1, 1])
            with col_btn:
                search_clicked = st.button("Search", type="primary", use_container_width=True)
            with col_results:
                top_k = st.slider("Results", 3, 20, 10, key="search_results", label_visibility="collapsed")
        
        if st.session_state.show_filters:
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            st.markdown("### Advanced Filters")
            col1, col2 = st.columns(2)
            with col1:
                min_year = st.slider("Min Year", 1900, 2024, DEFAULT_MIN_YEAR)
                min_rating = st.slider("Min Rating", 0.0, 10.0, 0.0, 0.5)
                min_pop = st.slider("Min Popularity", 0, 500, 0, 10)
            with col2:
                max_year = st.slider("Max Year", 1900, 2024, DEFAULT_MAX_YEAR)
                max_rating = st.slider("Max Rating", 0.0, 10.0, 10.0, 0.5)
                genre_filter = st.selectbox("Genre Filter", ["All"] + GENRES)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            min_year = DEFAULT_MIN_YEAR
            max_year = DEFAULT_MAX_YEAR
            min_rating = 0.0
            max_rating = 10.0
            min_pop = 0
            genre_filter = "All"
        
        if search_clicked and query:
            if st.session_state.text_model is None:
                with st.spinner("Loading AI model..."):
                    st.session_state.text_model = initialize_text_model()
            if st.session_state.text_collection is None:
                st.session_state.text_collection = create_text_collection()
            
            if st.session_state.text_model and st.session_state.text_collection:
                add_to_search_history(st.session_state, query)
                
                kwargs = build_search_kwargs(
                    top_k, st.session_state.show_filters,
                    min_year, max_year, min_rating, max_rating,
                    min_pop, genre_filter
                )
                
                with st.spinner("Searching..."):
                    try:
                        results = search_similar_movies(
                            st.session_state.text_collection,
                            st.session_state.text_model,
                            query,
                            **kwargs
                        )
                        
                        if results:
                            st.success(f"Found {len(results)} movies!")
                            for idx, movie in enumerate(results):
                                display_movie_card(movie, card_key=f"search_{idx}")
                        else:
                            st.warning("No movies found. Try adjusting your filters or search query.")
                    except Exception as e:
                        st.error(f"Search error: {e}")
        
        if st.session_state.search_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Recent Searches</div>", unsafe_allow_html=True)
            for idx, item in enumerate(st.session_state.search_history[:5]):
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(f"🔍 {item['query']}", key=f"hist_{idx}", use_container_width=True):
                        st.session_state['quick_search_query'] = item['query']
                        st.rerun()
                with col2:
                    st.markdown(f"<p style='color: rgba(255,255,255,0.3); font-size: 0.75rem; text-align: right; padding-top: 8px;'>{item['timestamp'].split()[1]}</p>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='search-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Visual Movie Search</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: rgba(255,255,255,0.6);'>Upload a movie poster or scene to find visually similar movies!</p>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=SUPPORTED_IMAGE_TYPES,
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            num_results = st.slider("Number of results", 3, 10, DEFAULT_IMAGE_RESULTS)
        with col2:
            search_image_btn = st.button("🔍 Search by Image", type="primary", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if search_image_btn and uploaded_file:
            if st.session_state.image_model is None:
                with st.spinner("Loading image model..."):
                    result = initialize_image_model()
                    if result:
                        st.session_state.image_model, st.session_state.image_processor, st.session_state.image_device = result
            
            if st.session_state.image_collection is None:
                st.session_state.image_collection = create_image_collection()
            
            if st.session_state.image_model and st.session_state.image_collection:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                    
                    with st.spinner("Finding visually similar movies..."):
                        results = search_similar_images(
                            st.session_state.image_collection,
                            st.session_state.image_model,
                            st.session_state.image_processor,
                            st.session_state.image_device,
                            tmp_path,
                            top_k=num_results
                        )
                        
                        if results:
                            st.success(f"Found {len(results)} visually similar movies!")
                            for idx, movie in enumerate(results):
                                display_movie_card(movie, card_key=f"image_{idx}")
                        else:
                            st.warning("No similar movies found")
                except Exception as e:
                    st.error(f"Image search error: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    with tab4:
        st.markdown("<div class='section-title'>My Watchlist</div>", unsafe_allow_html=True)
        
        if st.session_state.watchlist:
            col_stats, col_export, col_clear = st.columns([3, 1, 1])
            with col_stats:
                watch_time = calculate_watch_time(st.session_state.watchlist)
                avg_rating = calculate_average_rating(st.session_state.watchlist)
                st.markdown(f"""
                    <p style='color: rgba(255,255,255,0.7);'>
                        <strong>{len(st.session_state.watchlist)}</strong> movies 
                        - Estimated watch time: <strong>{watch_time}h</strong> 
                        - Average rating: <strong>{avg_rating:.1f}</strong>/10
                    </p>
                """, unsafe_allow_html=True)
            with col_export:
                export_data = export_list_to_json(st.session_state.watchlist, "Watchlist")
                st.download_button(
                    label="Export",
                    data=export_data,
                    file_name="movieflix_watchlist.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_clear:
                if st.button("Clear All", type="secondary", use_container_width=True):
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
                        note_preview = f"<br><span style='color: rgba(255,255,255,0.4); font-size: 0.8rem;'>{st.session_state.movie_notes[movie_id][:40]}...</span>"
                    
                    st.markdown(f"""
                        <div class='list-item'>
                            <h4>{movie.get('title', 'Unknown')} <span style='color: #fcd34d; font-size: 0.9rem;'>{stars}</span></h4>
                            <p>
                                {movie.get('release_date', 'N/A')[:4] if movie.get('release_date') else 'N/A'} | 
                                {movie.get('vote_average', 'N/A')}/10 | 
                                {movie.get('genre', 'N/A')[:30] if movie.get('genre') else 'N/A'}
                                {note_preview}
                            </p>
                            <p style='font-size: 0.75rem; color: rgba(255,255,255,0.3); margin-top: 8px;'>
                                Added: {movie.get('added_date', 'Unknown')}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div style='display: flex; flex-direction: column; gap: 8px; padding-top: 10px;'>", unsafe_allow_html=True)
                    if st.button("X", key=f"rm_w_{idx}", use_container_width=True, help="Remove from watchlist"):
                        remove_from_watchlist(st.session_state, idx)
                        st.rerun()
                    if st.button("Fav", key=f"move_fav_{idx}", use_container_width=True, help="Move to favorites"):
                        if add_to_favorites(st.session_state, movie):
                            remove_from_watchlist(st.session_state, idx)
                            st.toast("Moved to favorites!")
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='text-align: center; padding: 80px 20px; color: rgba(255,255,255,0.4);'>
                    <p style='font-size: 4rem; margin-bottom: 20px;'>LIST</p>
                    <p style='font-size: 1.2rem;'>Your watchlist is empty</p>
                    <p style='font-size: 0.9rem; margin-top: 10px;'>Go to Discover or Search to add movies!</p>
                </div>
            """, unsafe_allow_html=True)

    with tab5:
        st.markdown("<div class='section-title'>My Favorite Movies</div>", unsafe_allow_html=True)
        
        if st.session_state.favorites:
            col_stats, col_export, col_clear = st.columns([3, 1, 1])
            with col_stats:
                avg_rating = calculate_average_rating(st.session_state.favorites)
                st.markdown(f"""
                    <p style='color: rgba(255,255,255,0.7);'>
                        <strong>{len(st.session_state.favorites)}</strong> favorite movies 
                        - Average rating: <strong>{avg_rating:.1f}</strong>/10
                    </p>
                """, unsafe_allow_html=True)
            with col_export:
                export_data = export_list_to_json(st.session_state.favorites, "Favorites")
                st.download_button(
                    label="Export",
                    data=export_data,
                    file_name="movieflix_favorites.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_clear:
                if st.button("Clear All", type="secondary", key="clear_fav", use_container_width=True):
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
                        note_preview = f"<br><span style='color: rgba(255,255,255,0.4); font-size: 0.8rem;'>{st.session_state.movie_notes[movie_id][:40]}...</span>"
                    
                    st.markdown(f"""
                        <div class='list-item' style='border-left-color: #ec4899;'>
                            <h4>{movie.get('title', 'Unknown')} <span style='color: #fcd34d; font-size: 0.9rem;'>{stars}</span></h4>
                            <p>
                                {movie.get('release_date', 'N/A')[:4] if movie.get('release_date') else 'N/A'} | 
                                {movie.get('vote_average', 'N/A')}/10 | 
                                {movie.get('genre', 'N/A')[:30] if movie.get('genre') else 'N/A'}
                                {note_preview}
                            </p>
                            <p style='font-size: 0.75rem; color: rgba(255,255,255,0.3); margin-top: 8px;'>
                                Added: {movie.get('added_date', 'Unknown')}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div style='display: flex; flex-direction: column; gap: 8px; padding-top: 10px;'>", unsafe_allow_html=True)
                    if st.button("X", key=f"rm_f_{idx}", use_container_width=True, help="Remove from favorites"):
                        remove_from_favorites(st.session_state, idx)
                        st.rerun()
                    with st.popover("Note"):
                        current_note = st.session_state.movie_notes.get(movie_id, "")
                        new_note = st.text_area("Your notes:", value=current_note, key=f"fav_note_{idx}", height=80)
                        if st.button("Save", key=f"save_fav_note_{idx}"):
                            save_movie_note(st.session_state, movie_id, new_note)
                            st.toast("Note saved!")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='text-align: center; padding: 80px 20px; color: rgba(255,255,255,0.4);'>
                    <p style='font-size: 4rem; margin-bottom: 20px;'>FAVS</p>
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
            <p style='font-size: 1.8rem; font-weight: 800; margin-bottom: 15px; background: linear-gradient(135deg, #e50914, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>MovieFlix</p>
            <p style='color: rgba(255,255,255,0.5);'>Powered by AI Vector Search - Built with Streamlit & Milvus</p>
            <p style='font-size: 0.8rem; margin-top: 15px; color: rgba(255,255,255,0.3);'>
                Discover - Explore - Enjoy - Your perfect movie is just a search away
            </p>
            <div style='margin-top: 20px; display: flex; justify-content: center; gap: 20px;'>
                <span style='color: rgba(255,255,255,0.4);'>AI-Powered</span>
                <span style='color: rgba(255,255,255,0.4);'>Semantic Search</span>
                <span style='color: rgba(255,255,255,0.4);'>Visual Discovery</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()