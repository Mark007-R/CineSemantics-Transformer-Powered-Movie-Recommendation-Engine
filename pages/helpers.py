import json
from datetime import datetime

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

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Drama",
    "Horror", "Romance", "Science Fiction", "Thriller", 
    "Documentary", "Mystery"
]

CATEGORIES = [
    "Top Rated Movies", "Most Popular", "Recent Releases",
    "Classic Films", "By Genre"
]

TAB_NAMES = [
    "Discover",
    "Smart Search", 
    "Visual Search",
    "Watchlist",
    "Favorites"
]

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]

MAX_SEARCH_HISTORY = 10
MAX_GENRES_DISPLAY = 4
DEFAULT_TOP_K = 8
DEFAULT_DISCOVER_LIMIT = 10
DEFAULT_IMAGE_RESULTS = 6

DEFAULT_MIN_YEAR = 1990
DEFAULT_MAX_YEAR = 2024
DEFAULT_MIN_RATING = 0.0
DEFAULT_MAX_RATING = 10.0
DEFAULT_MIN_POPULARITY = 0.0

CATEGORY_SEARCH_PARAMS = {
    "Top Rated": {
        "query": "highly rated acclaimed masterpiece",
        "min_rating": 7.5
    },
    "Popular": {
        "query": "popular trending blockbuster",
        "min_popularity": 100
    },
    "Recent": {
        "query": "new recent latest releases",
        "min_year": 2020
    },
    "Classic": {
        "query": "classic legendary iconic",
        "max_year": 1990,
        "min_rating": 7.0
    }
}

SESSION_STATE_DEFAULTS = {
    'text_model': None,
    'image_model': None,
    'image_processor': None,
    'image_device': None,
    'milvus_connected': False,
    'text_collection': None,
    'image_collection': None,
    'watchlist': [],
    'favorites': [],
    'search_history': [],
    'show_filters': False,
    'current_view': 'home',
    'movie_notes': {},
    'personal_ratings': {},
    'theme': 'dark'
}

def init_session_state(session_state):
    for key, default_value in SESSION_STATE_DEFAULTS.items():
        if key not in session_state:
            if isinstance(default_value, (list, dict)):
                session_state[key] = default_value.copy() if default_value else type(default_value)()
            else:
                session_state[key] = default_value

def get_movie_id(movie):
    return f"{movie.get('title', '')}_{movie.get('release_date', '')}"

def get_star_rating(rating):
    if not rating:
        return ""
    stars = int(float(rating) / 2)
    half_star = (float(rating) / 2) % 1 >= 0.5
    full_stars = "*" * stars
    half = "+" if half_star and stars < 5 else ""
    empty = "-" * (5 - stars - (1 if half_star else 0))
    return full_stars + half + empty

def export_list_to_json(list_data, list_name):
    export_data = {
        'exported_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'list_name': list_name,
        'movies': list_data
    }
    return json.dumps(export_data, indent=2)

def add_to_watchlist(session_state, movie):
    movie_id = get_movie_id(movie)
    existing_ids = [get_movie_id(m) for m in session_state.watchlist]
    if movie_id not in existing_ids:
        movie_copy = movie.copy()
        movie_copy['added_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        session_state.watchlist.append(movie_copy)
        return True
    return False

def add_to_favorites(session_state, movie):
    movie_id = get_movie_id(movie)
    existing_ids = [get_movie_id(m) for m in session_state.favorites]
    if movie_id not in existing_ids:
        movie_copy = movie.copy()
        movie_copy['added_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        session_state.favorites.append(movie_copy)
        return True
    return False

def remove_from_watchlist(session_state, index):
    if 0 <= index < len(session_state.watchlist):
        session_state.watchlist.pop(index)
        return True
    return False

def remove_from_favorites(session_state, index):
    if 0 <= index < len(session_state.favorites):
        session_state.favorites.pop(index)
        return True
    return False

def add_to_search_history(session_state, query):
    if query and query.strip():
        history_item = {
            'query': query.strip(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        session_state.search_history = [
            h for h in session_state.search_history 
            if h['query'].lower() != query.lower()
        ]
        session_state.search_history.insert(0, history_item)
        session_state.search_history = session_state.search_history[:MAX_SEARCH_HISTORY]

def save_movie_note(session_state, movie_id, note):
    session_state.movie_notes[movie_id] = note

def save_personal_rating(session_state, movie_id, rating):
    session_state.personal_ratings[movie_id] = rating

def get_note_preview(session_state, movie_id, max_length=40):
    if movie_id in session_state.movie_notes and session_state.movie_notes[movie_id]:
        note = session_state.movie_notes[movie_id]
        if len(note) > max_length:
            return f"{note[:max_length]}..."
        return note
    return ""

def calculate_watch_time(watchlist, avg_movie_hours=2):
    return len(watchlist) * avg_movie_hours

def calculate_average_rating(movies):
    if not movies:
        return 0.0
    total = sum(float(m.get('vote_average', 0)) for m in movies)
    return total / len(movies)

def get_category_search_params(category, genre=None, limit=DEFAULT_DISCOVER_LIMIT):
    for key, params in CATEGORY_SEARCH_PARAMS.items():
        if key in category:
            result = {'top_k': limit}
            result.update(params)
            return result['query'], {k: v for k, v in result.items() if k != 'query'}
    
    if genre:
        return f"best {genre} movies", {'top_k': limit, 'genre_filter': genre}
    
    return "popular movies", {'top_k': limit}

def build_search_kwargs(top_k, show_filters, min_year, max_year, 
                        min_rating, max_rating, min_pop, genre_filter):
    kwargs = {'top_k': top_k}
    if show_filters:
        kwargs.update({
            'min_year': min_year,
            'max_year': max_year,
            'min_rating': min_rating,
            'max_rating': max_rating
        })
        if min_pop > 0:
            kwargs['min_popularity'] = min_pop
        if genre_filter != "All":
            kwargs['genre_filter'] = genre_filter
    return kwargs
