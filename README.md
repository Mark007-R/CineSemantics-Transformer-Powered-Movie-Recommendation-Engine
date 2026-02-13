# CineSemantics Transformer-Powered Movie Recommendation Engine

## Overview
CineSemantics is a movie recommendation engine that leverages transformer-based models, vector databases, and a modern Streamlit web interface to provide advanced, semantic search and recommendations for movies. Users can search by text, poster image, browse categories, manage a watchlist and favorites, and enjoy a Netflix/Disney+ inspired UI.

## Features
- Semantic search using transformer models (Sentence Transformers, CLIP)
- Text-based movie recommendations (plot, genre, ratings, etc.)
- Image-based movie recommendations (poster similarity)
- Vector database integration with Milvus for fast similarity search
- Streamlit web app with modern UI and multiple tabs
- Browse by category, genre, rating, popularity, and year
- Smart search with advanced filters
- Visual search by uploading poster images
- Manage personal watchlist and favorites
- Configurable parameters via a central config file
- Batch processing for efficient embedding extraction
- Logging for monitoring and debugging

## Project Structure

```
├── data/                # Movie CSV data
├── data_cleaning/       # Data cleaning scripts and notebooks
├── posters/             # Movie poster images
├── pages/               # Streamlit app scripts
│   ├── helpers.py       # UI helpers and shared utilities
│   ├── movieflix.py     # Main Streamlit app
│   └── style.css        # Custom styles
├── utils/               # Core scripts and configuration
│   ├── config.py        # Central configuration file
│   ├── text_embedder.py # Text embedding utilities
│   ├── image_embedder.py# Image embedding utilities
│   ├── milvus_vectordb.py# Milvus vector DB integration
│   ├── text_main.py     # Main script for text-based search
│   ├── image_main.py    # Main script for image-based search
│   └── ...
├── data_cleaning/       # Data cleaning notebook
│   └── data_cleaning.ipynb
├── vector_db/           # Vector database files
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup

1. **Install Python dependencies:**
   - Run `pip install -r requirements.txt` to install all required packages.

2. **Milvus vector database (embedded vs. Docker):**
   - **Default (recommended for quick start):** The app uses Milvus Lite in embedded mode and will start it automatically. No Docker setup is required for this mode.
   - **Optional: Run Milvus as a separate service via Docker Compose:**
     - Ensure you have Docker and Docker Compose installed.
     - From the project root, start Milvus using the provided compose file:
       - `docker-compose -f utils/docker-compose.yml up -d`
     - By default, Milvus will be available on `localhost:19530`. Make sure the Milvus host and port in `utils/config.py` match these values if you choose this option.

3. **Prepare your data:**
   - Place your movie CSV file in the `data/` directory.
   - Place movie poster images in the `posters/` directory.

4. **Configure parameters:**
   - Edit `utils/config.py` to adjust model names, batch sizes, database settings, and other parameters as needed.

## Usage

### Streamlit Web App

Launch the modern web interface for movie discovery:

```
streamlit run pages/movieflix.py
```

#### Main Features
- **Discover Tab:** Browse movies by category, genre, rating, popularity, or year.
- **Search Tab:** Use smart text search with advanced filters (year, rating, popularity, genre).
- **Visual Search Tab:** Upload a poster image to find visually similar movies.
- **Watchlist Tab:** Manage your personal watchlist.
- **Favorites Tab:** Manage your favorite movies.

#### Example Workflow
1. Start the Streamlit app.
2. Browse or search for movies using text or poster images.
3. Add movies to your watchlist or favorites.
4. Use advanced filters for refined search.
5. Enjoy a modern, responsive UI inspired by Netflix/Disney+.


### Text-Based Recommendations (CLI)
Run the main script for text-based search:

```
python utils/text_main.py
```
This will connect to Milvus, load the transformer model, and allow you to search for movies by plot, genre, rating, and more.

### Image-Based Recommendations (CLI)
Run the main script for image-based search:

```
python utils/image_main.py
```
This will connect to Milvus, load the CLIP model, and allow you to search for movies by poster image similarity.

## Configuration

All configurable parameters are located in `utils/config.py`, including:
- Model names and embedding dimensions
- Milvus connection settings
- Collection names and index parameters
- Batch sizes and field length limits
- Default file paths
- Logging settings

## Requirements
- Python 3.8+
- PyTorch
- Sentence Transformers
- Transformers
- Milvus
- pandas, numpy, tqdm, Pillow
- Streamlit

See `requirements.txt` for the full list.

## License
MIT License. See [LICENSE](LICENSE).

## Contact
For questions or contributions, please open an issue or contact the project maintainer.
