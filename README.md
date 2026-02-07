# CineSemantics Transformer-Powered Movie Recommendation Engine

## Overview
CineSemantics is a movie recommendation engine that leverages transformer-based models and vector databases to provide advanced, semantic search and recommendations for movies. It supports both text-based and image-based queries, allowing users to find similar movies by plot, genre, ratings, and even poster images.

## Features
- Semantic search using transformer models (Sentence Transformers, CLIP)
- Text-based movie recommendations (plot, genre, ratings, etc.)
- Image-based movie recommendations (poster similarity)
- Vector database integration with Milvus for fast similarity search
- Configurable parameters via a central config file
- Batch processing for efficient embedding extraction
- Logging for monitoring and debugging

## Project Structure

```
├── data/                # Movie CSV data
├── data_cleaning/       # Data cleaning scripts and notebooks
├── posters/             # Movie poster images
├── utils/               # Core scripts and configuration
│   ├── config.py        # Central configuration file
│   ├── text_embedder.py # Text embedding utilities
│   ├── image_embedder.py# Image embedding utilities
│   ├── milvus_vectordb.py# Milvus vector DB integration
│   ├── text_main.py     # Main script for text-based search
│   ├── image_main.py    # Main script for image-based search
│   └── ...
├── vector_db/           # Vector database files
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup

1. **Install Python dependencies:**
	- Run `pip install -r requirements.txt` to install all required packages.

2. **Prepare your data:**
	- Place your movie CSV file in the `data/` directory.
	- Place movie poster images in the `posters/` directory.

3. **Configure parameters:**
	- Edit `utils/config.py` to adjust model names, batch sizes, database settings, and other parameters as needed.

## Usage

### Text-Based Recommendations
Run the main script for text-based search:

```
python utils/text_main.py
```
This will connect to Milvus, load the transformer model, and allow you to search for movies by plot, genre, rating, and more.

### Image-Based Recommendations
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

See `requirements.txt` for the full list.

## License
This project is for educational and research purposes. Please check individual package licenses for commercial use.

## Contact
For questions or contributions, please open an issue or contact the project maintainer.
