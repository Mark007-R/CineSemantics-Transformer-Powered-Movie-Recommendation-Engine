"""CineSemantics retrieval package (Day-5 Phase-3 home)."""
from .embedder import ChampionEmbedder, get_embedder, build_catalog_text
from .index import MovieIndex
from .metadata_filter import genre_match, genre_tokens, passes_filters

__all__ = [
    "ChampionEmbedder", "get_embedder", "build_catalog_text",
    "MovieIndex", "genre_match", "genre_tokens", "passes_filters",
]
