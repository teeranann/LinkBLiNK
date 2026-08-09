"""Appearance embeddings used to re-identify particles across blinking gaps."""

from .siamese_embedding import SiameseEmbedder, extract_patch

__all__ = ["SiameseEmbedder", "extract_patch"]
