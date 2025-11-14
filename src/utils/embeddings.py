#!/usr/bin/env python3
"""Text embedding generation"""
import numpy as np
from sentence_transformers import SentenceTransformer
import os


class EmbeddingGenerator:
    """Generate embeddings for text samples"""
    
    def __init__(self, config: dict):
        self.config = config['embeddings']
        self.model_name = self.config['model']
        self.device = self.config.get('device', 'auto')
        self.batch_size = self.config.get('batch_size', 32)
        self.normalize = self.config.get('normalize', True)
        
        # Load model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Cache
        self.cache_enabled = self.config.get('cache', False)
        self.cache = {}
    
    def encode(self, texts, show_progress=False):
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize
        )
        
        return np.array(embeddings)
