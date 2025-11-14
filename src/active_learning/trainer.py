#!/usr/bin/env python3
"""
Weighted Model Trainer
Trains classification model with sample weights
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import joblib


class WeightedModelTrainer:
    """Train models with weighted samples"""
    
    def __init__(self, config: dict):
        self.config = config['model']
        self.architecture = self.config['architecture']
        
        # For simplicity, using sklearn initially
        # Can be extended to transformers
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = None
    
    def initialize_model(self):
        """Initialize a new model"""
        return LogisticRegression(random_state=42, max_iter=1000)
    
    def train(self, training_data: list, sample_weights: list):
        """
        Train model with weighted samples.
        
        Args:
            training_data: List of dicts with 'text' and 'label'
            sample_weights: List of weights for each sample
            
        Returns:
            Trained model
        """
        if not training_data:
            return self.initialize_model()
        
        # Extract texts and labels
        texts = [s['example'] for s in training_data]
        labels = [s['Label'] for s in training_data]
        
        # Vectorize
        X = self.vectorizer.fit_transform(texts)
        
        # Train with weights
        model = self.initialize_model()
        model.fit(X, labels, sample_weight=sample_weights)
        
        # Store label classes
        model.classes_ = np.unique(labels)
        
        # Wrap with prediction methods
        self.model = ModelWrapper(model, self.vectorizer)
        return self.model
    
    def save_model(self, model, path: str):
        """Save model to file"""
        joblib.dump({
            'model': model.model,
            'vectorizer': model.vectorizer
        }, path)
    
    def load_model(self, path: str):
        """Load model from file"""
        data = joblib.load(path)
        self.model = ModelWrapper(data['model'], data['vectorizer'])
        return self.model


class ModelWrapper:
    """Wrapper to provide consistent interface"""
    
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.classes_ = model.classes_
    
    def predict(self, texts: list):
        """Predict labels"""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: list):
        """Predict probabilities"""
        if not texts:  # empty list/tuple
            import numpy as np
            return np.empty((0, len(self.classes_)), dtype=float)
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)
