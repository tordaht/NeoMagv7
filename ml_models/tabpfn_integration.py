"""NeoMag V7 - TabPFN Integration Module"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    logging.warning("TabPFN not available. Install with: pip install tabpfn")

@dataclass
class TabPFNPrediction:
    """TabPFN tahmin sonucu"""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    prediction_time: float = 0.0
    model_type: str = "classification"

class NeoMagTabPFNPredictor:
    """NeoMag simülasyonu için optimize edilmiş TabPFN wrapper"""
    
    def __init__(self, device: str = 'cpu', use_ensemble: bool = True):
        self.device = device
        self.use_ensemble = use_ensemble
        self.classifier = None
        self.regressor = None
        self.is_initialized = False
        
        if TABPFN_AVAILABLE:
            self._initialize_models()
        else:
            logging.error("TabPFN not available. Please install tabpfn package.")
    
    def _initialize_models(self):
        """TabPFN modellerini başlat"""
        try:
            self.classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=32 if self.use_ensemble else 1)
            self.regressor = TabPFNRegressor(device=self.device, N_ensemble_configurations=32 if self.use_ensemble else 1)
            self.is_initialized = True
            logging.info(f"TabPFN initialized on {self.device} with ensemble={self.use_ensemble}")
        except Exception as e:
            logging.error(f"TabPFN initialization error: {e}")
            self.is_initialized = False
    
    def predict_bacterial_behavior(self, bacterial_features: np.ndarray, behavior_labels: np.ndarray, 
                                  test_features: np.ndarray) -> TabPFNPrediction:
        """Bakterilerin davranış tahmini"""
        if not self.is_initialized:
            raise RuntimeError("TabPFN not initialized")
        
        start_time = time.time()
        
        # Veri validasyonu
        bacterial_features, behavior_labels = self._validate_and_preprocess(
            bacterial_features, behavior_labels, task_type='classification'
        )
        
        # TabPFN fit & predict
        self.classifier.fit(bacterial_features, behavior_labels)
        predictions = self.classifier.predict(test_features)
        probabilities = self.classifier.predict_proba(test_features)
        
        prediction_time = time.time() - start_time
        
        return TabPFNPrediction(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=np.max(probabilities, axis=1),
            prediction_time=prediction_time,
            model_type="classification"
        )
    
    def predict_fitness_landscape(self, genetic_profiles: np.ndarray, fitness_scores: np.ndarray,
                                 test_profiles: np.ndarray) -> TabPFNPrediction:
        """Genetik profil -> Fitness mapping"""
        if not self.is_initialized:
            raise RuntimeError("TabPFN not initialized")
        
        start_time = time.time()
        
        # Veri validasyonu
        genetic_profiles, fitness_scores = self._validate_and_preprocess(
            genetic_profiles, fitness_scores, task_type='regression'
        )
        
        # TabPFN fit & predict
        self.regressor.fit(genetic_profiles, fitness_scores)
        predictions = self.regressor.predict(test_profiles)
        
        prediction_time = time.time() - start_time
        
        return TabPFNPrediction(
            predictions=predictions,
            prediction_time=prediction_time,
            model_type="regression"
        )
    
    def predict_chemical_response(self, environmental_data: np.ndarray, response_classes: np.ndarray,
                                test_environments: np.ndarray) -> TabPFNPrediction:
        """Kimyasal çevre tepki tahmini"""
        if not self.is_initialized:
            raise RuntimeError("TabPFN not initialized")
        
        start_time = time.time()
        
        # Veri validasyonu
        environmental_data, response_classes = self._validate_and_preprocess(
            environmental_data, response_classes, task_type='classification'
        )
        
        # TabPFN fit & predict
        self.classifier.fit(environmental_data, response_classes)
        predictions = self.classifier.predict(test_environments)
        probabilities = self.classifier.predict_proba(test_environments)
        
        prediction_time = time.time() - start_time
        
        return TabPFNPrediction(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=np.max(probabilities, axis=1),
            prediction_time=prediction_time,
            model_type="classification"
        )
    
    def _validate_and_preprocess(self, X: np.ndarray, y: np.ndarray, 
                                task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Veri validasyon ve ön işleme"""
        
        # TabPFN kısıtlamaları
        if X.shape[0] > 1000:
            logging.warning(f"Sample count {X.shape[0]} > 1000, subsampling...")
            indices = np.random.choice(X.shape[0], 1000, replace=False)
            X = X[indices]
            y = y[indices]
        
        if X.shape[1] > 100:
            logging.warning(f"Feature count {X.shape[1]} > 100, selecting top features...")
            # Feature importance based selection (basit correlation)
            correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
            top_features = np.argsort(correlations)[-100:]
            X = X[:, top_features]
        
        # Missing value kontrolü
        if np.isnan(X).any() or np.isnan(y).any():
            logging.warning("Missing values detected, applying imputation...")
            X = np.nan_to_num(X, nan=np.nanmean(X))
            y = np.nan_to_num(y, nan=np.nanmean(y))
        
        # Classification için class sayısı kontrolü
        if task_type == 'classification':
            unique_classes = len(np.unique(y))
            if unique_classes > 10:
                logging.warning(f"Class count {unique_classes} > 10, binning classes...")
                y = self._bin_classes(y, max_classes=10)
        
        return X.astype(np.float32), y.astype(np.int32 if task_type == 'classification' else np.float32)
    
    def _bin_classes(self, y: np.ndarray, max_classes: int = 10) -> np.ndarray:
        """Çok sayıda sınıfı gruplandır"""
        if len(np.unique(y)) <= max_classes:
            return y
        
        # Quantile-based binning
        bins = np.quantile(y, np.linspace(0, 1, max_classes + 1))
        return np.digitize(y, bins) - 1

class BiophysicalPredictor(NeoMagTabPFNPredictor):
    """Biyofiziksel özellik tahmin modeli"""
    
    def predict_membrane_potential(self, ion_concentrations: np.ndarray, 
                                  membrane_potentials: np.ndarray,
                                  test_concentrations: np.ndarray) -> TabPFNPrediction:
        """İyon konsantrasyonlarından membran potansiyeli tahmin et"""
        return self.predict_fitness_landscape(
            ion_concentrations, membrane_potentials, test_concentrations
        )
    
    def predict_growth_rate(self, environmental_features: np.ndarray,
                           growth_rates: np.ndarray,
                           test_features: np.ndarray) -> TabPFNPrediction:
        """Çevresel koşullardan büyüme hızı tahmin et"""
        return self.predict_fitness_landscape(
            environmental_features, growth_rates, test_features
        )

class PopulationPredictor(NeoMagTabPFNPredictor):
    """Popülasyon dinamik tahmin modeli"""
    
    def predict_population_dynamics(self, population_states: np.ndarray,
                                   next_states: np.ndarray,
                                   test_states: np.ndarray) -> TabPFNPrediction:
        """Popülasyon dinamik tahmini"""
        return self.predict_fitness_landscape(
            population_states, next_states, test_states
        )
    
    def predict_extinction_risk(self, population_features: np.ndarray,
                               survival_labels: np.ndarray,
                               test_features: np.ndarray) -> TabPFNPrediction:
        """Popülasyon yok olma riski tahmini"""
        return self.predict_bacterial_behavior(
            population_features, survival_labels, test_features
        )

class RLValuePredictor(NeoMagTabPFNPredictor):
    """RL Value Function Approximation"""
    
    def predict_action_values(self, state_action_pairs: np.ndarray,
                             q_values: np.ndarray,
                             test_pairs: np.ndarray) -> TabPFNPrediction:
        """Q-value function approximation"""
        return self.predict_fitness_landscape(
            state_action_pairs, q_values, test_pairs
        )
    
    def predict_optimal_actions(self, states: np.ndarray,
                               optimal_actions: np.ndarray,
                               test_states: np.ndarray) -> TabPFNPrediction:
        """Optimal action prediction"""
        return self.predict_bacterial_behavior(
            states, optimal_actions, test_states
        )

# Factory function
def create_tabpfn_predictor(predictor_type: str = "general", 
                           device: str = 'cpu', 
                           use_ensemble: bool = True) -> NeoMagTabPFNPredictor:
    """TabPFN predictor factory"""
    
    predictors = {
        "general": NeoMagTabPFNPredictor,
        "biophysical": BiophysicalPredictor,
        "population": PopulationPredictor,
        "rl_value": RLValuePredictor
    }
    
    if predictor_type not in predictors:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    return predictors[predictor_type](device=device, use_ensemble=use_ensemble)

# Utility functions
def benchmark_tabpfn_performance():
    """TabPFN performans testi"""
    if not TABPFN_AVAILABLE:
        print("❌ TabPFN not available for benchmarking")
        return
    
    print("🚀 TabPFN Performance Benchmark")
    print("=" * 40)
    
    # Test data oluştur
    n_samples = 500
    n_features = 50
    n_classes = 5
    
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randint(0, n_classes, n_samples)
    X_test = np.random.randn(100, n_features).astype(np.float32)
    
    # Predictor test
    predictor = create_tabpfn_predictor("general", device='cpu')
    
    start_time = time.time()
    result = predictor.predict_bacterial_behavior(X_train, y_train, X_test)
    total_time = time.time() - start_time
    
    print(f"✅ Prediction completed in {total_time:.3f} seconds")
    print(f"📊 Predictions shape: {result.predictions.shape}")
    print(f"🎯 Confidence scores: {result.confidence_scores.mean():.3f} ± {result.confidence_scores.std():.3f}")
    print(f"⚡ TabPFN prediction time: {result.prediction_time:.3f} seconds")

if __name__ == "__main__":
    benchmark_tabpfn_performance()
