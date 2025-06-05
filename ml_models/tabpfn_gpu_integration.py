"""NeoMag V7 - TabPFN GPU Hızlandırma Modülü (RTX 3060 Optimize)"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import time
import psutil
import os

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    logging.warning("TabPFN not available. Install with: pip install tabpfn")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. Install with: pip install cupy-cuda12x")

try:
    from numba import cuda, jit
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    logging.warning("Numba CUDA not available. Install with: pip install numba")

@dataclass
class GPUSystemInfo:
    """GPU sistem bilgileri"""
    gpu_name: str
    gpu_memory_total: float  # GB
    gpu_memory_free: float   # GB
    cuda_version: str
    torch_cuda_available: bool
    cupy_available: bool
    numba_cuda_available: bool
    recommended_ensemble_size: int

def detect_gpu_capabilities() -> GPUSystemInfo:
    """GPU özelliklerini otomatik tespit et"""
    if not torch.cuda.is_available():
        return GPUSystemInfo(
            gpu_name="None",
            gpu_memory_total=0.0,
            gpu_memory_free=0.0,
            cuda_version="None",
            torch_cuda_available=False,
            cupy_available=False,
            numba_cuda_available=False,
            recommended_ensemble_size=1
        )
    
    gpu_props = torch.cuda.get_device_properties(0)
    total_memory_gb = gpu_props.total_memory / (1024**3)
    free_memory_gb = (gpu_props.total_memory - torch.cuda.memory_allocated()) / (1024**3)
    
    # RTX 3060 8GB için ensemble boyutu önerileri
    if total_memory_gb >= 12:
        recommended_ensemble = 32
    elif total_memory_gb >= 8:
        recommended_ensemble = 16  # RTX 3060 için optimal
    elif total_memory_gb >= 6:
        recommended_ensemble = 8
    else:
        recommended_ensemble = 4
    
    return GPUSystemInfo(
        gpu_name=gpu_props.name,
        gpu_memory_total=total_memory_gb,
        gpu_memory_free=free_memory_gb,
        cuda_version=torch.version.cuda,
        torch_cuda_available=torch.cuda.is_available(),
        cupy_available=CUPY_AVAILABLE,
        numba_cuda_available=NUMBA_CUDA_AVAILABLE,
        recommended_ensemble_size=recommended_ensemble
    )

class TabPFNGPUAccelerator:
    """TabPFN GPU hızlandırma sınıfı"""
    
    def __init__(self, auto_detect: bool = True, force_ensemble_size: Optional[int] = None):
        self.gpu_info = detect_gpu_capabilities()
        self.device = 'cuda' if self.gpu_info.torch_cuda_available else 'cpu'
        
        # Ensemble boyutunu belirle
        if force_ensemble_size:
            self.ensemble_size = force_ensemble_size
        elif auto_detect:
            self.ensemble_size = self.gpu_info.recommended_ensemble_size
        else:
            self.ensemble_size = 16  # RTX 3060 default
        
        self.classifier = None
        self.regressor = None
        self.is_initialized = False
        
        self._log_system_info()
        self._initialize_models()
    
    def _log_system_info(self):
        """Sistem bilgilerini logla"""
        logging.info(f"🔥 GPU HIZLANDIRMA AKTIF 🔥")
        logging.info(f"GPU: {self.gpu_info.gpu_name}")
        logging.info(f"VRAM: {self.gpu_info.gpu_memory_total:.1f}GB total, {self.gpu_info.gpu_memory_free:.1f}GB free")
        logging.info(f"CUDA: {self.gpu_info.cuda_version}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Ensemble Size: {self.ensemble_size}")
        logging.info(f"CuPy: {self.gpu_info.cupy_available}")
        logging.info(f"Numba CUDA: {self.gpu_info.numba_cuda_available}")
    
    def _initialize_models(self):
        """GPU optimize TabPFN modelleri"""
        if not TABPFN_AVAILABLE:
            logging.error("❌ TabPFN not available")
            return
        
        try:
            # VRAM kontrolü
            if self.device == 'cuda':
                torch.cuda.empty_cache()  # Cache temizle
                
            # TabPFN modelleri (GPU device ile)
            self.classifier = TabPFNClassifier(
                device=self.device, 
                n_estimators=self.ensemble_size
            )
            self.regressor = TabPFNRegressor(
                device=self.device,
                n_estimators=self.ensemble_size
            )
            
            self.is_initialized = True
            
            if self.device == 'cuda':
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                logging.info(f"✅ TabPFN GPU başlatıldı - VRAM kullanımı: {memory_used:.2f}GB")
            else:
                logging.info("✅ TabPFN CPU modunda başlatıldı")
                
        except Exception as e:
            logging.error(f"❌ TabPFN GPU başlatma hatası: {e}")
            # Fallback CPU mode
            try:
                self.device = 'cpu'
                self.ensemble_size = 4  # CPU için azalt
                self.classifier = TabPFNClassifier(device='cpu', n_estimators=self.ensemble_size)
                self.regressor = TabPFNRegressor(device='cpu', n_estimators=self.ensemble_size)
                self.is_initialized = True
                logging.info("⚠️ GPU başarısız, CPU moduna geçildi")
            except Exception as e2:
                logging.error(f"❌ CPU fallback de başarısız: {e2}")
                self.is_initialized = False
    
    def predict_with_gpu_optimization(self, X_train: np.ndarray, y_train: np.ndarray, 
                                    X_test: np.ndarray, task_type: str = 'classification', 
                                    batch_size: int = None) -> Dict[str, Any]:
        """GPU optimize edilmiş tahmin"""
        if not self.is_initialized:
            raise RuntimeError("TabPFN GPU not initialized")
        
        # Pre-processing
        start_time = time.time()
        X_train, y_train = self._preprocess_for_gpu(X_train, y_train, task_type)
        X_test = self._preprocess_test_data(X_test)
        preprocess_time = time.time() - start_time
        
        # GPU bellek kontrolü
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / (1024**3)
        
        # TabPFN prediction with Mixed Precision (FP16) for RTX 3060
        prediction_start = time.time()
        
        if self.device == 'cuda':
            from torch.cuda.amp import autocast
            with torch.no_grad():
                with autocast():  # Mixed precision hızlandırma
                    if task_type == 'classification':
                        self.classifier.fit(X_train, y_train)
                        predictions = self.classifier.predict(X_test)
                        probabilities = self.classifier.predict_proba(X_test)
                        confidence = np.max(probabilities, axis=1)
                    else:  # regression
                        self.regressor.fit(X_train, y_train)
                        predictions = self.regressor.predict(X_test)
                        probabilities = None
                        confidence = np.ones(len(predictions))  # Regression için dummy
        else:
            # CPU fallback - normal precision
            if task_type == 'classification':
                self.classifier.fit(X_train, y_train)
                predictions = self.classifier.predict(X_test)
                probabilities = self.classifier.predict_proba(X_test)
                confidence = np.max(probabilities, axis=1)
            else:  # regression
                self.regressor.fit(X_train, y_train)
                predictions = self.regressor.predict(X_test)
                probabilities = None
                confidence = np.ones(len(predictions))  # Regression için dummy
        
        prediction_time = time.time() - prediction_start
        total_time = time.time() - start_time
        
        # GPU bellek sonrası
        if self.device == 'cuda':
            memory_after = torch.cuda.memory_allocated() / (1024**3)
            memory_used = memory_after - memory_before
        else:
            memory_used = 0.0
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence,
            'performance_metrics': {
                'preprocess_time': preprocess_time,
                'prediction_time': prediction_time,
                'total_time': total_time,
                'gpu_memory_used_gb': memory_used,
                'device': self.device,
                'ensemble_size': self.ensemble_size,
                'samples_processed': len(X_test),
                'throughput_samples_per_sec': len(X_test) / prediction_time if prediction_time > 0 else 0
            }
        }
    
    def _preprocess_for_gpu(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """GPU için optimize edilmiş ön işleme"""
        
        # TabPFN kısıtlamaları (GPU'da daha katı)
        max_samples = 800 if self.device == 'cuda' else 1000  # GPU için biraz azalt
        max_features = 80 if self.device == 'cuda' else 100   # GPU için biraz azalt
        
        if X.shape[0] > max_samples:
            logging.warning(f"🔥 GPU: Subsampling {X.shape[0]} → {max_samples}")
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        if X.shape[1] > max_features:
            logging.warning(f"🔥 GPU: Feature selection {X.shape[1]} → {max_features}")
            correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
            top_features = np.argsort(correlations)[-max_features:]
            X = X[:, top_features]
        
        # GPU için float32 optimize
        X = X.astype(np.float32)
        
        if task_type == 'classification':
            y = y.astype(np.int32)
            # Class sayısını sınırla
            unique_classes = len(np.unique(y))
            if unique_classes > 8:  # GPU için daha az class
                y = self._bin_classes_gpu(y, max_classes=8)
        else:
            y = y.astype(np.float32)
        
        # NaN temizle
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def predict_in_batches(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, task_type: str = 'classification',
                         batch_size: int = 256) -> Dict[str, Any]:
        """RTX 3060 için optimize edilmiş batch processing"""
        if not self.is_initialized:
            raise RuntimeError("TabPFN GPU not initialized")
            
        # Eğer test verisi küçükse batch'leme gereksiz
        if len(X_test) <= batch_size:
            return self.predict_with_gpu_optimization(X_train, y_train, X_test, task_type)
        
        logging.info(f"🔥 Batch processing: {len(X_test)} samples in {batch_size} batch size")
        
        # İlk batch'te model eğit
        start_time = time.time()
        X_train, y_train = self._preprocess_for_gpu(X_train, y_train, task_type)
        
        if self.device == 'cuda':
            from torch.cuda.amp import autocast
            with torch.no_grad():
                with autocast():
                    if task_type == 'classification':
                        self.classifier.fit(X_train, y_train)
                    else:
                        self.regressor.fit(X_train, y_train)
        else:
            if task_type == 'classification':
                self.classifier.fit(X_train, y_train)
            else:
                self.regressor.fit(X_train, y_train)
        
        # Batch'lerde tahmin yap
        all_predictions = []
        all_probabilities = []
        all_confidence = []
        total_memory_used = 0.0
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            batch_processed = self._preprocess_test_data(batch)
            
            if self.device == 'cuda':
                memory_before = torch.cuda.memory_allocated() / (1024**3)
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    with autocast():
                        if task_type == 'classification':
                            batch_pred = self.classifier.predict(batch_processed)
                            batch_prob = self.classifier.predict_proba(batch_processed)
                            batch_conf = np.max(batch_prob, axis=1)
                        else:
                            batch_pred = self.regressor.predict(batch_processed)
                            batch_prob = None
                            batch_conf = np.ones(len(batch_pred))
                
                memory_after = torch.cuda.memory_allocated() / (1024**3)
                total_memory_used += (memory_after - memory_before)
            else:
                if task_type == 'classification':
                    batch_pred = self.classifier.predict(batch_processed)
                    batch_prob = self.classifier.predict_proba(batch_processed)
                    batch_conf = np.max(batch_prob, axis=1)
                else:
                    batch_pred = self.regressor.predict(batch_processed)
                    batch_prob = None
                    batch_conf = np.ones(len(batch_pred))
            
            all_predictions.append(batch_pred)
            if batch_prob is not None:
                all_probabilities.append(batch_prob)
            all_confidence.append(batch_conf)
        
        # Sonuçları birleştir
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities) if all_probabilities else None
        confidence = np.concatenate(all_confidence)
        
        total_time = time.time() - start_time
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_scores': confidence,
            'performance_metrics': {
                'total_time': total_time,
                'batch_size': batch_size,
                'num_batches': len(range(0, len(X_test), batch_size)),
                'gpu_memory_used_gb': total_memory_used,
                'device': self.device,
                'ensemble_size': self.ensemble_size,
                'samples_processed': len(X_test),
                'throughput_samples_per_sec': len(X_test) / total_time if total_time > 0 else 0
            }
        }
    
    def _preprocess_test_data(self, X_test: np.ndarray) -> np.ndarray:
        """Test verisi ön işleme"""
        return np.nan_to_num(X_test.astype(np.float32), nan=0.0)
    
    def _bin_classes_gpu(self, y: np.ndarray, max_classes: int = 8) -> np.ndarray:
        """GPU optimize sınıf gruplama"""
        if len(np.unique(y)) <= max_classes:
            return y
        
        bins = np.quantile(y, np.linspace(0, 1, max_classes + 1))
        return np.digitize(y, bins) - 1
    
    def benchmark_gpu_performance(self, n_samples: int = 500, n_features: int = 50, n_classes: int = 5):
        """GPU performans testi"""
        if not self.is_initialized:
            logging.error("❌ TabPFN GPU not initialized for benchmark")
            return
        
        logging.info(f"🔥 GPU PERFORMANS TESTİ BAŞLIYOR 🔥")
        logging.info(f"Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")
        
        # Test verisi oluştur
        np.random.seed(42)
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_samples)
        X_test = np.random.randn(100, n_features).astype(np.float32)
        
        # Classification benchmark
        try:
            result_cls = self.predict_with_gpu_optimization(X_train, y_train, X_test, 'classification')
            metrics_cls = result_cls['performance_metrics']
            
            logging.info("📊 CLASSIFICATION BENCHMARK:")
            logging.info(f"  Prediction Time: {metrics_cls['prediction_time']:.6f}s")
            logging.info(f"  Total Time: {metrics_cls['total_time']:.6f}s")
            logging.info(f"  Throughput: {metrics_cls['throughput_samples_per_sec']:.1f} samples/sec")
            logging.info(f"  GPU Memory: {metrics_cls['gpu_memory_used_gb']:.3f}GB")
            logging.info(f"  Device: {metrics_cls['device']}")
            logging.info(f"  Ensemble: {metrics_cls['ensemble_size']}")
            
        except Exception as e:
            logging.error(f"❌ Classification benchmark failed: {e}")
        
        # Regression benchmark
        try:
            y_train_reg = np.random.randn(n_samples).astype(np.float32)
            result_reg = self.predict_with_gpu_optimization(X_train, y_train_reg, X_test, 'regression')
            metrics_reg = result_reg['performance_metrics']
            
            logging.info("📊 REGRESSION BENCHMARK:")
            logging.info(f"  Prediction Time: {metrics_reg['prediction_time']:.6f}s")
            logging.info(f"  Total Time: {metrics_reg['total_time']:.6f}s")
            logging.info(f"  Throughput: {metrics_reg['throughput_samples_per_sec']:.1f} samples/sec")
            logging.info(f"  GPU Memory: {metrics_reg['gpu_memory_used_gb']:.3f}GB")
            
        except Exception as e:
            logging.error(f"❌ Regression benchmark failed: {e}")
        
        # VRAM kullanım raporu
        if self.device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            logging.info(f"🔥 VRAM DURUMU:")
            logging.info(f"  Allocated: {memory_allocated:.3f}GB")
            logging.info(f"  Reserved: {memory_reserved:.3f}GB")
            logging.info(f"  Free: {self.gpu_info.gpu_memory_total - memory_reserved:.3f}GB")

# Factory function
def create_gpu_tabpfn_predictor(auto_detect: bool = True, 
                               force_ensemble_size: Optional[int] = None) -> TabPFNGPUAccelerator:
    """GPU TabPFN predictor oluştur"""
    return TabPFNGPUAccelerator(auto_detect=auto_detect, force_ensemble_size=force_ensemble_size)

# Benchmark function
def run_gpu_benchmark():
    """Standalone GPU benchmark"""
    predictor = create_gpu_tabpfn_predictor()
    predictor.benchmark_gpu_performance()
    return predictor

if __name__ == "__main__":
    # GPU test
    logging.basicConfig(level=logging.INFO)
    
    print("🔥 NeoMag V7 - TabPFN GPU Accelerator Test 🔥")
    
    # System info
    gpu_info = detect_gpu_capabilities()
    print(f"GPU: {gpu_info.gpu_name}")
    print(f"VRAM: {gpu_info.gpu_memory_total:.1f}GB")
    print(f"Recommended Ensemble: {gpu_info.recommended_ensemble_size}")
    
    # Benchmark
    if gpu_info.torch_cuda_available:
        predictor = run_gpu_benchmark()
    else:
        print("❌ CUDA not available, skipping GPU benchmark")