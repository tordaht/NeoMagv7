#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TabPFN GPU Integration Module
Modular TabPFN service for both test and production servers
"""

import logging
import torch
import numpy as np
import time
import sys
import os
from pathlib import Path

# Logger ayarları
logger = logging.getLogger("tabpfn_integration")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class TabPFNService:
    """
    Unified TabPFN GPU service for production and testing
    Handles initialization, status checking, and predictions
    """
    
    def __init__(self, ensemble_size=4):
        self.model = None
        self.gpu_tabpfn = None
        self.ensemble_size = ensemble_size
        self.is_initialized = False
        self.status = "uninitialized"
        self.device = "cpu"
        self.gpu_info = {}
        
        logger.debug(f"TabPFNService created with ensemble_size={ensemble_size}")
    
    def initialize(self):
        """Initialize TabPFN GPU acceleration"""
        try:
            logger.debug("=== TabPFN GPU Initialization Starting ===")
            
            # CUDA check
            cuda_available = torch.cuda.is_available()
            logger.debug(f"CUDA mevcut mu? {cuda_available}")
            
            if not cuda_available:
                logger.info("🔧 CPU mode: CUDA not available, using CPU optimization")
                self.device = "cpu"
                self._initialize_cpu_fallback()
                return
            
            # Import TabPFN
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            logger.debug("✅ TabPFN base import successful")
            
            # Try GPU TabPFN integration
            try:
                current_dir = Path(__file__).parent
                project_root = current_dir.parent
                ml_models_dir = project_root / "ml_models"
                
                if ml_models_dir.exists():
                    sys.path.insert(0, str(ml_models_dir))
                    from tabpfn_gpu_integration import TabPFNGPUAccelerator, detect_gpu_capabilities
                    
                    gpu_info = detect_gpu_capabilities()
                    self.gpu_info = {
                        "gpu_name": gpu_info.gpu_name,
                        "vram_total": gpu_info.gpu_memory_total,
                        "cuda_version": gpu_info.cuda_version,
                        "device": "cuda"
                    }
                    
                    logger.debug(f"🔥 GPU: {gpu_info.gpu_name}")
                    logger.debug(f"🔥 VRAM: {gpu_info.gpu_memory_total:.1f}GB")
                    
                    self.gpu_tabpfn = TabPFNGPUAccelerator(auto_detect=True)
                    self.device = "cuda"
                    self.is_initialized = True
                    self.status = "gpu_ready"
                    
                    logger.debug("✅ TabPFN GPU acceleration enabled!")
                    
                else:
                    logger.warning("ML models directory not found, using CPU fallback")
                    self._initialize_cpu_fallback()
                    
            except Exception as gpu_e:
                logger.warning(f"GPU TabPFN failed: {gpu_e}, using CPU fallback")
                self._initialize_cpu_fallback()
                
        except Exception as e:
            logger.exception(f"TabPFN initialization failed: {e}")
            self.status = "error"
            self.is_initialized = False
            raise
    
    def _initialize_cpu_fallback(self):
        """Initialize CPU-only TabPFN"""
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            
            self.model = {
                'classifier': TabPFNClassifier(device='cpu', n_estimators=self.ensemble_size),
                'regressor': TabPFNRegressor(device='cpu', n_estimators=self.ensemble_size)
            }
            
            self.device = "cpu"
            self.is_initialized = True
            self.status = "cpu_ready"
            self.gpu_info = {"device": "cpu", "message": "GPU not available"}
            
            logger.debug("✅ TabPFN CPU fallback initialized")
            
        except Exception as e:
            logger.exception(f"CPU TabPFN fallback failed: {e}")
            self.status = "error"
            self.is_initialized = False
            raise
    
    def get_status(self):
        """Get current TabPFN service status"""
        if not self.is_initialized:
            return {
                "status": self.status,
                "initialized": False,
                "message": "TabPFN not initialized"
            }
        
        if self.device == "cuda" and self.gpu_tabpfn:
            return {
                "status": self.status,
                "initialized": True,
                "tabpfn_gpu": True,
                "device": self.device,
                "gpu_name": self.gpu_info.get("gpu_name", "Unknown"),
                "vram_total": self.gpu_info.get("vram_total", 0),
                "ensemble_size": self.ensemble_size
            }
        else:
            return {
                "status": self.status,
                "initialized": True,
                "tabpfn_gpu": False,
                "device": self.device,
                "message": "CPU fallback mode"
            }
    
    def predict_bacterial_features(self, X_train, y_train, X_test, task_type='regression'):
        """
        Predict bacterial features using TabPFN
        Args:
            X_train: Training features (numpy array)
            y_train: Training targets (numpy array)
            X_test: Test features (numpy array)
            task_type: 'regression' or 'classification'
        """
        if not self.is_initialized:
            raise RuntimeError("TabPFN service not initialized")
        
        try:
            logger.debug(f"Bacterial prediction: {X_train.shape} -> {X_test.shape}")
            
            start_time = time.time()
            
            if self.device == "cuda" and self.gpu_tabpfn:
                # GPU prediction
                result = self.gpu_tabpfn.predict_with_gpu_optimization(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    task_type=task_type
                )
                
                prediction_time = time.time() - start_time
                
                return {
                    "predictions": result["predictions"],
                    "prediction_time": prediction_time,
                    "device_used": "cuda",
                    "gpu_acceleration": result.get("gpu_acceleration_factor", 1.0),
                    "status": "success"
                }
                
            else:
                # CPU fallback
                if task_type == 'regression':
                    model = self.model['regressor']
                else:
                    model = self.model['classifier']
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                prediction_time = time.time() - start_time
                
                return {
                    "predictions": predictions,
                    "prediction_time": prediction_time,
                    "device_used": "cpu",
                    "gpu_acceleration": 1.0,
                    "status": "success"
                }
                
        except Exception as e:
            logger.exception(f"Prediction failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "prediction_time": 0,
                "device_used": self.device
            }
    
    def test_performance(self, n_samples=100, n_features=10):
        """Test TabPFN performance with synthetic data"""
        if not self.is_initialized:
            raise RuntimeError("TabPFN service not initialized")
        
        # Generate test data
        X_train = np.random.rand(n_samples, n_features)
        y_train = np.random.rand(n_samples)
        X_test = np.random.rand(20, n_features)
        
        return self.predict_bacterial_features(X_train, y_train, X_test, 'regression')

# Global TabPFN service instance
tabpfn_service = TabPFNService(ensemble_size=4)

def get_tabpfn_service():
    """Get the global TabPFN service instance"""
    return tabpfn_service 