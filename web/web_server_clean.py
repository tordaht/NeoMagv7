#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeoMag V7 - Clean Web Server for TabPFN Debug
"""

from flask import Flask, jsonify
import logging
import sys
import os
import time

# Basit logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("=== WEB SERVER BAŞLATIYOR ===")

# Flask app
app = Flask(__name__)

print("=== SIMULATION OLUŞTURULUYOR ===")

try:
    # TabPFN test
    print("=== TabPFN TEST BAŞLIYOR ===")
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    print("✅ TabPFN import başarılı")
    
    # GPU TabPFN test
    print("=== GPU TabPFN TEST BAŞLIYOR ===")
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # GPU module import test
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_file_dir)
        ml_models_dir = os.path.join(parent_dir, "ml_models")
        
        print(f"Current file: {__file__}")
        print(f"ML models path: {ml_models_dir}")
        print(f"Path exists: {os.path.exists(ml_models_dir)}")
        
        if os.path.exists(ml_models_dir):
            target_file = os.path.join(ml_models_dir, "tabpfn_gpu_integration.py")
            print(f"Target file: {target_file}")
            print(f"Target file exists: {os.path.exists(target_file)}")
            
            if ml_models_dir not in sys.path:
                sys.path.insert(0, ml_models_dir)
                print(f"Added to sys.path: {ml_models_dir}")
            
            from tabpfn_gpu_integration import TabPFNGPUAccelerator
            print("✅ TabPFNGPUAccelerator import başarılı!")
            
            gpu_tabpfn = TabPFNGPUAccelerator()
            print(f"✅ GPU TabPFN initialized: {gpu_tabpfn.device}")
            
        else:
            print("❌ ML models directory not found")
            
    except Exception as gpu_e:
        print(f"❌ GPU TabPFN failed: {gpu_e}")
        import traceback
        traceback.print_exc()
    
    print("✅ TabPFN test tamamlandı")
    
except Exception as e:
    print(f"❌ TabPFN test failed: {e}")
    import traceback
    traceback.print_exc()

@app.route('/')
def index():
    return "NeoMag V7 Clean Debug Server"

@app.route('/test')
def test():
    return jsonify({"status": "OK", "message": "TabPFN test server running"})

@app.route('/quick_test', methods=['GET'])
def quick_test():
    print("=== QUICK TABPFN TEST ===")
    
    if 'gpu_tabpfn' in globals():
        info = f"GPU: {gpu_tabpfn.gpu_info.gpu_name}, Device: {gpu_tabpfn.device}, Ensemble: {gpu_tabpfn.ensemble_size}"
        return jsonify({
            "status": "GPU_READY", 
            "gpu_info": info,
            "message": "TabPFN GPU acceleration ready"
        })
    else:
        return jsonify({"status": "NO_GPU", "message": "TabPFN GPU not initialized"})

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    print("=== SIMULATION REQUEST GELDİ ===")
    
    try:
        # TabPFN GPU test simulation
        print("=== TabPFN GPU SIMULATION TEST ===")
        
        # Örnek veri oluştur
        import numpy as np
        X_test = np.random.rand(100, 5)
        y_test = np.random.randint(0, 2, 100)
        
        print(f"Test data shape: {X_test.shape}")
        
        # GPU TabPFN ile prediction
        if 'gpu_tabpfn' in globals():
            print("=== GPU TabPFN PREDICTION ===")
            
            # Training data oluştur
            X_train = np.random.rand(50, 5)
            y_train = np.random.randint(0, 2, 50)
            
            start_time = time.time()
            result = gpu_tabpfn.predict_with_gpu_optimization(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                task_type='classification'
            )
            prediction_time = time.time() - start_time
            print(f"✅ GPU TabPFN prediction time: {prediction_time:.4f}s")
            
            return jsonify({
                "status": "success",
                "prediction_time": prediction_time,
                "gpu_acceleration": result["gpu_acceleration_factor"],
                "device": result["device_used"],
                "message": "GPU TabPFN simulation completed successfully"
            })
        else:
            print("❌ GPU TabPFN not available")
            return jsonify({"status": "error", "message": "GPU TabPFN not initialized"})
            
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("=== FLASK SERVER BAŞLATIYOR ===")
    app.run(host='0.0.0.0', port=5001, debug=True) 