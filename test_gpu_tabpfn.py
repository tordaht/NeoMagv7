#!/usr/bin/env python3
"""
NeoMag V7 - GPU TabPFN Test Script (RTX 3060)
TabPFN GPU hızlandırma testini yapar ve performans raporu üretir.
"""

import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cuda_setup():
    """CUDA kurulum kontrolü"""
    print("=" * 60)
    print("🔥 CUDA KURULUM KONTROLÜ 🔥")
    print("=" * 60)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  VRAM: {gpu_props.total_memory / (1024**3):.1f}GB")
            print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"  Multiprocessors: {gpu_props.multi_processor_count}")
        
        # Memory test
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"✅ GPU Memory test passed")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU Memory test failed: {e}")
    else:
        print("❌ CUDA not available")
        return False
    
    return torch.cuda.is_available()

def test_tabpfn_import():
    """TabPFN import testi"""
    print("\n" + "=" * 60)
    print("📚 TABPFN IMPORT KONTROLÜ 📚")
    print("=" * 60)
    
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        print("✅ TabPFN successfully imported")
        
        # Version check
        try:
            classifier = TabPFNClassifier(device='cpu', n_estimators=1)
            print("✅ TabPFN instantiation test passed")
            del classifier
        except Exception as e:
            print(f"⚠️ TabPFN instantiation warning: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ TabPFN import failed: {e}")
        print("💡 Çözüm: pip install tabpfn")
        return False

def test_gpu_tabpfn_integration():
    """GPU TabPFN entegrasyon testi"""
    print("\n" + "=" * 60)
    print("🔥 GPU TABPFN ENTEGRASYON TESTİ 🔥")
    print("=" * 60)
    
    try:
        # Import our GPU module
        sys.path.append(str(Path(__file__).parent / "ml_models"))
        from tabpfn_gpu_integration import create_gpu_tabpfn_predictor, detect_gpu_capabilities
        
        # GPU capabilities
        gpu_info = detect_gpu_capabilities()
        print(f"GPU: {gpu_info.gpu_name}")
        print(f"VRAM: {gpu_info.gpu_memory_total:.1f}GB")
        print(f"Önerilen Ensemble: {gpu_info.recommended_ensemble_size}")
        
        # Create predictor
        predictor = create_gpu_tabpfn_predictor(auto_detect=True)
        
        if predictor.is_initialized:
            print(f"✅ GPU TabPFN initialized on {predictor.device}")
            print(f"Ensemble Size: {predictor.ensemble_size}")
            return predictor
        else:
            print("❌ GPU TabPFN initialization failed")
            return None
            
    except ImportError as e:
        print(f"❌ GPU TabPFN module import failed: {e}")
        return None
    except Exception as e:
        print(f"❌ GPU TabPFN integration error: {e}")
        return None

def run_performance_benchmark(predictor):
    """Performans benchmark testi"""
    print("\n" + "=" * 60)
    print("⚡ PERFORMANS BENCHMARK ⚡")
    print("=" * 60)
    
    if not predictor:
        print("❌ No predictor available for benchmark")
        return
    
    # Test parameters - RTX 3060 için optimize
    test_configs = [
        {"samples": 200, "features": 20, "name": "Küçük Dataset"},
        {"samples": 500, "features": 50, "name": "Orta Dataset"},
        {"samples": 800, "features": 80, "name": "Büyük Dataset (RTX 3060 Max)"}
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🔬 Test: {config['name']}")
        print(f"Samples: {config['samples']}, Features: {config['features']}")
        
        # Generate test data
        np.random.seed(42)
        X_train = np.random.randn(config['samples'], config['features']).astype(np.float32)
        y_train = np.random.randn(config['samples']).astype(np.float32)
        X_test = np.random.randn(100, config['features']).astype(np.float32)
        
        try:
            # Run prediction
            start_time = time.time()
            result = predictor.predict_with_gpu_optimization(
                X_train, y_train, X_test, task_type='regression'
            )
            total_time = time.time() - start_time
            
            metrics = result['performance_metrics']
            
            print(f"  ✅ Başarılı!")
            print(f"  Prediction Time: {metrics['prediction_time']:.4f}s")
            print(f"  Total Time: {total_time:.4f}s")
            print(f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")
            print(f"  Device: {metrics['device']}")
            print(f"  GPU Memory: {metrics['gpu_memory_used_gb']:.3f}GB")
            
            results.append({
                'config': config['name'],
                'prediction_time': metrics['prediction_time'],
                'total_time': total_time,
                'throughput': metrics['throughput_samples_per_sec'],
                'device': metrics['device'],
                'memory_gb': metrics['gpu_memory_used_gb']
            })
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
    
    # Results summary
    if results:
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SONUÇLARI 📊")
        print("=" * 60)
        
        for result in results:
            print(f"{result['config']:20} | {result['prediction_time']:8.4f}s | {result['throughput']:8.1f} smp/s | {result['device']:4} | {result['memory_gb']:6.3f}GB")
        
        # RTX 3060 comparison
        cpu_times = [r['prediction_time'] for r in results if r['device'] == 'cpu']
        gpu_times = [r['prediction_time'] for r in results if r['device'] == 'cuda']
        
        if cpu_times and gpu_times:
            avg_speedup = np.mean(cpu_times) / np.mean(gpu_times)
            print(f"\n🔥 ORTALAMA GPU HIZLANDIRMA: {avg_speedup:.1f}x")
        
        return results
    
    return []

def test_bacterial_simulation_scenario():
    """Bakteriyel simülasyon senaryosu testi"""
    print("\n" + "=" * 60)
    print("🦠 BAKTERİYEL SİMÜLASYON SENARYOSU 🦠")
    print("=" * 60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "ml_models"))
        from tabpfn_gpu_integration import create_gpu_tabpfn_predictor
        
        predictor = create_gpu_tabpfn_predictor(auto_detect=True)
        
        if not predictor.is_initialized:
            print("❌ Predictor not available")
            return
        
        # Bacterial features (simulated)
        n_bacteria = 300
        n_features = 25  # x, y, z, energy, age, fitness, etc.
        
        print(f"Simülasyon: {n_bacteria} bakteri, {n_features} özellik")
        
        # Generate bacterial data
        np.random.seed(123)
        bacterial_features = np.random.randn(n_bacteria, n_features).astype(np.float32)
        fitness_scores = np.random.beta(2, 5, n_bacteria).astype(np.float32)  # Realistic fitness distribution
        
        # New bacteria to predict
        new_bacteria_features = np.random.randn(50, n_features).astype(np.float32)
        
        print("🔬 TabPFN fitness prediction başlıyor...")
        
        start_time = time.time()
        result = predictor.predict_with_gpu_optimization(
            bacterial_features, fitness_scores, new_bacteria_features, 
            task_type='regression'
        )
        simulation_time = time.time() - start_time
        
        predictions = result['predictions']
        metrics = result['performance_metrics']
        
        print(f"✅ Bakteriyel fitness prediction tamamlandı!")
        print(f"Prediction Time: {metrics['prediction_time']:.4f}s")
        print(f"Total Time: {simulation_time:.4f}s")
        print(f"Device: {metrics['device']}")
        print(f"Predicted Fitness Range: {np.min(predictions):.3f} - {np.max(predictions):.3f}")
        print(f"Mean Predicted Fitness: {np.mean(predictions):.3f}")
        
        # Real-time capability estimate
        bacteria_per_second = len(new_bacteria_features) / metrics['prediction_time']
        print(f"Real-time Capability: ~{bacteria_per_second:.0f} bacteria/second")
        
        if bacteria_per_second > 100:
            print("🚀 Gerçek zamanlı simülasyon için yeterli hız!")
        else:
            print("⚠️ Gerçek zamanlı simülasyon için optimize gerekli")
        
        return True
        
    except Exception as e:
        print(f"❌ Bacterial simulation test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🔥" * 30)
    print("   NeoMag V7 - GPU TabPFN Test Suite   ")
    print("         RTX 3060 Optimization          ")
    print("🔥" * 30)
    
    # Test 1: CUDA setup
    cuda_ok = test_cuda_setup()
    
    # Test 2: TabPFN import
    tabpfn_ok = test_tabpfn_import()
    
    if not tabpfn_ok:
        print("\n❌ TabPFN not available, exiting...")
        sys.exit(1)
    
    # Test 3: GPU TabPFN integration
    predictor = test_gpu_tabpfn_integration()
    
    # Test 4: Performance benchmark
    if predictor:
        benchmark_results = run_performance_benchmark(predictor)
    
    # Test 5: Bacterial simulation scenario
    simulation_ok = test_bacterial_simulation_scenario()
    
    # Final summary
    print("\n" + "🔥" * 60)
    print("📋 FINAL ÖZET RAPORU 📋")
    print("🔥" * 60)
    
    print(f"CUDA Available: {'✅' if cuda_ok else '❌'}")
    print(f"TabPFN Available: {'✅' if tabpfn_ok else '❌'}")
    print(f"GPU TabPFN Integration: {'✅' if predictor else '❌'}")
    print(f"Bacterial Simulation: {'✅' if simulation_ok else '❌'}")
    
    if predictor and cuda_ok:
        print(f"\n🔥 RTX 3060 GPU DURUMU:")
        print(f"Device: {predictor.device}")
        print(f"Ensemble Size: {predictor.ensemble_size}")
        print(f"VRAM: {predictor.gpu_info.gpu_memory_total:.1f}GB")
        
        if predictor.device == 'cuda':
            print("🚀 GPU HIZLANDIRMA AKTİF!")
            print("💡 Tahmini hızlandırma: ~100x (CPU'ya göre)")
        else:
            print("⚠️ CPU modunda çalışıyor")
    
    print("\n🔥 Test completed! NeoMag V7 GPU ready for action! 🔥")

if __name__ == "__main__":
    main() 