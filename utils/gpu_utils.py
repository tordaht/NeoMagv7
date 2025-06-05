"""
NeoMag V7 - GPU Utilities
GPU kullanımı için yardımcı fonksiyonlar
"""

import torch
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, any]:
    """
    GPU kullanılabilirliğini kontrol et
    
    Returns:
        GPU bilgileri içeren dict
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': None,
        'device_count': 0,
        'devices': [],
        'current_device': None,
        'allocated_memory': 0,
        'cached_memory': 0
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        
        for i in range(info['device_count']):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'capability': torch.cuda.get_device_capability(i),
                'total_memory': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i)
            }
            info['devices'].append(device_info)
        
        info['allocated_memory'] = torch.cuda.memory_allocated()
        info['cached_memory'] = torch.cuda.memory_reserved()
    
    return info


def get_optimal_device() -> torch.device:
    """
    En uygun cihazı seç (GPU varsa GPU, yoksa CPU)
    
    Returns:
        torch.device nesnesi
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (CUDA not available)")
    
    return device


def move_to_device(data: any, device: Optional[torch.device] = None) -> any:
    """
    Veriyi belirtilen cihaza taşı
    
    Args:
        data: Taşınacak veri (tensor, list, dict vb.)
        device: Hedef cihaz (None ise optimal cihaz seçilir)
    
    Returns:
        Cihaza taşınmış veri
    """
    if device is None:
        device = get_optimal_device()
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    else:
        return data


def optimize_batch_size(model_memory_mb: float, available_memory_mb: Optional[float] = None) -> int:
    """
    Model bellek kullanımına göre optimal batch size hesapla
    
    Args:
        model_memory_mb: Model başına MB cinsinden bellek kullanımı
        available_memory_mb: Kullanılabilir bellek (None ise otomatik tespit)
    
    Returns:
        Önerilen batch size
    """
    if available_memory_mb is None:
        if torch.cuda.is_available():
            available_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            # Güvenlik için %80'ini kullan
            available_memory_mb *= 0.8
        else:
            # CPU için varsayılan 4GB
            available_memory_mb = 4096
    
    # Batch size hesapla (minimum 1, maksimum 512)
    batch_size = int(available_memory_mb / model_memory_mb)
    batch_size = max(1, min(512, batch_size))
    
    logger.info(f"Optimal batch size: {batch_size} (available memory: {available_memory_mb:.0f}MB)")
    return batch_size


def clear_gpu_cache():
    """
    GPU önbelleğini temizle
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def monitor_memory_usage() -> Dict[str, float]:
    """
    Bellek kullanımını izle
    
    Returns:
        Bellek kullanım bilgileri (MB cinsinden)
    """
    usage = {
        'cpu_memory_used': 0,
        'gpu_memory_allocated': 0,
        'gpu_memory_cached': 0,
        'gpu_memory_free': 0
    }
    
    # CPU bellek kullanımı
    try:
        import psutil
        process = psutil.Process()
        usage['cpu_memory_used'] = process.memory_info().rss / 1024 / 1024
    except ImportError:
        logger.warning("psutil not installed, cannot monitor CPU memory")
    
    # GPU bellek kullanımı
    if torch.cuda.is_available():
        usage['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024
        usage['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024 / 1024
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        usage['gpu_memory_free'] = total_memory - usage['gpu_memory_cached']
    
    return usage


def parallel_compute(func, data_list: List, batch_size: Optional[int] = None) -> List:
    """
    GPU üzerinde paralel hesaplama yap
    
    Args:
        func: Uygulanacak fonksiyon
        data_list: Veri listesi
        batch_size: Batch boyutu (None ise otomatik)
    
    Returns:
        Sonuç listesi
    """
    device = get_optimal_device()
    
    if batch_size is None:
        batch_size = optimize_batch_size(10)  # Varsayılan 10MB per item
    
    results = []
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        batch_tensor = move_to_device(batch, device)
        
        with torch.no_grad():
            batch_results = func(batch_tensor)
        
        # CPU'ya geri taşı
        if isinstance(batch_results, torch.Tensor):
            batch_results = batch_results.cpu().numpy()
        
        results.extend(batch_results)
    
    return results


def setup_gpu_environment():
    """
    GPU ortamını başlangıçta yapılandır
    """
    # Deterministik sonuçlar için
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # CUDNN optimizasyonları
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
        logger.info("GPU environment configured successfully")
    
    # GPU bilgilerini logla
    gpu_info = check_gpu_availability()
    logger.info(f"GPU Status: {gpu_info}")
    
    return gpu_info
