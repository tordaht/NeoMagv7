"""
NeoMag V7 - Data Validators
Simülasyon verilerini doğrulama ve temizleme fonksiyonları
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def validate_bacterium_data(bacterium: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Tek bir bakterinin verilerini doğrula ve düzelt
    
    Args:
        bacterium: Bakteri verisi dict
    
    Returns:
        Doğrulanmış bakteri verisi veya None (geçersizse)
    """
    try:
        # Zorunlu alanlar
        required_fields = ['x', 'y', 'energy']
        for field in required_fields:
            if field not in bacterium:
                logger.warning(f"Missing required field: {field}")
                return None
        
        # Koordinat doğrulama
        x = float(bacterium['x'])
        y = float(bacterium['y'])
        if x < 0 or x > 2000 or y < 0 or y > 1000:
            logger.warning(f"Invalid coordinates: ({x}, {y})")
            return None
        
        # Enerji doğrulama
        energy = float(bacterium['energy'])
        if energy < 0 or energy > 200:
            energy = max(0, min(200, energy))
            bacterium['energy'] = energy
        
        # Fitness doğrulama
        if 'fitness' in bacterium:
            fitness = float(bacterium['fitness'])
            bacterium['fitness'] = max(0, min(1, fitness))
        
        # Generation doğrulama
        if 'generation' in bacterium:
            bacterium['generation'] = max(1, int(bacterium['generation']))
        
        # Classification doğrulama
        valid_classes = ['elite', 'veteran', 'strong', 'energetic', 'young', 'basic']
        if 'classification' in bacterium:
            if bacterium['classification'] not in valid_classes:
                bacterium['classification'] = 'basic'
        
        return bacterium
        
    except Exception as e:
        logger.error(f"Error validating bacterium data: {e}")
        return None


def validate_simulation_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tüm simülasyon verilerini doğrula
    
    Args:
        data: Simülasyon verileri
    
    Returns:
        Doğrulanmış simülasyon verileri
    """
    validated_data = {}
    
    # Bakteri listesini doğrula
    if 'bacteria' in data:
        validated_bacteria = []
        for bacterium in data['bacteria']:
            validated = validate_bacterium_data(bacterium)
            if validated:
                validated_bacteria.append(validated)
        validated_data['bacteria'] = validated_bacteria
        logger.info(f"Validated {len(validated_bacteria)} bacteria")
    
    # İstatistikleri doğrula
    if 'stats' in data:
        stats = data['stats']
        validated_stats = {
            'total_bacteria': max(0, int(stats.get('total_bacteria', 0))),
            'avg_fitness': max(0, min(1, float(stats.get('avg_fitness', 0.5)))),
            'avg_energy': max(0, float(stats.get('avg_energy', 100))),
            'generation': max(1, int(stats.get('generation', 1))),
            'fps': max(0, float(stats.get('fps', 60)))
        }
        validated_data['stats'] = validated_stats
    
    # Food listesini doğrula
    if 'food' in data:
        validated_food = []
        for food in data['food']:
            if isinstance(food, dict) and 'x' in food and 'y' in food:
                x = float(food['x'])
                y = float(food['y'])
                if 0 <= x <= 2000 and 0 <= y <= 1000:
                    validated_food.append({'x': x, 'y': y})
        validated_data['food'] = validated_food
    
    return validated_data


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simülasyon konfigürasyonunu doğrula
    
    Args:
        config: Konfigürasyon dict
    
    Returns:
        Doğrulanmış konfigürasyon
    """
    defaults = {
        'bacteria_count': 30,
        'food_count': 100,
        'canvas_width': 1200,
        'canvas_height': 600,
        'mutation_rate': 0.05,
        'reproduction_threshold': 80,
        'max_bacteria': 500,
        'simulation_speed': 1.0
    }
    
    validated_config = {}
    
    for key, default_value in defaults.items():
        if key in config:
            try:
                # Değer tipini koru
                if isinstance(default_value, int):
                    value = int(config[key])
                elif isinstance(default_value, float):
                    value = float(config[key])
                else:
                    value = config[key]
                
                # Sınırları kontrol et
                if key == 'bacteria_count':
                    value = max(1, min(100, value))
                elif key == 'food_count':
                    value = max(0, min(500, value))
                elif key == 'mutation_rate':
                    value = max(0, min(1, value))
                elif key == 'simulation_speed':
                    value = max(0.1, min(10, value))
                
                validated_config[key] = value
            except:
                validated_config[key] = default_value
        else:
            validated_config[key] = default_value
    
    return validated_config


def sanitize_user_input(input_str: str) -> str:
    """
    Kullanıcı girdisini güvenlik için temizle
    
    Args:
        input_str: Kullanıcı girdisi
    
    Returns:
        Temizlenmiş girdi
    """
    # HTML/JavaScript injection'ı önle
    dangerous_chars = ['<', '>', '"', "'", '&', '\\', '{', '}', '(', ')', ';']
    sanitized = input_str
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    # Maksimum uzunluk
    sanitized = sanitized[:1000]
    
    return sanitized.strip()


def validate_api_request(request_data: Dict[str, Any], endpoint: str) -> Optional[str]:
    """
    API isteğini doğrula
    
    Args:
        request_data: İstek verisi
        endpoint: API endpoint adı
    
    Returns:
        Hata mesajı (varsa) veya None
    """
    # Endpoint'e göre doğrulama
    if endpoint == 'start':
        if 'bacteria_count' in request_data:
            count = request_data['bacteria_count']
            if not isinstance(count, int) or count < 1 or count > 100:
                return "bacteria_count must be between 1 and 100"
    
    elif endpoint == 'add_bacteria':
        if 'count' in request_data:
            count = request_data['count']
            if not isinstance(count, int) or count < 1 or count > 50:
                return "count must be between 1 and 50"
    
    elif endpoint == 'ai_question':
        if 'question' not in request_data:
            return "question field is required"
        if len(request_data['question']) > 500:
            return "question too long (max 500 chars)"
    
    return None
