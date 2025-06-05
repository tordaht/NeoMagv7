"""
NeoMag V7 - Data Loaders
Simülasyon verilerini farklı formatlardan yükleme fonksiyonları
"""

import json
import csv
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_from_csv(filename):
    """
    CSV dosyasından bakteri verilerini yükle
    
    Args:
        filename: CSV dosya yolu
    
    Returns:
        Bakteri listesi (dict formatında)
    """
    try:
        bacteria_data = []
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Sayısal değerleri dönüştür
                bacterium = {
                    'id': int(row.get('id', 0)),
                    'x': float(row.get('x', 0)),
                    'y': float(row.get('y', 0)),
                    'energy': float(row.get('energy', 100)),
                    'fitness': float(row.get('fitness', 0.5)),
                    'generation': int(row.get('generation', 1)),
                    'classification': row.get('classification', 'basic'),
                    'speed': float(row.get('speed', 2.0)),
                    'size': int(row.get('size', 10)),
                    'lifetime': int(row.get('lifetime', 0)),
                    'mutation_count': int(row.get('mutation_count', 0)),
                    'food_eaten': int(row.get('food_eaten', 0)),
                    'distance_traveled': float(row.get('distance_traveled', 0))
                }
                bacteria_data.append(bacterium)
        
        logger.info(f"Loaded {len(bacteria_data)} bacteria from {filename}")
        return bacteria_data
        
    except Exception as e:
        logger.error(f"Error loading from CSV: {e}")
        return []


def load_from_json(filename):
    """
    JSON dosyasından simülasyon verilerini yükle
    
    Args:
        filename: JSON dosya yolu
    
    Returns:
        Simülasyon verileri dict
    """
    try:
        with open(filename, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
        
        # Veri formatını kontrol et
        if 'data' in data:
            simulation_data = data['data']
        else:
            simulation_data = data
        
        logger.info(f"Loaded simulation data from {filename}")
        return simulation_data
        
    except Exception as e:
        logger.error(f"Error loading from JSON: {e}")
        return {}


def load_from_excel(filename):
    """
    Excel dosyasından simülasyon verilerini yükle
    
    Args:
        filename: Excel dosya yolu
    
    Returns:
        Simülasyon verileri dict
    """
    try:
        simulation_data = {}
        
        # Bakteri verilerini yükle
        if 'Bacteria' in pd.ExcelFile(filename).sheet_names:
            df_bacteria = pd.read_excel(filename, sheet_name='Bacteria')
            simulation_data['bacteria'] = df_bacteria.to_dict('records')
        
        # İstatistikleri yükle
        if 'Statistics' in pd.ExcelFile(filename).sheet_names:
            df_stats = pd.read_excel(filename, sheet_name='Statistics')
            if not df_stats.empty:
                simulation_data['stats'] = df_stats.iloc[0].to_dict()
        
        # Tarihçeyi yükle
        if 'History' in pd.ExcelFile(filename).sheet_names:
            df_history = pd.read_excel(filename, sheet_name='History')
            simulation_data['history'] = df_history.to_dict('records')
        
        logger.info(f"Loaded simulation data from Excel: {filename}")
        return simulation_data
        
    except Exception as e:
        logger.error(f"Error loading from Excel: {e}")
        return {}


def load_simulation_state(filename):
    """
    Kaydedilmiş simülasyon durumunu yükle
    
    Args:
        filename: Durum dosyası yolu
    
    Returns:
        Simülasyon durumu dict
    """
    try:
        file_path = Path(filename)
        
        if file_path.suffix == '.json':
            return load_from_json(filename)
        elif file_path.suffix == '.csv':
            return {'bacteria': load_from_csv(filename)}
        elif file_path.suffix in ['.xlsx', '.xls']:
            return load_from_excel(filename)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return {}
            
    except Exception as e:
        logger.error(f"Error loading simulation state: {e}")
        return {}


def load_configuration(config_file='config.json'):
    """
    Simülasyon konfigürasyonunu yükle
    
    Args:
        config_file: Konfigürasyon dosyası yolu
    
    Returns:
        Konfigürasyon dict
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_file}")
        return config
        
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_file}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}
