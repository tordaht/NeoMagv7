"""
NeoMag V7 - Scientific Predictors
Bio-fizik tabanlı bilimsel tahmin modelleri
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BiophysicalPredictor(nn.Module):
    """
    Bio-fizik süreçleri tahmin eden neural network
    """
    def __init__(self, input_features: int = 15, hidden_layers: List[int] = [128, 64, 32]):
        super(BiophysicalPredictor, self).__init__()
        
        layers = []
        prev_size = input_features
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layers for multiple predictions
        self.shared_layers = nn.Sequential(*layers)
        
        # Task-specific heads
        self.energy_head = nn.Linear(prev_size, 1)
        self.fitness_head = nn.Linear(prev_size, 1)
        self.lifetime_head = nn.Linear(prev_size, 1)
        self.mutation_head = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        # Shared representation
        features = self.shared_layers(x)
        
        # Multiple outputs
        energy = torch.sigmoid(self.energy_head(features)) * 150  # 0-150 range
        fitness = torch.sigmoid(self.fitness_head(features))  # 0-1 range
        lifetime = torch.relu(self.lifetime_head(features))  # 0+ range
        mutation_prob = torch.sigmoid(self.mutation_head(features))  # 0-1 range
        
        return {
            'energy': energy,
            'fitness': fitness,
            'lifetime': lifetime,
            'mutation_probability': mutation_prob
        }


class MetabolicRatePredictor:
    """
    Metabolik hız tahmin modeli
    """
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, bacterium_data: Dict) -> np.ndarray:
        """Bakteri verisinden özellik vektörü oluştur"""
        features = [
            bacterium_data.get('energy', 100),
            bacterium_data.get('size', 10),
            bacterium_data.get('speed', 2.0),
            bacterium_data.get('temperature', 37),  # Çevre sıcaklığı
            bacterium_data.get('ph_level', 7.0),
            bacterium_data.get('oxygen_level', 0.8),
            bacterium_data.get('generation', 1),
            bacterium_data.get('food_eaten', 0),
            bacterium_data.get('distance_traveled', 0),
            bacterium_data.get('age', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Dict], metabolic_rates: List[float]):
        """Modeli eğit"""
        X = np.vstack([self.prepare_features(d) for d in training_data])
        y = np.array(metabolic_rates)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Metabolic rate predictor trained on {len(X)} samples")
        
    def predict(self, bacterium_data: Dict) -> float:
        """Metabolik hızı tahmin et"""
        if not self.is_trained:
            # Varsayılan formül kullan
            return self._default_metabolic_rate(bacterium_data)
        
        features = self.prepare_features(bacterium_data)
        features_scaled = self.scaler.transform(features)
        return float(self.model.predict(features_scaled)[0])
    
    def _default_metabolic_rate(self, bacterium_data: Dict) -> float:
        """Varsayılan metabolik hız hesaplama"""
        base_rate = 1.0
        
        # Size effect (larger bacteria have lower metabolic rate per unit mass)
        size_factor = 10 / max(bacterium_data.get('size', 10), 1)
        
        # Temperature effect (Q10 rule)
        temp = bacterium_data.get('temperature', 37)
        temp_factor = 2 ** ((temp - 20) / 10)
        
        # Energy availability
        energy_factor = bacterium_data.get('energy', 100) / 100
        
        return base_rate * size_factor * temp_factor * energy_factor


class GrowthRatePredictor:
    """
    Büyüme hızı tahmin modeli
    """
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def calculate_growth_rate(self, 
                            nutrient_level: float,
                            temperature: float,
                            ph: float,
                            population_density: float) -> float:
        """
        Monod equation ve çevresel faktörlere dayalı büyüme hızı
        """
        # Monod equation for nutrient limitation
        ks = 0.5  # Half-saturation constant
        max_growth_rate = 0.8
        nutrient_factor = nutrient_level / (ks + nutrient_level)
        
        # Temperature effect (Cardinal temperature model)
        t_min, t_opt, t_max = 10, 37, 45
        if temperature < t_min or temperature > t_max:
            temp_factor = 0
        else:
            temp_factor = ((temperature - t_min) * (temperature - t_max)) / \
                         ((t_opt - t_min) * (t_opt - t_max))
            temp_factor = max(0, temp_factor)
        
        # pH effect (Cardinal pH model)
        ph_min, ph_opt, ph_max = 4, 7, 10
        if ph < ph_min or ph > ph_max:
            ph_factor = 0
        else:
            ph_factor = ((ph - ph_min) * (ph - ph_max)) / \
                       ((ph_opt - ph_min) * (ph_opt - ph_max))
            ph_factor = max(0, ph_factor)
        
        # Population density effect (logistic growth)
        carrying_capacity = 1000
        density_factor = 1 - (population_density / carrying_capacity)
        density_factor = max(0, density_factor)
        
        # Combined growth rate
        growth_rate = max_growth_rate * nutrient_factor * temp_factor * \
                     ph_factor * density_factor
        
        return growth_rate


class EvolutionPredictor:
    """
    Evrimsel değişim tahmin modeli
    """
    def __init__(self):
        self.mutation_model = self._build_mutation_model()
        self.selection_model = self._build_selection_model()
        
    def _build_mutation_model(self) -> nn.Module:
        """Mutasyon tahmin modeli"""
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)  # Mutasyon tipi olasılıkları
        )
    
    def _build_selection_model(self) -> nn.Module:
        """Seleksiyon baskısı tahmin modeli"""
        return nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Hayatta kalma olasılığı
        )
    
    def predict_mutation(self, 
                        bacterium_data: Dict,
                        environmental_stress: float = 0.5) -> Dict[str, float]:
        """
        Mutasyon olasılıklarını tahmin et
        """
        # Çevresel stres mutasyon oranını artırır
        base_mutation_rate = 0.001
        stress_multiplier = 1 + (environmental_stress * 5)
        
        mutation_rate = base_mutation_rate * stress_multiplier
        
        # Farklı mutasyon tipleri
        mutation_types = {
            'point_mutation': mutation_rate * 0.7,
            'insertion': mutation_rate * 0.15,
            'deletion': mutation_rate * 0.15,
            'beneficial': mutation_rate * 0.1,  # Faydalı mutasyon
            'neutral': mutation_rate * 0.7,     # Nötr mutasyon
            'deleterious': mutation_rate * 0.2  # Zararlı mutasyon
        }
        
        return mutation_types
    
    def predict_selection_pressure(self, 
                                 population_data: List[Dict],
                                 environment: Dict) -> float:
        """
        Seleksiyon baskısını tahmin et
        """
        if not population_data:
            return 0.5
        
        # Çevresel faktörler
        resource_scarcity = 1 - environment.get('food_availability', 0.5)
        temperature_stress = abs(environment.get('temperature', 37) - 37) / 20
        competition_level = len(population_data) / environment.get('carrying_capacity', 1000)
        
        # Seleksiyon baskısı
        selection_pressure = (resource_scarcity + temperature_stress + competition_level) / 3
        
        return min(1.0, selection_pressure)


class PopulationDynamicsPredictor:
    """
    Popülasyon dinamikleri tahmin modeli
    """
    def __init__(self):
        self.history = []
        self.model = None
        
    def lotka_volterra(self, 
                      prey_count: float,
                      predator_count: float,
                      dt: float = 0.1) -> Tuple[float, float]:
        """
        Lotka-Volterra av-avcı dinamikleri
        """
        # Model parametreleri
        alpha = 1.0    # Av büyüme hızı
        beta = 0.1     # Avlanma hızı
        gamma = 0.075  # Avcı ölüm hızı
        delta = 0.05   # Avcı üreme verimi
        
        # Diferansiyel denklemler
        dprey_dt = alpha * prey_count - beta * prey_count * predator_count
        dpredator_dt = delta * prey_count * predator_count - gamma * predator_count
        
        # Euler integrasyonu
        new_prey = prey_count + dprey_dt * dt
        new_predator = predator_count + dpredator_dt * dt
        
        # Negatif popülasyonu önle
        new_prey = max(0, new_prey)
        new_predator = max(0, new_predator)
        
        return new_prey, new_predator
    
    def predict_population_trend(self, 
                               current_population: int,
                               history: List[int],
                               environment: Dict) -> Dict[str, float]:
        """
        Popülasyon trendini tahmin et
        """
        if len(history) < 3:
            return {
                'next_population': current_population,
                'growth_rate': 0,
                'trend': 'stable'
            }
        
        # Basit trend analizi
        recent_history = history[-10:]
        growth_rates = [(recent_history[i] - recent_history[i-1]) / max(recent_history[i-1], 1) 
                       for i in range(1, len(recent_history))]
        
        avg_growth_rate = np.mean(growth_rates)
        
        # Çevresel kapasite
        carrying_capacity = environment.get('carrying_capacity', 1000)
        capacity_factor = 1 - (current_population / carrying_capacity)
        
        # Tahmin
        adjusted_growth = avg_growth_rate * capacity_factor
        next_population = int(current_population * (1 + adjusted_growth))
        next_population = max(0, min(carrying_capacity, next_population))
        
        # Trend belirleme
        if avg_growth_rate > 0.05:
            trend = 'growing'
        elif avg_growth_rate < -0.05:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'next_population': next_population,
            'growth_rate': avg_growth_rate,
            'trend': trend,
            'capacity_utilization': current_population / carrying_capacity
        }


def save_predictor(predictor: Union[nn.Module, object], filepath: str):
    """Tahmin modelini kaydet"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(predictor, nn.Module):
        torch.save(predictor.state_dict(), filepath)
    else:
        joblib.dump(predictor, filepath)
    
    logger.info(f"Predictor saved to {filepath}")


def load_predictor(predictor_class: type, filepath: str):
    """Tahmin modelini yükle"""
    if not Path(filepath).exists():
        logger.warning(f"Predictor file not found: {filepath}")
        return predictor_class()
    
    predictor = predictor_class()
    
    if isinstance(predictor, nn.Module):
        predictor.load_state_dict(torch.load(filepath))
    else:
        predictor = joblib.load(filepath)
    
    logger.info(f"Predictor loaded from {filepath}")
    return predictor
