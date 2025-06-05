"""
NeoMag V7 - Main Simulation Engine
Bio-fizik tabanlı ana simülasyon motoru
"""

import numpy as np
import random
import threading
import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Simülasyon konfigürasyonu"""
    width: int = 1200
    height: int = 600
    initial_bacteria: int = 30
    initial_food: int = 100
    max_bacteria: int = 500
    min_bacteria: int = 5
    food_spawn_rate: float = 0.1
    mutation_rate: float = 0.05
    reproduction_threshold: float = 80.0
    energy_consumption_rate: float = 0.5
    food_energy_value: float = 20.0
    collision_radius: float = 15.0
    simulation_speed: float = 1.0
    enable_pheromones: bool = True
    enable_evolution: bool = True
    enable_ai_analysis: bool = True


@dataclass
class SimulationState:
    """Simülasyon durumu"""
    running: bool = False
    paused: bool = False
    step: int = 0
    generation: int = 1
    total_bacteria: int = 0
    total_food: int = 0
    avg_fitness: float = 0.0
    avg_energy: float = 0.0
    births: int = 0
    deaths: int = 0
    mutations: int = 0
    fps: float = 60.0
    last_update: float = field(default_factory=time.time)


class BiophysicalSimulator:
    """
    Bio-fizik tabanlı simülasyon motoru
    """
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.state = SimulationState()
        
        # Simülasyon bileşenleri
        self.bacteria: List[Dict] = []
        self.food: List[Dict] = []
        self.pheromone_map = np.zeros((self.config.height // 10, self.config.width // 10))
        self.history = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'on_step': [],
            'on_birth': [],
            'on_death': [],
            'on_mutation': [],
            'on_generation': []
        }
        
        # İstatistikler
        self.statistics = {
            'population_history': [],
            'fitness_history': [],
            'energy_history': [],
            'generation_times': []
        }
        
    def initialize(self):
        """Simülasyonu başlat"""
        with self.lock:
            logger.info("Initializing simulation...")
            
            # Bakterileri oluştur
            self.bacteria = []
            for i in range(self.config.initial_bacteria):
                bacterium = self._create_bacterium(
                    x=random.uniform(50, self.config.width - 50),
                    y=random.uniform(50, self.config.height - 50)
                )
                self.bacteria.append(bacterium)
            
            # Yiyecekleri oluştur
            self.food = []
            for _ in range(self.config.initial_food):
                self.food.append(self._create_food())
            
            # Durumu güncelle
            self.state.total_bacteria = len(self.bacteria)
            self.state.total_food = len(self.food)
            self.state.step = 0
            self.state.generation = 1
            
            # Feromon haritasını sıfırla
            self.pheromone_map.fill(0)
            
            logger.info(f"Simulation initialized with {self.state.total_bacteria} bacteria and {self.state.total_food} food")
    
    def _create_bacterium(self, x: float, y: float, parent: Dict = None) -> Dict:
        """Yeni bakteri oluştur"""
        bacterium = {
            'id': int(time.time() * 1000000) + random.randint(0, 999),
            'x': x,
            'y': y,
            'vx': random.uniform(-1, 1),
            'vy': random.uniform(-1, 1),
            'energy': 100.0,
            'size': 10 + random.randint(-2, 2),
            'speed': 2.0 + random.uniform(-0.5, 0.5),
            'generation': parent['generation'] + 1 if parent else 1,
            'age': 0,
            'lifetime': 0,
            'food_eaten': 0,
            'distance_traveled': 0.0,
            'mutation_count': 0,
            'dna': self._generate_dna(parent),
            'fitness': 0.5,
            'reproduction_cooldown': 0,
            'pheromone_sensitivity': random.uniform(0.5, 1.5),
            'metabolic_rate': 1.0 + random.uniform(-0.2, 0.2)
        }
        
        # Fitness hesapla
        bacterium['fitness'] = self._calculate_fitness(bacterium)
        
        # Sınıflandır
        bacterium['classification'] = self._classify_bacterium(bacterium)
        
        return bacterium
    
    def _generate_dna(self, parent: Dict = None) -> List[float]:
        """DNA oluştur veya mutasyona uğrat"""
        if parent is None:
            # Rastgele DNA
            return [random.random() for _ in range(10)]
        else:
            # Parent DNA'sını kopyala ve mutasyona uğrat
            dna = parent['dna'].copy()
            
            if random.random() < self.config.mutation_rate:
                # Mutasyon
                mutation_index = random.randint(0, len(dna) - 1)
                dna[mutation_index] += random.uniform(-0.1, 0.1)
                dna[mutation_index] = max(0, min(1, dna[mutation_index]))
                self.state.mutations += 1
                
                # Callback
                self._trigger_callback('on_mutation', parent, dna)
            
            return dna
    
    def _calculate_fitness(self, bacterium: Dict) -> float:
        """Fitness değerini hesapla"""
        # DNA tabanlı fitness
        base_fitness = np.mean(bacterium['dna'])
        
        # Enerji etkisi
        energy_factor = bacterium['energy'] / 100.0
        
        # Yaş etkisi
        age_factor = 1.0 - (bacterium['age'] / 1000.0)
        age_factor = max(0.1, age_factor)
        
        # Size etkisi
        size_factor = bacterium['size'] / 10.0
        
        # Kombine fitness
        fitness = base_fitness * 0.4 + energy_factor * 0.3 + age_factor * 0.2 + size_factor * 0.1
        
        return max(0, min(1, fitness))
    
    def _classify_bacterium(self, bacterium: Dict) -> str:
        """Bakteriyi sınıflandır"""
        fitness = bacterium['fitness']
        generation = bacterium['generation']
        
        if fitness > 0.9 and generation > 5:
            return 'elite'
        elif fitness > 0.8 and generation > 3:
            return 'veteran'
        elif fitness > 0.7:
            return 'strong'
        elif fitness > 0.6 and bacterium['energy'] > 80:
            return 'energetic'
        elif generation <= 2:
            return 'young'
        else:
            return 'basic'
    
    def _create_food(self) -> Dict:
        """Yeni yiyecek oluştur"""
        return {
            'x': random.uniform(20, self.config.width - 20),
            'y': random.uniform(20, self.config.height - 20),
            'energy': self.config.food_energy_value,
            'type': random.choice(['standard', 'nutrient_rich', 'toxic'])
        }
    
    def step(self):
        """Simülasyon adımı"""
        if not self.state.running or self.state.paused:
            return
        
        with self.lock:
            start_time = time.time()
            
            # Bakterileri güncelle
            self._update_bacteria()
            
            # Yiyecekleri güncelle
            self._update_food()
            
            # Feromon haritasını güncelle
            if self.config.enable_pheromones:
                self._update_pheromones()
            
            # İstatistikleri güncelle
            self._update_statistics()
            
            # Jenerasyon kontrolü
            self._check_generation()
            
            # Durum güncelle
            self.state.step += 1
            self.state.last_update = time.time()
            
            # FPS hesapla
            elapsed = time.time() - start_time
            self.state.fps = 1.0 / max(elapsed, 0.001)
            
            # Callback
            self._trigger_callback('on_step', self.state)
            
            # History kaydet
            self._record_history()
    
    def _update_bacteria(self):
        """Bakterileri güncelle"""
        new_bacteria = []
        
        for bacterium in self.bacteria:
            # Enerji tüketimi
            bacterium['energy'] -= self.config.energy_consumption_rate * bacterium['metabolic_rate']
            bacterium['age'] += 1
            bacterium['lifetime'] += 1
            
            # Hareket
            self._move_bacterium(bacterium)
            
            # Yiyecek kontrolü
            self._check_food_collision(bacterium)
            
            # Üreme kontrolü
            if self._can_reproduce(bacterium):
                child = self._reproduce(bacterium)
                if child:
                    new_bacteria.append(child)
            
            # Ölüm kontrolü
            if bacterium['energy'] <= 0 or bacterium['age'] > 2000:
                self.state.deaths += 1
                self._trigger_callback('on_death', bacterium)
                continue
            
            # Fitness güncelle
            bacterium['fitness'] = self._calculate_fitness(bacterium)
            bacterium['classification'] = self._classify_bacterium(bacterium)
        
        # Yeni bakterileri ekle
        self.bacteria.extend(new_bacteria)
        
        # Ölüleri temizle
        self.bacteria = [b for b in self.bacteria if b['energy'] > 0]
        
        # Popülasyon limitleri
        if len(self.bacteria) > self.config.max_bacteria:
            # En düşük fitness'a sahip olanları öldür
            self.bacteria.sort(key=lambda b: b['fitness'], reverse=True)
            self.bacteria = self.bacteria[:self.config.max_bacteria]
    
    def _move_bacterium(self, bacterium: Dict):
        """Bakteriyi hareket ettir"""
        # En yakın yiyeceği bul
        if self.food:
            nearest_food = min(self.food, 
                             key=lambda f: self._distance(bacterium, f))
            
            # Yiyeceğe doğru hareket
            dx = nearest_food['x'] - bacterium['x']
            dy = nearest_food['y'] - bacterium['y']
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 0:
                # Normalize et ve hız uygula
                bacterium['vx'] = (dx / dist) * bacterium['speed']
                bacterium['vy'] = (dy / dist) * bacterium['speed']
        
        # Feromon takibi
        if self.config.enable_pheromones:
            pheromone_influence = self._get_pheromone_influence(bacterium)
            bacterium['vx'] += pheromone_influence[0] * bacterium['pheromone_sensitivity']
            bacterium['vy'] += pheromone_influence[1] * bacterium['pheromone_sensitivity']
        
        # Pozisyonu güncelle
        bacterium['x'] += bacterium['vx'] * self.config.simulation_speed
        bacterium['y'] += bacterium['vy'] * self.config.simulation_speed
        
        # Sınırları kontrol et
        bacterium['x'] = max(10, min(self.config.width - 10, bacterium['x']))
        bacterium['y'] = max(10, min(self.config.height - 10, bacterium['y']))
        
        # Mesafe takibi
        bacterium['distance_traveled'] += np.sqrt(bacterium['vx']**2 + bacterium['vy']**2)
    
    def _distance(self, obj1: Dict, obj2: Dict) -> float:
        """İki nesne arasındaki mesafe"""
        return np.sqrt((obj1['x'] - obj2['x'])**2 + (obj1['y'] - obj2['y'])**2)
    
    def _check_food_collision(self, bacterium: Dict):
        """Yiyecek çarpışması kontrolü"""
        eaten_food = []
        
        for food in self.food:
            if self._distance(bacterium, food) < self.config.collision_radius:
                # Yiyeceği ye
                energy_gain = food['energy']
                
                # Toksik yiyecek kontrolü
                if food['type'] == 'toxic':
                    energy_gain *= -0.5  # Enerji kaybı
                elif food['type'] == 'nutrient_rich':
                    energy_gain *= 1.5  # Ekstra enerji
                
                bacterium['energy'] += energy_gain
                bacterium['energy'] = min(150, bacterium['energy'])
                bacterium['food_eaten'] += 1
                
                eaten_food.append(food)
        
        # Yenilen yiyecekleri kaldır
        for food in eaten_food:
            self.food.remove(food)
    
    def _can_reproduce(self, bacterium: Dict) -> bool:
        """Üreme kontrolü"""
        return (bacterium['energy'] >= self.config.reproduction_threshold and
                bacterium['reproduction_cooldown'] <= 0 and
                len(self.bacteria) < self.config.max_bacteria)
    
    def _reproduce(self, parent: Dict) -> Optional[Dict]:
        """Bakteriyi üret"""
        # Enerji maliyeti
        parent['energy'] *= 0.5
        parent['reproduction_cooldown'] = 50
        
        # Yavru oluştur
        child = self._create_bacterium(
            x=parent['x'] + random.uniform(-20, 20),
            y=parent['y'] + random.uniform(-20, 20),
            parent=parent
        )
        
        self.state.births += 1
        self._trigger_callback('on_birth', parent, child)
        
        return child
    
    def _update_food(self):
        """Yiyecekleri güncelle"""
        # Yeni yiyecek ekle
        if random.random() < self.config.food_spawn_rate:
            self.food.append(self._create_food())
        
        # Maksimum yiyecek sayısını kontrol et
        if len(self.food) > self.config.initial_food * 2:
            self.food = self.food[:self.config.initial_food * 2]
        
        self.state.total_food = len(self.food)
    
    def _update_pheromones(self):
        """Feromon haritasını güncelle"""
        # Evaporation
        self.pheromone_map *= 0.99
        
        # Bakterilerin feromon salgılaması
        for bacterium in self.bacteria:
            grid_x = int(bacterium['x'] // 10)
            grid_y = int(bacterium['y'] // 10)
            
            if 0 <= grid_x < self.pheromone_map.shape[1] and 0 <= grid_y < self.pheromone_map.shape[0]:
                # Fitness'a göre feromon sal
                self.pheromone_map[grid_y, grid_x] += bacterium['fitness'] * 0.1
        
        # Maksimum değeri sınırla
        self.pheromone_map = np.clip(self.pheromone_map, 0, 10)
    
    def _get_pheromone_influence(self, bacterium: Dict) -> Tuple[float, float]:
        """Feromon etkisini hesapla"""
        grid_x = int(bacterium['x'] // 10)
        grid_y = int(bacterium['y'] // 10)
        
        # Gradient hesapla
        dx, dy = 0, 0
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = grid_x + i, grid_y + j
                
                if 0 <= nx < self.pheromone_map.shape[1] and 0 <= ny < self.pheromone_map.shape[0]:
                    strength = self.pheromone_map[ny, nx]
                    dx += i * strength
                    dy += j * strength
        
        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        return dx, dy
    
    def _update_statistics(self):
        """İstatistikleri güncelle"""
        if self.bacteria:
            self.state.total_bacteria = len(self.bacteria)
            self.state.avg_fitness = np.mean([b['fitness'] for b in self.bacteria])
            self.state.avg_energy = np.mean([b['energy'] for b in self.bacteria])
        else:
            self.state.total_bacteria = 0
            self.state.avg_fitness = 0
            self.state.avg_energy = 0
    
    def _check_generation(self):
        """Jenerasyon kontrolü"""
        if self.bacteria:
            max_generation = max(b['generation'] for b in self.bacteria)
            if max_generation > self.state.generation:
                self.state.generation = max_generation
                self._trigger_callback('on_generation', self.state.generation)
                logger.info(f"New generation reached: {self.state.generation}")
    
    def _record_history(self):
        """Tarihçe kaydet"""
        record = {
            'step': self.state.step,
            'time': time.time(),
            'total_bacteria': self.state.total_bacteria,
            'total_food': self.state.total_food,
            'avg_fitness': self.state.avg_fitness,
            'avg_energy': self.state.avg_energy,
            'generation': self.state.generation,
            'births': self.state.births,
            'deaths': self.state.deaths,
            'mutations': self.state.mutations
        }
        
        self.history.append(record)
        
        # İstatistikleri güncelle
        self.statistics['population_history'].append(self.state.total_bacteria)
        self.statistics['fitness_history'].append(self.state.avg_fitness)
        self.statistics['energy_history'].append(self.state.avg_energy)
    
    def _trigger_callback(self, event: str, *args):
        """Callback tetikle"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Error in callback {event}: {e}")
    
    def register_callback(self, event: str, callback: Callable):
        """Callback kaydet"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event: {event}")
    
    def start(self):
        """Simülasyonu başlat"""
        with self.lock:
            if not self.bacteria:
                self.initialize()
            
            self.state.running = True
            self.state.paused = False
            logger.info("Simulation started")
    
    def pause(self):
        """Simülasyonu duraklat"""
        with self.lock:
            self.state.paused = True
            logger.info("Simulation paused")
    
    def resume(self):
        """Simülasyonu devam ettir"""
        with self.lock:
            self.state.paused = False
            logger.info("Simulation resumed")
    
    def stop(self):
        """Simülasyonu durdur"""
        with self.lock:
            self.state.running = False
            self.state.paused = False
            logger.info("Simulation stopped")
    
    def reset(self):
        """Simülasyonu sıfırla"""
        with self.lock:
            self.stop()
            self.state = SimulationState()
            self.bacteria = []
            self.food = []
            self.pheromone_map.fill(0)
            self.history.clear()
            self.statistics = {
                'population_history': [],
                'fitness_history': [],
                'energy_history': [],
                'generation_times': []
            }
            logger.info("Simulation reset")
    
    def get_state(self) -> Dict:
        """Simülasyon durumunu al"""
        with self.lock:
            return {
                'state': self.state.__dict__,
                'bacteria': [b.copy() for b in self.bacteria],
                'food': [f.copy() for f in self.food],
                'statistics': self.statistics.copy(),
                'config': self.config.__dict__
            }
    
    def save_state(self, filename: str):
        """Simülasyon durumunu kaydet"""
        state = self.get_state()
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Simulation state saved to {filename}")
    
    def load_state(self, filename: str):
        """Simülasyon durumunu yükle"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        with self.lock:
            # Config yükle
            for key, value in data['config'].items():
                setattr(self.config, key, value)
            
            # State yükle
            for key, value in data['state'].items():
                setattr(self.state, key, value)
            
            # Varlıkları yükle
            self.bacteria = data['bacteria']
            self.food = data['food']
            self.statistics = data['statistics']
        
        logger.info(f"Simulation state loaded from {filename}")
