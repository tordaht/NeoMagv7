"""
NeoMag V7 - Reinforcement Learning Networks
Bio-fizik tabanlı reinforcement learning modelleri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class BiophysicalDQN(nn.Module):
    """
    Bio-fizik prensiplere dayalı Deep Q-Network
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(BiophysicalDQN, self).__init__()
        
        # Neural layers - biological neuron yapısını taklit
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Dropout for regularization (synaptic pruning simulasyonu)
        self.dropout = nn.Dropout(0.2)
        
        # Batch normalization (homeostasis simulasyonu)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # Biological activation functions (ReLU ~ biological threshold)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class BacteriumRLAgent:
    """
    Bakteri davranışı için RL ajanı
    """
    def __init__(self, 
                 state_size: int = 10,
                 action_size: int = 4,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Neural networks
        self.q_network = BiophysicalDQN(state_size, action_size)
        self.target_network = BiophysicalDQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Target network'ü güncelle"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Deneyimi hafızaya ekle"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Epsilon-greedy policy ile aksiyon seç"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.data.numpy())
    
    def replay(self, batch_size: int = 32):
        """Experience replay ile öğren"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class EvolutionaryRL:
    """
    Evrimsel strateji ile reinforcement learning
    """
    def __init__(self, 
                 population_size: int = 50,
                 elite_fraction: float = 0.2,
                 mutation_rate: float = 0.1):
        
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.generation = 0
        
        # Popülasyon başlat
        self.population = []
        self.fitness_history = []
        
    def create_individual(self, state_size: int, action_size: int) -> Dict:
        """Yeni birey oluştur"""
        return {
            'network': BiophysicalDQN(state_size, action_size),
            'fitness': 0,
            'age': 0
        }
    
    def mutate(self, network: nn.Module):
        """Network parametrelerini mutasyona uğrat"""
        with torch.no_grad():
            for param in network.parameters():
                if random.random() < self.mutation_rate:
                    noise = torch.randn_like(param) * 0.1
                    param.add_(noise)
    
    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        """İki parent network'ten child oluştur"""
        child = BiophysicalDQN(
            parent1.fc1.in_features,
            parent1.fc4.out_features
        )
        
        # Parametreleri karıştır
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(
                child.parameters(),
                parent1.parameters(),
                parent2.parameters()
            ):
                mask = torch.rand_like(child_param) > 0.5
                child_param.data = torch.where(mask, p1_param.data, p2_param.data)
        
        return child
    
    def evolve(self):
        """Popülasyonu evrimleştir"""
        # Fitness'a göre sırala
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Elite bireyleri koru
        elite_size = int(self.population_size * self.elite_fraction)
        new_population = self.population[:elite_size]
        
        # Yeni nesil oluştur
        while len(new_population) < self.population_size:
            # Parent seçimi (tournament selection)
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()
            
            # Crossover
            child_network = self.crossover(
                parent1['network'],
                parent2['network']
            )
            
            # Mutation
            self.mutate(child_network)
            
            new_population.append({
                'network': child_network,
                'fitness': 0,
                'age': 0
            })
        
        self.population = new_population
        self.generation += 1
        
        # İstatistikleri kaydet
        avg_fitness = np.mean([ind['fitness'] for ind in self.population])
        self.fitness_history.append(avg_fitness)
        logger.info(f"Generation {self.generation}, Avg Fitness: {avg_fitness:.3f}")
    
    def tournament_select(self, tournament_size: int = 3) -> Dict:
        """Tournament selection ile parent seç"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x['fitness'])


class BioPhysicsEnvironment:
    """
    Bio-fizik simülasyon ortamı
    """
    def __init__(self, width: int = 1200, height: int = 600):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        """Ortamı sıfırla"""
        self.bacteria_positions = []
        self.food_positions = []
        self.pheromone_map = np.zeros((self.height // 10, self.width // 10))
        self.time_step = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Mevcut durumu al"""
        # Basitleştirilmiş durum temsili
        state = []
        
        # Yakın bakterilerin sayısı
        nearby_bacteria = self.count_nearby_entities(self.bacteria_positions)
        state.append(nearby_bacteria)
        
        # Yakın yiyeceklerin sayısı
        nearby_food = self.count_nearby_entities(self.food_positions)
        state.append(nearby_food)
        
        # Feromon yoğunluğu
        pheromone_level = self.get_pheromone_level()
        state.append(pheromone_level)
        
        # Zaman bilgisi (circadian rhythm simulasyonu)
        state.append(np.sin(self.time_step * 0.1))
        state.append(np.cos(self.time_step * 0.1))
        
        return np.array(state, dtype=np.float32)
    
    def count_nearby_entities(self, positions: List, radius: float = 50) -> int:
        """Yakındaki varlıkları say"""
        # Basitleştirilmiş implementasyon
        return min(len(positions), 10) / 10.0
    
    def get_pheromone_level(self) -> float:
        """Feromon seviyesini al"""
        return np.mean(self.pheromone_map)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Ortamda bir adım at"""
        # Aksiyon uygula
        reward = self.apply_action(action)
        
        # Ortamı güncelle
        self.update_environment()
        
        # Yeni durum
        next_state = self.get_state()
        
        # Episode bitişi kontrolü
        done = self.time_step > 1000
        
        info = {
            'time_step': self.time_step,
            'bacteria_count': len(self.bacteria_positions),
            'food_count': len(self.food_positions)
        }
        
        return next_state, reward, done, info
    
    def apply_action(self, action: int) -> float:
        """Aksiyonu uygula ve ödül hesapla"""
        reward = 0.0
        
        if action == 0:  # Hareket et
            reward += 0.1
        elif action == 1:  # Yiyecek ara
            if len(self.food_positions) > 0:
                reward += 1.0
        elif action == 2:  # Üre
            if len(self.bacteria_positions) < 100:
                reward += 2.0
        elif action == 3:  # Feromon sal
            reward += 0.5
        
        return reward
    
    def update_environment(self):
        """Ortamı güncelle"""
        self.time_step += 1
        
        # Feromon haritasını azalt (evaporation)
        self.pheromone_map *= 0.99
        
        # Rastgele yiyecek ekle
        if random.random() < 0.1:
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            self.food_positions.append((x, y))
