# Advanced Bacterium V7 - Gerçek Biyofiziksel Hesaplamalar
# Based on research: Moleküler Dinamik, Popülasyon Genetiği ve AI Entegrasyonu

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class BacteriumState(Enum):
    GROWING = "growing"
    REPRODUCING = "reproducing"
    STRESSED = "stressed"
    DYING = "dying"
    DORMANT = "dormant"

class MetabolicPathway(Enum):
    GLYCOLYSIS = "glycolysis"
    TCA_CYCLE = "tca_cycle"
    OXIDATIVE_PHOSPHORYLATION = "oxidative_phosphorylation"
    FERMENTATION = "fermentation"

@dataclass
class GeneticProfile:
    fitness: float
    mutation_rate: float
    adaptation_speed: float
    stress_resistance: float
    metabolic_efficiency: float
    age: int = 0
    generation: int = 0

@dataclass
class BiophysicalProperties:
    mass: float
    surface_area: float
    charge: float
    hydrophobicity: float
    membrane_permeability: float

class AdvancedBacteriumV7:
    """
    Gerçek biyofiziksel hesaplamalar ile gelişmiş bakterium modeli
    Moleküler dinamik, popülasyon genetiği ve AI decision making entegrasyonu
    """
    
    def __init__(self, x=0.0, y=0.0, z=0.0, bacterium_id=None):
        self.id = bacterium_id or f"bacteria_{random.randint(10000, 99999)}"
        
        # 3D pozisyon ve hareket
        self.position = np.array([x, y, z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # Temel özellikler  
        self.energy = random.uniform(70.0, 100.0)
        self.size = random.uniform(0.5, 2.0)  # mikrometers
        self.age = 0
        self.state = BacteriumState.GROWING
        
        # Genetik profil (Wright-Fisher model tabanlı)
        self.genetics = GeneticProfile(
            fitness=random.gauss(0.75, 0.15),  # Normal dağılım
            mutation_rate=random.uniform(1e-6, 1e-4),
            adaptation_speed=random.uniform(0.1, 0.9),
            stress_resistance=random.uniform(0.2, 0.8),
            metabolic_efficiency=random.uniform(0.5, 1.0)
        )
        
        # Biyofiziksel özellikler
        self.biophysics = BiophysicalProperties(
            mass=self.size * 1e-12,  # gram
            surface_area=4 * math.pi * (self.size/2)**2,  # um^2
            charge=random.uniform(-0.1, 0.1),  # relative charge
            hydrophobicity=random.uniform(0.3, 0.7),
            membrane_permeability=random.uniform(0.1, 0.5)
        )
        
        # Metabolik durum
        self.atp_level = random.uniform(50.0, 100.0)
        self.nadh_level = random.uniform(10.0, 50.0)
        self.active_pathway = MetabolicPathway.GLYCOLYSIS
        
        # Çevresel etkileşim
        self.neighbors = []
        self.stress_factors = 0.0
        self.fitness_history = [self.genetics.fitness]
        
        # Fiziksel kuvvetler
        self.forces = np.array([0.0, 0.0, 0.0], dtype=float)
        
    def calculate_molecular_forces(self, other_bacteria: List['AdvancedBacteriumV7'], 
                                 environment_params: Dict) -> np.array:
        """
        Van der Waals ve elektrostatik kuvvetleri hesaplar
        Based on molecular dynamics research
        """
        total_force = np.array([0.0, 0.0, 0.0], dtype=float)
        
        for other in other_bacteria:
            if other.id == self.id:
                continue
                
            # Mesafe hesabı
            distance_vec = other.position - self.position
            distance = np.linalg.norm(distance_vec)
            
            if distance < 0.1:  # Çok yakın
                continue
                
            # Van der Waals kuvveti (Lennard-Jones)
            sigma = (self.size + other.size) / 4.0  # Characteristic distance
            epsilon = 0.995  # Energy parameter
            
            if distance < 5.0 * sigma:  # Cutoff distance
                r_over_sigma = distance / sigma
                lj_force_magnitude = 4 * epsilon * (
                    12 * (sigma/distance)**13 - 6 * (sigma/distance)**7
                ) / distance
                
                # Elektrostatik kuvvet
                k_coulomb = 8.99e9  # Coulomb sabiti
                electrostatic_force = k_coulomb * self.biophysics.charge * other.biophysics.charge / (distance**2)
                
                # Toplam kuvvet
                total_force_magnitude = lj_force_magnitude + electrostatic_force
                
                # Kuvvet yönü
                direction = distance_vec / distance
                total_force += total_force_magnitude * direction
        
        # Çevresel kuvvetler (brownian motion, fluid drag)
        brownian_force = np.random.normal(0, 0.1, 3)  # Termal hareket
        drag_coefficient = 6 * math.pi * environment_params.get('viscosity', 1e-3) * self.size
        drag_force = -drag_coefficient * self.velocity
        
        total_force += brownian_force + drag_force
        
        return total_force
    
    def update_genetics_wright_fisher(self, population_size: int, selection_pressure: float):
        """
        Wright-Fisher modeli ile genetik güncelleme
        """
        # Effective population size
        ne = min(population_size, 1000)
        
        # Binomial sampling for genetic drift
        if random.random() < 1.0 / (2 * ne):  # Genetic drift
            drift_effect = random.gauss(0, 0.01)
            self.genetics.fitness += drift_effect
            
        # Natural selection
        if selection_pressure > 0.1:
            if self.genetics.fitness > 0.7:  # Beneficial
                self.genetics.fitness += selection_pressure * 0.01
            else:  # Deleterious
                self.genetics.fitness -= selection_pressure * 0.01
        
        # Mutation
        if random.random() < self.genetics.mutation_rate:
            mutation_effect = random.gauss(0, 0.05)
            self.genetics.fitness += mutation_effect
            self.genetics.age = 0  # Reset age for new variant
            
        # Bounds
        self.genetics.fitness = np.clip(self.genetics.fitness, 0.1, 1.0)
        
        # Update fitness history
        self.fitness_history.append(self.genetics.fitness)
        if len(self.fitness_history) > 100:
            self.fitness_history.pop(0)
    
    def calculate_atp_synthesis(self, environment_params: Dict) -> float:
        """
        ATP sentezi hesabı - gerçek biyokimyasal yolaklar
        """
        glucose_concentration = environment_params.get('glucose', 0.5)
        oxygen_level = environment_params.get('oxygen', 0.6)
        ph_level = environment_params.get('ph', 7.0)
        temperature = environment_params.get('temperature', 37.0)
        
        # Glycolysis efficiency
        glycolysis_efficiency = min(glucose_concentration * 2.0, 1.0)
        
        # Optimal pH effect (pH 6.5-7.5 optimal)
        ph_factor = 1.0 - abs(ph_level - 7.0) / 3.0
        ph_factor = max(0.1, ph_factor)
        
        # Temperature effect (Arrhenius equation)
        optimal_temp = 37.0
        temp_factor = math.exp(-abs(temperature - optimal_temp) / 10.0)
        
        # Pathway selection based on oxygen
        if oxygen_level > 0.3:
            # Aerobic respiration
            self.active_pathway = MetabolicPathway.OXIDATIVE_PHOSPHORYLATION
            atp_yield = 38 * glycolysis_efficiency * ph_factor * temp_factor
        else:
            # Anaerobic fermentation
            self.active_pathway = MetabolicPathway.FERMENTATION
            atp_yield = 2 * glycolysis_efficiency * ph_factor * temp_factor
        
        # Metabolic efficiency factor
        atp_yield *= self.genetics.metabolic_efficiency
        
        return atp_yield
    
    def decide_action_ai(self, environment_state: Dict, neighbor_info: List[Dict]) -> str:
        """
        AI tabanlı karar verme sistemi
        Basit neural network benzeri karar ağacı
        """
        # Input features
        features = [
            self.energy / 100.0,
            self.atp_level / 100.0,
            environment_state.get('nutrient_density', 0.5),
            environment_state.get('toxin_level', 0.0),
            len(neighbor_info) / 10.0,  # Normalize neighbor count
            self.genetics.fitness,
            self.stress_factors
        ]
        
        # Basit decision network (weights learned from experience)
        # Bu gerçek implementasyonda reinforcement learning ile öğrenilebilir
        
        move_score = sum([
            features[0] * 0.3,  # Energy level
            features[2] * 0.4,  # Nutrient availability
            -features[3] * 0.5,  # Avoid toxins
            -features[4] * 0.2   # Avoid crowding
        ])
        
        reproduce_score = sum([
            features[0] * 0.6,  # High energy needed
            features[1] * 0.4,  # ATP availability
            features[5] * 0.3   # Fitness
        ])
        
        rest_score = sum([
            -features[0] * 0.4,  # Low energy
            features[6] * 0.5    # High stress
        ])
        
        # Decision
        scores = {'move': move_score, 'reproduce': reproduce_score, 'rest': rest_score}
        action = max(scores, key=scores.get)
        
        # State update based on decision
        if action == 'reproduce' and reproduce_score > 0.7:
            self.state = BacteriumState.REPRODUCING
        elif action == 'rest' or self.energy < 20:
            self.state = BacteriumState.DORMANT
        elif self.stress_factors > 0.8:
            self.state = BacteriumState.STRESSED
        else:
            self.state = BacteriumState.GROWING
            
        return action
    
    def update_biophysics(self, dt: float, forces: np.array):
        """
        Verlet algoritması ile fiziksel güncelleme
        """
        # Acceleration: F = ma
        self.acceleration = forces / self.biophysics.mass
        
        # Velocity update: v(t+dt) = v(t) + a*dt
        self.velocity += self.acceleration * dt
        
        # Terminal velocity (drag limit)
        max_velocity = 10.0  # um/s
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > max_velocity:
            self.velocity = self.velocity * max_velocity / velocity_magnitude
        
        # Position update: r(t+dt) = r(t) + v*dt + 0.5*a*dt^2
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt * dt
        
        # Boundary conditions (optional)
        self.position = np.clip(self.position, -100, 100)
        
    def metabolic_update(self, environment_params: Dict, dt: float):
        """
        Metabolik süreçleri güncelle
        """
        # ATP synthesis
        atp_production = self.calculate_atp_synthesis(environment_params)
        self.atp_level += atp_production * dt
        
        # ATP consumption
        base_consumption = 5.0 * dt
        movement_cost = np.linalg.norm(self.velocity) * 2.0 * dt
        growth_cost = 0.0
        
        if self.state == BacteriumState.GROWING:
            growth_cost = 10.0 * dt
        elif self.state == BacteriumState.REPRODUCING:
            growth_cost = 20.0 * dt
            
        total_consumption = base_consumption + movement_cost + growth_cost
        self.atp_level -= total_consumption
        
        # Energy update based on ATP
        if self.atp_level > 50:
            self.energy += 2.0 * dt
        elif self.atp_level < 20:
            self.energy -= 5.0 * dt
            
        # Bounds
        self.atp_level = np.clip(self.atp_level, 0, 200)
        self.energy = np.clip(self.energy, 0, 150)
        
        # Age increment
        self.age += dt
        self.genetics.age += dt
    
    def calculate_stress_response(self, environment_params: Dict):
        """
        Stres faktörlerini hesapla ve stress response
        """
        stress_sources = []
        
        # Temperature stress
        temp_stress = abs(environment_params.get('temperature', 37) - 37) / 20.0
        stress_sources.append(temp_stress)
        
        # pH stress
        ph_stress = abs(environment_params.get('ph', 7.0) - 7.0) / 3.0
        stress_sources.append(ph_stress)
        
        # Toxin stress
        toxin_stress = environment_params.get('toxin_level', 0.0)
        stress_sources.append(toxin_stress)
        
        # Crowding stress
        neighbor_count = len(self.neighbors)
        crowding_stress = max(0, (neighbor_count - 5) / 10.0)
        stress_sources.append(crowding_stress)
        
        # Total stress
        self.stress_factors = min(1.0, sum(stress_sources))
        
        # Stress resistance effect
        effective_stress = self.stress_factors * (1.0 - self.genetics.stress_resistance)
        
        if effective_stress > 0.5:
            self.energy -= effective_stress * 3.0
            if effective_stress > 0.8:
                self.state = BacteriumState.DYING
    
    def attempt_reproduction(self) -> Optional['AdvancedBacteriumV7']:
        """
        Üreme denemesi - yeni bakterium oluşturma
        """
        if (self.state == BacteriumState.REPRODUCING and 
            self.energy > 60 and 
            self.atp_level > 80):
            
            # Energy cost
            self.energy -= 40
            self.atp_level -= 60
            
            # Create offspring with inheritance and mutation
            offspring = AdvancedBacteriumV7(
                x=self.position[0] + random.uniform(-2, 2),
                y=self.position[1] + random.uniform(-2, 2),  
                z=self.position[2] + random.uniform(-2, 2)
            )
            
            # Genetic inheritance with mutation
            offspring.genetics.fitness = self.genetics.fitness + random.gauss(0, 0.02)
            offspring.genetics.mutation_rate = self.genetics.mutation_rate
            offspring.genetics.adaptation_speed = self.genetics.adaptation_speed
            offspring.genetics.stress_resistance = self.genetics.stress_resistance  
            offspring.genetics.metabolic_efficiency = self.genetics.metabolic_efficiency
            offspring.genetics.generation = self.genetics.generation + 1
            
            # Size inheritance
            offspring.size = self.size * random.uniform(0.8, 1.2)
            offspring.biophysics.mass = offspring.size * 1e-12
            offspring.biophysics.surface_area = 4 * math.pi * (offspring.size/2)**2
            
            return offspring
        
        return None
    
    def should_die(self) -> bool:
        """
        Ölüm koşullarını kontrol et
        """
        if self.energy <= 0:
            return True
        if self.state == BacteriumState.DYING and self.stress_factors > 0.9:
            return True
        if self.age > 1000:  # Max lifespan
            return True
        return False
    
    def get_state_summary(self) -> Dict:
        """
        Bakteriyum durumunun özetini al
        """
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'energy': self.energy,
            'atp_level': self.atp_level,
            'size': self.size,
            'age': self.age,
            'state': self.state.value,
            'fitness': self.genetics.fitness,
            'generation': self.genetics.generation,
            'metabolic_pathway': self.active_pathway.value,
            'stress_factors': self.stress_factors,
            'mass': self.biophysics.mass,
            'charge': self.biophysics.charge
        }

def test_advanced_bacterium():
    """
    AdvancedBacteriumV7 test fonksiyonu
    """
    print("=== Advanced Bacterium V7 Test ===")
    
    # Test environment
    environment = {
        'glucose': 0.7,
        'oxygen': 0.8,
        'ph': 7.2,
        'temperature': 37.5,
        'toxin_level': 0.1,
        'viscosity': 1e-3
    }
    
    # Create bacteria population
    bacteria = []
    for i in range(5):
        bacterium = AdvancedBacteriumV7(
            x=random.uniform(-10, 10),
            y=random.uniform(-10, 10),
            z=random.uniform(-10, 10)
        )
        bacteria.append(bacterium)
    
    print(f"Created {len(bacteria)} bacteria")
    
    # Simulation steps
    dt = 0.1
    for step in range(50):
        for bacterium in bacteria:
            # Calculate forces from other bacteria
            forces = bacterium.calculate_molecular_forces(bacteria, environment)
            
            # Physics update
            bacterium.update_biophysics(dt, forces)
            
            # Metabolic update
            bacterium.metabolic_update(environment, dt)
            
            # Genetic update
            bacterium.update_genetics_wright_fisher(len(bacteria), 0.2)
            
            # Stress response
            bacterium.calculate_stress_response(environment)
            
            # AI decision
            neighbor_info = [b.get_state_summary() for b in bacteria if b.id != bacterium.id]
            action = bacterium.decide_action_ai(environment, neighbor_info)
            
            # Reproduction attempt
            if action == 'reproduce':
                offspring = bacterium.attempt_reproduction()
                if offspring:
                    bacteria.append(offspring)
                    print(f"Step {step}: New bacterium born! Population: {len(bacteria)}")
        
        # Remove dead bacteria
        bacteria = [b for b in bacteria if not b.should_die()]
        
        if step % 10 == 0:
            avg_fitness = np.mean([b.genetics.fitness for b in bacteria])
            avg_energy = np.mean([b.energy for b in bacteria])
            print(f"Step {step}: Population={len(bacteria)}, "
                  f"Avg Fitness={avg_fitness:.3f}, Avg Energy={avg_energy:.1f}")
    
    print(f"Final population: {len(bacteria)}")

if __name__ == "__main__":
    test_advanced_bacterium() 