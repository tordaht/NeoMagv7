"""NeoMag V7 - Advanced Bacterium Agent"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
import sys
sys.path.append('..')

try:
    from ..core.biophysical_properties import BiophysicalProperties
    from ..core.genetic_profile import GeneticProfile
except ImportError:
    from core.biophysical_properties import BiophysicalProperties
    from core.genetic_profile import GeneticProfile

class AdvancedBacteriumV7:
    """V7 gelişmiş bakteri sınıfı - modüler yapı"""
    _id_counter = 0

    def __init__(self, x: float, y: float, z: float = 0.0, bacteria_id: Optional[int] = None):
        if bacteria_id is None:
            AdvancedBacteriumV7._id_counter += 1
            self.id = AdvancedBacteriumV7._id_counter
        else:
            self.id = bacteria_id
            AdvancedBacteriumV7._id_counter = max(AdvancedBacteriumV7._id_counter, bacteria_id)

        # Biophysical properties - 3D pozisyon
        self.biophysical = BiophysicalProperties(position=np.array([x, y, z], dtype=float))
        self.size = np.random.uniform(0.5, 1.5)  # Mikrometre
        self.biophysical.mass = self.size**3 * 1e-15 * 1000  # kg

        self.genetic_profile = GeneticProfile(genome_length=np.random.randint(500, 2000))
        self.genetic_profile.recalculate_derived_properties()

        self.energy_level = np.random.uniform(80.0, 120.0)
        self.age = 0.0
        self.generation = 1
        self.metabolic_rate = np.random.uniform(0.05, 0.15)

        # Etkileşim sayaçları
        self.md_interactions_count = 0
        self.genetic_operations_count = 0
        self.ai_decisions_count = 0
        self.current_fitness = self.calculate_fitness()

    @property
    def x(self): 
        return self.biophysical.position[0]
    
    @x.setter
    def x(self, value): 
        self.biophysical.position[0] = value

    @property
    def y(self): 
        return self.biophysical.position[1]
    
    @y.setter
    def y(self, value): 
        self.biophysical.position[1] = value

    @property
    def z(self): 
        return self.biophysical.position[2]
    
    @z.setter
    def z(self, value): 
        self.biophysical.position[2] = value

    def update_molecular_state(self, md_engine, all_bacteria: List['AdvancedBacteriumV7'], dt: float):
        """Moleküler durumu güncelle"""
        # MD engine'den gelen kuvvetlere bağlı olarak iç durumu güncelle
        # ATP/ADP değişimleri, membran potansiyeli vs.
        
        # Basit metabolizma modeli
        atp_consumption = self.metabolic_rate * dt
        current_atp = self.biophysical.ion_concentrations.get('ATP', 5.0)
        new_atp = max(0.1, current_atp - atp_consumption)
        self.biophysical.ion_concentrations['ATP'] = new_atp
        
        # Enerji seviyesi güncelleme
        self.energy_level = max(0, self.energy_level - atp_consumption * 10)
        
        self.md_interactions_count += 1

    def update_genetic_state(self, pop_gen_engine, current_population: List['AdvancedBacteriumV7']):
        """Genetik durumu güncelle"""
        # Fitness güncellemesi
        self.current_fitness = self.calculate_fitness()
        
        # Rastgele gen ifadesi değişikliği (çok düşük olasılık)
        if self.genetic_profile.gene_expression_levels and np.random.rand() < 0.001:
            random_gene = np.random.choice(list(self.genetic_profile.gene_expression_levels.keys()))
            change = np.random.normal(0, 0.05)
            current_expr = self.genetic_profile.gene_expression_levels[random_gene]
            self.genetic_profile.gene_expression_levels[random_gene] = np.clip(current_expr + change, 0.0, 1.0)
        
        self.genetic_operations_count += 1

    def calculate_fitness(self) -> float:
        """Fitness hesaplama"""
        # Enerji komponenti
        energy_component = np.tanh(self.energy_level / 100.0)
        
        # Yaş komponenti (genç ve orta yaşlılar daha fit)
        age_component = max(0, 1 - (self.age / 600.0)**2)
        
        # Genetik komponent
        genetic_component = 0.5
        if self.genetic_profile.fitness_landscape_position:
            genetic_component = np.mean(self.genetic_profile.fitness_landscape_position)
            genetic_component = np.clip(genetic_component, 0.0, 1.0)

        # ATP komponenti
        atp_level = self.biophysical.ion_concentrations.get('ATP', 5.0)
        atp_component = min(1.0, atp_level / 5.0)

        fitness = (energy_component * 0.3 +
                   age_component * 0.2 +
                   genetic_component * 0.3 +
                   atp_component * 0.2)

        return np.clip(fitness, 0.01, 1.0)

    def make_decision(self, ai_engine, environment_state: Dict[str, Any], world_dims: Tuple[float, float, float]) -> str:
        """AI motoru ile karar alma"""
        # State representation oluştur
        state = self._get_state_representation(environment_state, world_dims)
        possible_actions = self._get_possible_actions()
        
        # AI engine'den action al
        chosen_action = ai_engine.select_action(state, possible_actions, agent_id=self.id)
        self.ai_decisions_count += 1
        return chosen_action

    def _get_state_representation(self, environment_state: Dict[str, Any], world_dims: Tuple[float, float, float]) -> np.ndarray:
        """Durum vektörü oluştur"""
        world_width, world_height, world_depth = world_dims
        
        # Pozisyon normalizasyonu
        norm_x = self.x / world_width if world_width > 0 else 0.0
        norm_y = self.y / world_height if world_height > 0 else 0.0
        norm_z = self.z / world_depth if world_depth > 0 else 0.0
        
        # Durum vektörü (15 boyut - ai_decision.py'deki state_size ile uyumlu)
        state = np.array([
            norm_x, norm_y, norm_z,                          # Pozisyon (3)
            self.energy_level / 100.0,                       # Enerji (1)
            self.age / 600.0,                                 # Yaş (1)
            self.size / 2.0,                                  # Boyut (1)
            self.biophysical.ion_concentrations.get('ATP', 5.0) / 10.0,  # ATP (1)
            self.current_fitness,                             # Fitness (1)
            len(environment_state.get('nearby_bacteria', [])) / 10.0,     # Yakın bakteri sayısı (1)
            len(environment_state.get('nearby_food', [])) / 10.0,         # Yakın besin sayısı (1)
            np.linalg.norm(self.biophysical.velocity) / 10.0,             # Hız büyüklüğü (1)
            self.biophysical.membrane_potential / 100.0,     # Membran potansiyeli (1)
            self.biophysical.ph_gradient / 10.0,             # pH gradient (1)
            environment_state.get('local_density', 0.0),     # Lokal yoğunluk (1)
            environment_state.get('food_concentration', 0.0) # Besin konsantrasyonu (1)
        ], dtype=np.float32)
        
        return state

    def _get_possible_actions(self) -> List[str]:
        """Mümkün eylemler"""
        return ["move_up", "move_down", "move_left", "move_right", "consume", "wait"]

    def reproduce(self, partner: Optional['AdvancedBacteriumV7'] = None, pop_gen_engine=None) -> Optional['AdvancedBacteriumV7']:
        """Üreme işlemi"""
        if self.energy_level < 40:  # Minimum enerji gereksinimi
            return None

        self.energy_level -= 30  # Üreme maliyeti

        # Yavru pozisyonu
        offset = np.random.randn(3) * self.size * 2
        offspring_pos = self.biophysical.position + offset

        offspring = AdvancedBacteriumV7(
            x=offspring_pos[0], 
            y=offspring_pos[1], 
            z=offspring_pos[2]
        )
        offspring.generation = self.generation + 1

        # Genetik miras
        if pop_gen_engine:
            mutation_rate = pop_gen_engine.mutation_rate
            recombination_rate = pop_gen_engine.recombination_rate
        else:
            mutation_rate = 1e-6
            recombination_rate = 1e-7

        offspring.genetic_profile = self._create_offspring_genetics(
            self.genetic_profile,
            partner.genetic_profile if partner else None,
            mutation_rate,
            recombination_rate
        )
        
        offspring.genetic_profile.recalculate_derived_properties()
        offspring.current_fitness = offspring.calculate_fitness()

        if partner:
            partner.energy_level -= 15  # Partner maliyeti

        return offspring

    def _create_offspring_genetics(self, parent1_profile: GeneticProfile,
                                   parent2_profile: Optional[GeneticProfile],
                                   mutation_rate: float,
                                   recombination_rate: float) -> GeneticProfile:
        """Yavru genetiğini oluştur"""
        # Parent1'den başlangıç genomu
        if not parent1_profile.genome_sequence:
            logging.warning(f"Ebeveyn {self.id} için genom dizisi eksik!")
            temp_genome_len = parent1_profile.genome_length if parent1_profile.genome_length > 0 else 1000
            p1_genome_seq_list = list(np.random.choice(['A', 'T', 'G', 'C'], size=temp_genome_len))
        else:
            p1_genome_seq_list = list(parent1_profile.genome_sequence)

        offspring_genome_seq_list = p1_genome_seq_list[:]

        # Rekombinasyon (eşeyli üreme)
        if parent2_profile and parent2_profile.genome_sequence and np.random.rand() < recombination_rate:
            p2_genome_seq_list = list(parent2_profile.genome_sequence)
            len1 = len(offspring_genome_seq_list)
            len2 = len(p2_genome_seq_list)
            recomb_len = min(len1, len2)

            if recomb_len > 0:
                recomb_point = np.random.randint(0, recomb_len)
                temp_offspring_part2 = p2_genome_seq_list[recomb_point:recomb_len]
                offspring_genome_seq_list = (offspring_genome_seq_list[:recomb_point] + 
                                            temp_offspring_part2 + 
                                            offspring_genome_seq_list[recomb_len:])

        # Mutasyon
        for i in range(len(offspring_genome_seq_list)):
            if np.random.rand() < mutation_rate:
                original_base = offspring_genome_seq_list[i]
                possible_mutations = [b for b in 'ATGC' if b != original_base]
                if possible_mutations:
                    offspring_genome_seq_list[i] = np.random.choice(possible_mutations)

        # Yeni genetik profil
        offspring_genome_str = "".join(offspring_genome_seq_list)
        new_profile = GeneticProfile(genome_sequence=offspring_genome_str, mutation_rate=mutation_rate)

        # Fitness landscape pozisyonu miras
        if parent1_profile.fitness_landscape_position:
            p1_pos = list(parent1_profile.fitness_landscape_position)
            new_pos = []
            for coord_idx, coord_val in enumerate(p1_pos):
                mut_effect = np.random.normal(0, 0.05)
                p2_coord_val = coord_val
                
                if (parent2_profile and parent2_profile.fitness_landscape_position and 
                    len(parent2_profile.fitness_landscape_position) > coord_idx):
                    p2_coord_val = parent2_profile.fitness_landscape_position[coord_idx]
                    new_coord = (coord_val + p2_coord_val) / 2.0 + mut_effect
                else:
                    new_coord = coord_val + mut_effect
                    
                new_pos.append(np.clip(new_coord, 0.0, 1.0))
            new_profile.fitness_landscape_position = tuple(new_pos)
        else:
            num_dims = len(new_profile.fitness_landscape_position) if new_profile.fitness_landscape_position else 10
            new_profile.fitness_landscape_position = tuple(np.random.rand(num_dims))

        return new_profile

    def update_age(self, dt: float):
        """Yaşlandırma"""
        self.age += dt

    def apply_physics_update(self, dt: float):
        """Fizik güncellemesi"""
        # Pozisyon güncelleme
        self.biophysical.update_position(dt)
        
        # Kuvvet tabanlı hız güncellemesi
        if self.biophysical.mass > 0:
            acceleration = self.biophysical.current_forces / self.biophysical.mass
            self.biophysical.update_velocity(acceleration, dt)
        
        # Kuvvetleri sıfırla
        self.biophysical.reset_forces()

    def consume_energy(self, amount: float):
        """Enerji tüketimi"""
        self.energy_level = max(0, self.energy_level - amount)

    def gain_energy(self, amount: float):
        """Enerji kazanımı"""
        self.energy_level = min(150, self.energy_level + amount)

    def is_alive(self) -> bool:
        """Yaşayıp yaşamadığını kontrol et"""
        return self.energy_level > 0 and self.age < 1000

    def get_status_summary(self) -> Dict[str, Any]:
        """Durum özeti"""
        fitness_val = 0.0
        if (self.genetic_profile.fitness_landscape_position and 
            len(self.genetic_profile.fitness_landscape_position) > 0):
            fitness_val = np.mean(self.genetic_profile.fitness_landscape_position)

        return {
            'id': self.id,
            'position': tuple(self.biophysical.position.tolist()),
            'velocity': tuple(self.biophysical.velocity.tolist()),
            'energy_level': self.energy_level,
            'age': self.age,
            'generation': self.generation,
            'size': self.size,
            'mass': self.biophysical.mass,
            'current_fitness_calculated': self.current_fitness,
            'genetic_fitness_proxy': fitness_val,
            'atp_level': self.biophysical.ion_concentrations.get('ATP', 5.0),
            'md_interactions': self.md_interactions_count,
            'genetic_operations': self.genetic_operations_count,
            'ai_decisions': self.ai_decisions_count,
            'genome_length': self.genetic_profile.genome_length,
            'metabolic_rate': self.metabolic_rate,
            'is_alive': self.is_alive()
        }
