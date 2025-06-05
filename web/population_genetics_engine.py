# Popülasyon Genetiği Motoru - Wright-Fisher & Coalescent Teori
# Based on research: Advanced Topics in Population Genetics and Evolutionary Modeling

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SelectionType(Enum):
    NEUTRAL = "neutral"
    DIRECTIONAL = "directional"
    BALANCING = "balancing"
    DISRUPTIVE = "disruptive"

@dataclass
class Allele:
    id: str
    frequency: float
    fitness: float = 1.0
    age: int = 0  # Alelin yaşı (nesil sayısı)

@dataclass 
class Population:
    size: int
    alleles: List[Allele]
    generation: int = 0
    effective_size: Optional[int] = None

class WrightFisherModel:
    """
    Wright-Fisher modeli ile popülasyon dinamikleri
    Genetic drift, selection, mutation ve migration hesapları
    """
    
    def __init__(self, population_size: int, mutation_rate: float = 1e-6, 
                 migration_rate: float = 0.0):
        self.N = population_size  # Diploid birey sayısı (2N gen kopyası)
        self.mutation_rate = mutation_rate
        self.migration_rate = migration_rate
        self.generation_count = 0
        
    def hardy_weinberg_equilibrium(self, p: float) -> Tuple[float, float, float]:
        """
        Hardy-Weinberg dengesi hesaplaması
        Returns: (AA frequency, Aa frequency, aa frequency)
        """
        q = 1.0 - p
        aa_freq = p * p
        Aa_freq = 2 * p * q  
        aa_freq = q * q
        return aa_freq, Aa_freq, aa_freq
    
    def binomial_sampling(self, allele_count: int, total_count: int) -> int:
        """
        Wright-Fisher modelinde binomial örnekleme
        Her nesilde 2N gen kopyası rastgele örneklenir
        """
        if total_count == 0:
            return 0
        p = allele_count / total_count
        return np.random.binomial(2 * self.N, p)
    
    def calculate_selection_fitness(self, alleles: List[Allele], 
                                  selection_type: SelectionType = SelectionType.NEUTRAL,
                                  selection_coefficient: float = 0.01) -> List[float]:
        """
        Doğal seçilim etkilerini hesaplar
        """
        if selection_type == SelectionType.NEUTRAL:
            return [1.0] * len(alleles)
        
        fitness_values = []
        for allele in alleles:
            if selection_type == SelectionType.DIRECTIONAL:
                # Yönlü seçilim: belirli aleller avantajlı
                fitness = 1.0 + selection_coefficient if allele.fitness > 0.5 else 1.0
            elif selection_type == SelectionType.BALANCING:
                # Dengeli seçilim: heterozigot avantajı
                fitness = 1.0 + selection_coefficient * (1 - abs(2 * allele.frequency - 1))
            elif selection_type == SelectionType.DISRUPTIVE:
                # Parçalayıcı seçilim: ekstrem değerler avantajlı
                fitness = 1.0 + selection_coefficient * abs(2 * allele.frequency - 1)
            else:
                fitness = 1.0
                
            fitness_values.append(fitness)
        
        return fitness_values
    
    def apply_mutation(self, alleles: List[Allele]) -> List[Allele]:
        """
        Mutasyon etkilerini uygular
        """
        new_alleles = []
        for allele in alleles:
            # Mutasyon gerçekleşir mi?
            if random.random() < self.mutation_rate:
                # Yeni alel oluştur
                new_allele = Allele(
                    id=f"mut_{self.generation_count}_{len(new_alleles)}",
                    frequency=allele.frequency * 0.1,  # Küçük başlangıç frekansı
                    fitness=random.uniform(0.8, 1.2),
                    age=0
                )
                new_alleles.append(new_allele)
                
                # Orijinal alelin frekansını azalt
                allele.frequency *= 0.9
            
            if allele.frequency > 1e-6:  # Çok düşük frekanslı allelleri temizle
                allele.age += 1
                new_alleles.append(allele)
        
        return new_alleles
    
    def simulate_generation(self, population: Population, 
                          selection_type: SelectionType = SelectionType.NEUTRAL,
                          selection_coefficient: float = 0.01) -> Population:
        """
        Bir nesil simülasyonu (Wright-Fisher adımı)
        """
        # 1. Fitness hesaplama
        fitness_values = self.calculate_selection_fitness(
            population.alleles, selection_type, selection_coefficient
        )
        
        # 2. Seçilim etkisi ile frekans ayarlaması
        total_fitness = sum(f * a.frequency for f, a in zip(fitness_values, population.alleles))
        if total_fitness > 0:
            for i, allele in enumerate(population.alleles):
                allele.frequency = (allele.frequency * fitness_values[i]) / total_fitness
        
        # 3. Genetic drift (binomial örnekleme)
        total_gene_copies = 2 * self.N
        new_alleles = []
        
        for allele in population.alleles:
            expected_copies = allele.frequency * total_gene_copies
            actual_copies = self.binomial_sampling(
                int(expected_copies), total_gene_copies
            )
            
            if actual_copies > 0:
                new_frequency = actual_copies / total_gene_copies
                allele.frequency = new_frequency
                new_alleles.append(allele)
        
        # 4. Mutasyon
        new_alleles = self.apply_mutation(new_alleles)
        
        # 5. Frekansları normalize et
        total_freq = sum(a.frequency for a in new_alleles)
        if total_freq > 0:
            for allele in new_alleles:
                allele.frequency /= total_freq
        
        self.generation_count += 1
        
        return Population(
            size=population.size,
            alleles=new_alleles,
            generation=population.generation + 1,
            effective_size=self.calculate_effective_population_size(new_alleles)
        )
    
    def calculate_effective_population_size(self, alleles: List[Allele]) -> int:
        """
        Efektif popülasyon büyüklüğü hesabı
        Ne = 1 / (sum(pi^2)) where pi is allele frequency
        """
        if not alleles:
            return self.N
            
        sum_squared_freq = sum(a.frequency ** 2 for a in alleles)
        if sum_squared_freq > 0:
            ne = 1.0 / sum_squared_freq
            return min(int(ne), self.N)
        return self.N
    
    def calculate_heterozygosity(self, alleles: List[Allele]) -> Tuple[float, float]:
        """
        Heterozigotluk hesabı
        Returns: (Observed heterozygosity, Expected heterozygosity)
        """
        if len(alleles) < 2:
            return 0.0, 0.0
            
        # Beklenen heterozigotluk: He = 1 - sum(pi^2)
        he = 1.0 - sum(a.frequency ** 2 for a in alleles)
        
        # Gözlenen heterozigotluk (Hardy-Weinberg varsayımı ile)
        ho = he  # Panmiktik eşleşme varsayımı
        
        return ho, he

class CoalescentTheory:
    """
    Coalescent teori ile geriye dönük soy analizi
    MRCA (Most Recent Common Ancestor) hesaplamaları
    """
    
    def __init__(self, effective_population_size: int):
        self.Ne = effective_population_size
        
    def coalescence_rate(self, k: int) -> float:
        """
        k soy hattının coalescence oranı
        Rate = k(k-1) / (4*Ne)
        """
        if k <= 1:
            return 0.0
        return (k * (k - 1)) / (4.0 * self.Ne)
    
    def waiting_time_to_coalescence(self, k: int) -> float:
        """
        k soy hattından (k-1)'e geçiş için bekleme süresi
        Exponential distribution with rate = coalescence_rate
        """
        rate = self.coalescence_rate(k)
        if rate == 0:
            return float('inf')
        return np.random.exponential(1.0 / rate)
    
    def simulate_coalescent_tree(self, sample_size: int) -> Dict[str, float]:
        """
        Coalescent ağaç simülasyonu
        Returns: Dictionary with coalescence times
        """
        coalescence_times = {}
        current_lineages = sample_size
        total_time = 0.0
        
        while current_lineages > 1:
            waiting_time = self.waiting_time_to_coalescence(current_lineages)
            total_time += waiting_time
            
            coalescence_times[f"coalescence_{current_lineages}_to_{current_lineages-1}"] = total_time
            current_lineages -= 1
        
        coalescence_times["tmrca"] = total_time  # Time to Most Recent Common Ancestor
        return coalescence_times
    
    def calculate_genetic_diversity(self, sample_size: int, mutation_rate: float) -> float:
        """
        Genetik çeşitlilik hesabı (Theta = 4*Ne*mu)
        """
        theta = 4.0 * self.Ne * mutation_rate
        return theta
    
    def nucleotide_diversity(self, sequences: List[str]) -> float:
        """
        Nükleotid çeşitliliği hesabı (pi)
        """
        if len(sequences) < 2:
            return 0.0
            
        total_differences = 0
        total_comparisons = 0
        seq_length = len(sequences[0])
        
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                differences = sum(1 for k in range(seq_length) 
                                if sequences[i][k] != sequences[j][k])
                total_differences += differences
                total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
            
        return total_differences / (total_comparisons * seq_length)

class FitnessLandscape:
    """
    Fitness landscape optimizasyonu
    Evolutionary trajectory prediction
    """
    
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions
        self.landscape = {}
        
    def rugged_landscape(self, x: List[float]) -> float:
        """
        Pürüzlü fitness landscape (NK model benzeri)
        """
        fitness = 0.0
        for i in range(len(x)):
            # Epistatic interactions
            local_fitness = math.sin(x[i] * 2 * math.pi)
            if i < len(x) - 1:
                interaction = math.cos(x[i] * x[i+1] * math.pi)
                local_fitness += 0.3 * interaction
            fitness += local_fitness
            
        return fitness / len(x)
    
    def smooth_landscape(self, x: List[float]) -> float:
        """
        Düzgün fitness landscape (additive model)
        """
        return sum(xi ** 2 for xi in x) / len(x)
    
    def find_local_optima(self, initial_position: List[float], 
                         landscape_func, step_size: float = 0.01) -> List[float]:
        """
        Hill climbing ile lokal optimum bulma
        """
        position = initial_position.copy()
        current_fitness = landscape_func(position)
        
        for iteration in range(1000):  # Max iterations
            best_neighbor = position.copy()
            best_fitness = current_fitness
            
            # Check neighbors
            for i in range(len(position)):
                for direction in [-1, 1]:
                    neighbor = position.copy()
                    neighbor[i] += direction * step_size
                    
                    # Boundary constraints
                    neighbor[i] = max(0.0, min(1.0, neighbor[i]))
                    
                    neighbor_fitness = landscape_func(neighbor)
                    if neighbor_fitness > best_fitness:
                        best_neighbor = neighbor
                        best_fitness = neighbor_fitness
            
            if best_fitness <= current_fitness:
                break  # Local optimum found
                
            position = best_neighbor
            current_fitness = best_fitness
        
        return position

# Test functions
def test_population_genetics():
    """
    Popülasyon genetiği sistemlerini test eder
    """
    print("=== Wright-Fisher Model Test ===")
    
    # Initial population
    alleles = [
        Allele("A", 0.7, 1.0),
        Allele("a", 0.3, 0.95)
    ]
    
    population = Population(size=100, alleles=alleles)
    wf_model = WrightFisherModel(population_size=100, mutation_rate=1e-5)
    
    print(f"Başlangıç: A={alleles[0].frequency:.3f}, a={alleles[1].frequency:.3f}")
    
    # Simulate 20 generations
    for gen in range(20):
        population = wf_model.simulate_generation(
            population, 
            SelectionType.DIRECTIONAL,
            selection_coefficient=0.02
        )
        
        if gen % 5 == 0 and population.alleles:
            print(f"Nesil {gen}: Allel sayısı={len(population.alleles)}, "
                  f"Ne={population.effective_size}")
    
    print("\n=== Coalescent Theory Test ===")
    coalescent = CoalescentTheory(effective_population_size=1000)
    
    sample_size = 10
    tree = coalescent.simulate_coalescent_tree(sample_size)
    
    print(f"Sample size: {sample_size}")
    print(f"TMRCA: {tree['tmrca']:.2f} generations")
    
    theta = coalescent.calculate_genetic_diversity(sample_size, 1e-8)
    print(f"Genetic diversity (theta): {theta:.6f}")

if __name__ == "__main__":
    test_population_genetics() 