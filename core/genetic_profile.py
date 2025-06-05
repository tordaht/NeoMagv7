"""NeoMag V7 - Genetic Profile Core Module"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

class EvolutionaryForce(Enum):
    """Evrimsel kuvvet türleri"""
    NATURAL_SELECTION = "natural_selection"
    GENETIC_DRIFT = "genetic_drift"
    GENE_FLOW = "gene_flow"
    MUTATION = "mutation"
    EPISTASIS = "epistasis"

@dataclass
class GeneticProfile:
    """Gelişmiş genetik profil"""
    genome_length: int = 1000
    allele_frequencies: Dict[str, float] = field(default_factory=dict)
    mutation_rate: float = 1e-6
    fitness_landscape_position: Tuple[float, ...] = field(default_factory=tuple)
    epistatic_interactions: Dict[Tuple[int, int], float] = field(default_factory=dict)
    gene_expression_levels: Dict[str, float] = field(default_factory=dict)
    regulatory_network: Dict[str, List[str]] = field(default_factory=dict)
    phylogenetic_distance: float = 0.0
    hardy_weinberg_deviation: float = 0.0
    genome_sequence: str = "" # Genom dizisi eklendi

    def __post_init__(self):
        if not self.allele_frequencies and self.genome_length > 0:
            for i in range(self.genome_length):
                self.allele_frequencies[f"locus_{i}"] = np.random.random()
        if not self.fitness_landscape_position and self.genome_length > 0 : # Boş tuple kontrolü
             self.fitness_landscape_position = tuple(np.random.random(min(10, self.genome_length if self.genome_length > 0 else 10)))
        if not self.genome_sequence and self.genome_length > 0:
            self.genome_sequence = "".join(np.random.choice(['A', 'T', 'G', 'C'], size=self.genome_length))
        elif self.genome_sequence and not self.genome_length:
             self.genome_length = len(self.genome_sequence)

    def recalculate_derived_properties(self):
        """Genom dizisinden türetilmiş özellikleri yeniden hesaplar."""
        if self.genome_sequence:
            self.genome_length = len(self.genome_sequence)
            # Basit bir AFS ve fitness pozisyonu güncellemesi (daha karmaşık olabilir)
            # Bu kısım, gerçek biyolojik anlamlılığa göre detaylandırılmalıdır.
            new_afs = {}
            # Örnek: Her 10 bazda bir bir lokus tanımla
            num_loci_from_seq = self.genome_length // 10
            for i in range(num_loci_from_seq):
                # Bu sadece bir örnek, gerçek alel frekansları popülasyondan gelir.
                # Bireysel bir profil için bu alanlar genellikle popülasyon analizinden sonra doldurulur.
                new_afs[f"locus_seq_{i}"] = np.random.random()
            # self.allele_frequencies = new_afs # Var olanı korumak daha iyi olabilir, ya da birleştirme stratejisi lazım.

            # Fitness landscape pozisyonu genomdan türetilebilir (örn: belirli genlerin varlığı)
            # Şimdilik rastgele bırakalım veya basit bir hash fonksiyonu kullanalım.
            # new_fitness_pos_len = min(10, self.genome_length // 100 if self.genome_length >= 100 else 1)
            # if new_fitness_pos_len > 0:
            #    self.fitness_landscape_position = tuple(np.random.random(new_fitness_pos_len))
            pass

    def calculate_gc_content(self) -> float:
        """GC içeriği hesapla"""
        if not self.genome_sequence:
            return 0.0
        gc_count = self.genome_sequence.count('G') + self.genome_sequence.count('C')
        return gc_count / len(self.genome_sequence) if len(self.genome_sequence) > 0 else 0.0

    def calculate_diversity_metrics(self) -> Dict[str, float]:
        """Genetik çeşitlilik metriklerini hesapla"""
        if not self.allele_frequencies:
            return {}
        
        # Expected heterozygosity
        freq_values = list(self.allele_frequencies.values())
        he = 1.0 - sum(f**2 for f in freq_values if 0 <= f <= 1)
        
        # Allelic richness
        allelic_richness = len([f for f in freq_values if f > 0])
        
        return {
            'expected_heterozygosity': he,
            'allelic_richness': allelic_richness,
            'gc_content': self.calculate_gc_content()
        }

    def mutate(self, mutation_rate: Optional[float] = None) -> 'GeneticProfile':
        """Mutasyon uygula"""
        mut_rate = mutation_rate or self.mutation_rate
        
        # Genome sequence mutation
        if self.genome_sequence:
            new_sequence = list(self.genome_sequence)
            for i in range(len(new_sequence)):
                if np.random.random() < mut_rate:
                    new_sequence[i] = np.random.choice(['A', 'T', 'G', 'C'])
            
            # Yeni profil oluştur
            new_profile = GeneticProfile(
                genome_length=self.genome_length,
                allele_frequencies=self.allele_frequencies.copy(),
                mutation_rate=mut_rate,
                fitness_landscape_position=self.fitness_landscape_position,
                epistatic_interactions=self.epistatic_interactions.copy(),
                gene_expression_levels=self.gene_expression_levels.copy(),
                regulatory_network=self.regulatory_network.copy(),
                phylogenetic_distance=self.phylogenetic_distance,
                hardy_weinberg_deviation=self.hardy_weinberg_deviation,
                genome_sequence=''.join(new_sequence)
            )
            
            return new_profile
        
        return self

    def crossover(self, partner: 'GeneticProfile', crossover_rate: float = 0.5) -> 'GeneticProfile':
        """Çaprazlama işlemi"""
        if not self.genome_sequence or not partner.genome_sequence:
            return self
        
        min_length = min(len(self.genome_sequence), len(partner.genome_sequence))
        crossover_point = int(min_length * crossover_rate)
        
        new_sequence = (self.genome_sequence[:crossover_point] + 
                       partner.genome_sequence[crossover_point:min_length])
        
        # Allele frequencies birleştir
        new_allele_freq = {}
        all_loci = set(self.allele_frequencies.keys()) | set(partner.allele_frequencies.keys())
        for locus in all_loci:
            freq1 = self.allele_frequencies.get(locus, 0.0)
            freq2 = partner.allele_frequencies.get(locus, 0.0)
            new_allele_freq[locus] = (freq1 + freq2) / 2.0
        
        return GeneticProfile(
            genome_length=len(new_sequence),
            allele_frequencies=new_allele_freq,
            mutation_rate=(self.mutation_rate + partner.mutation_rate) / 2.0,
            fitness_landscape_position=self.fitness_landscape_position,
            epistatic_interactions=self.epistatic_interactions.copy(),
            gene_expression_levels=self.gene_expression_levels.copy(),
            regulatory_network=self.regulatory_network.copy(),
            phylogenetic_distance=0.0,  # Recalculate
            hardy_weinberg_deviation=0.0,  # Recalculate
            genome_sequence=new_sequence
        )

    def get_tabpfn_features(self) -> np.ndarray:
        """TabPFN için optimize edilmiş genetik feature vektörü"""
        features = []
        
        # Temel genomik özellikler
        features.extend([
            self.genome_length / 10000.0,  # Normalize
            self.mutation_rate * 1e6,  # Scale up
            self.phylogenetic_distance,
            self.hardy_weinberg_deviation,
            self.calculate_gc_content()
        ])
        
        # Fitness landscape position (ilk 10 boyut)
        fitness_pos = list(self.fitness_landscape_position)[:10]
        features.extend(fitness_pos + [0.0] * (10 - len(fitness_pos)))
        
        # Allele frequencies summary (top 20)
        if self.allele_frequencies:
            freq_values = sorted(self.allele_frequencies.values(), reverse=True)[:20]
            features.extend(freq_values + [0.0] * (20 - len(freq_values)))
        else:
            features.extend([0.0] * 20)
        
        # Gene expression levels summary (top 15)
        if self.gene_expression_levels:
            expr_values = sorted(self.gene_expression_levels.values(), reverse=True)[:15]
            features.extend(expr_values + [0.0] * (15 - len(expr_values)))
        else:
            features.extend([0.0] * 15)
        
        return np.array(features, dtype=np.float32)

    def get_state_vector(self) -> np.ndarray:
        """Durum vektörü al (AI/ML için)"""
        return self.get_tabpfn_features()

    def calculate_hamming_distance(self, other: 'GeneticProfile') -> float:
        """İki genetik profil arasındaki Hamming mesafesi"""
        if not self.genome_sequence or not other.genome_sequence:
            return float('inf')
        
        min_length = min(len(self.genome_sequence), len(other.genome_sequence))
        if min_length == 0:
            return float('inf')
        
        differences = sum(c1 != c2 for c1, c2 in zip(
            self.genome_sequence[:min_length], 
            other.genome_sequence[:min_length]
        ))
        
        return differences / min_length
