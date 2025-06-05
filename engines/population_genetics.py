"""NeoMag V7 - Population Genetics Engine"""

import numpy as np
import cupy as cp
import pandas as pd
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import sys
sys.path.append('..')

try:
    from ..core.genetic_profile import GeneticProfile, EvolutionaryForce
except ImportError:
    from core.genetic_profile import GeneticProfile, EvolutionaryForce

class PopulationGeneticsEngine:
    """Gelişmiş popülasyon genetiği motoru"""
    def __init__(self, mutation_rate=1e-6, recombination_rate=1e-7, selection_coefficient_dist='lognormal',
                 effective_population_size_model='constant', use_gpu=False,
                 abc_simulation_count=1000, abc_summary_statistics=None, ml_model_path=None):
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.selection_coefficient_dist = selection_coefficient_dist
        self.effective_population_size_model = effective_population_size_model
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        self.abc_simulation_count = abc_simulation_count
        self.abc_summary_statistics_calculators = {
            'allele_frequency_spectrum': self._calculate_afs,
            'tajimas_d': self._calculate_tajimas_d,
            'nucleotide_diversity': self._calculate_pi
        }
        self.active_abc_stats = abc_summary_statistics if abc_summary_statistics else ['allele_frequency_spectrum', 'tajimas_d']
        self.ml_model_path = ml_model_path
        self.ml_model = None
        if self.ml_model_path: 
            self.load_ml_model()
        self.genetic_calculations = 0
        self.last_calculation_time = 0.0
        logging.info(f"Population Genetics Engine initialized. Active ABC stats: {self.active_abc_stats}")

    def evolve_population(self, bacteria_population: List, generations=1) -> List:
        """Popülasyonu belirli bir jenerasyon boyunca evrimleştirir."""
        if not bacteria_population: 
            return []

        for _ in range(generations):
            start_time = time.perf_counter()
            num_individuals = len(bacteria_population)
            if num_individuals == 0: 
                break

            # 1. Seçilim (Fitness'a göre üreme olasılıkları)
            fitness_values = np.array([b.calculate_fitness() for b in bacteria_population])
            if np.sum(fitness_values) == 0:
                if num_individuals > 0:
                    probabilities = np.ones(num_individuals) / num_individuals
                else:
                    break
            else:
                probabilities = fitness_values / np.sum(fitness_values)

            # Yeni jenerasyon için ebeveynleri seç (Wright-Fisher benzeri)
            chosen_indices = np.random.choice(num_individuals, size=num_individuals, p=probabilities, replace=True)
            mating_pool = [bacteria_population[i] for i in chosen_indices]

            next_generation = []
            for i in range(num_individuals):
                parent1 = mating_pool[i]
                parent2 = None
                if self.recombination_rate > 0 and num_individuals > 1:
                    potential_partners = [p for idx, p in enumerate(mating_pool) if idx != i]
                    if potential_partners:
                        parent2 = np.random.choice(potential_partners)

                # Yavru oluşturma
                offspring = parent1.reproduce(partner=parent2, pop_gen_engine=self)
                if offspring:
                    next_generation.append(offspring)

            bacteria_population = next_generation
            if not bacteria_population: 
                break

            self.genetic_calculations += 1
            self.last_calculation_time = time.perf_counter() - start_time
        return bacteria_population

    def perform_abc_analysis(self, observed_data_summary_stats: Dict[str, Any],
                             prior_distributions: Dict[str, Tuple[str, float, float]],
                             tolerance_schedule: Optional[List[float]] = None) -> pd.DataFrame:
        """ABC analizi gerçekleştir"""
        if not self.active_abc_stats:
            logging.warning("ABC için aktif özet istatistikler tanımlanmamış.")
            return pd.DataFrame()

        accepted_parameters_list = []
        default_tolerance = 0.1

        def _run_simplified_simulation_for_abc(params: Dict[str, float], num_individuals=50, num_generations=100, genome_len_for_sim=100) -> List[GeneticProfile]:
            temp_population_profiles = []
            for _ in range(num_individuals):
                gp = GeneticProfile(genome_length=genome_len_for_sim, mutation_rate=params.get('mutation_rate', self.mutation_rate))
                temp_population_profiles.append(gp)

            current_mut_rate = params.get('mutation_rate', self.mutation_rate)

            for _ in range(num_generations):
                next_gen_profiles = []
                if not temp_population_profiles: 
                    break
                for profile in temp_population_profiles:
                    new_seq_list = list(profile.genome_sequence)
                    for i in range(len(new_seq_list)):
                        if self.xp.random.rand() < current_mut_rate:
                            original_base = new_seq_list[i]
                            possible_mutations = [b for b in 'ATGC' if b != original_base]
                            if possible_mutations:
                                new_seq_list[i] = self.xp.random.choice(possible_mutations)
                    
                    offspring_profile = GeneticProfile(genome_sequence="".join(new_seq_list), mutation_rate=current_mut_rate)
                    offspring_profile.recalculate_derived_properties()
                    next_gen_profiles.append(offspring_profile)
                temp_population_profiles = next_gen_profiles
            return temp_population_profiles

        for sim_idx in range(self.abc_simulation_count):
            simulated_params = {}
            for param_name, dist_info in prior_distributions.items():
                dist_type, val1, val2 = dist_info[0], dist_info[1], dist_info[2]
                if dist_type == 'uniform': 
                    simulated_params[param_name] = self.xp.random.uniform(val1, val2)
                elif dist_type == 'loguniform': 
                    simulated_params[param_name] = self.xp.exp(self.xp.random.uniform(self.xp.log(val1), self.xp.log(val2)))
                elif dist_type == 'normal': 
                    simulated_params[param_name] = self.xp.random.normal(val1, val2)
                else: 
                    raise ValueError(f"Desteklenmeyen dağılım tipi: {dist_type}")

            genome_len_for_sim = int(simulated_params.get('genome_length_for_sim', 100))
            simulated_population_profiles = _run_simplified_simulation_for_abc(simulated_params, genome_len_for_sim=genome_len_for_sim)
            simulated_summary_stats = self._calculate_summary_stats_for_abc(simulated_population_profiles)

            distance = 0
            num_valid_stats = 0
            for stat_name in self.active_abc_stats:
                obs_stat = observed_data_summary_stats.get(stat_name)
                sim_stat = simulated_summary_stats.get(stat_name)
                if obs_stat is not None and sim_stat is not None:
                    if isinstance(obs_stat, (int, float, np.number, cp.number)):
                        dist_val = ((obs_stat - sim_stat) / (self.xp.abs(obs_stat) + 1e-9))**2
                    elif isinstance(obs_stat, (np.ndarray, cp.ndarray)):
                        if obs_stat.shape == sim_stat.shape and obs_stat.size > 0:
                             dist_val = self.xp.sum(((obs_stat - sim_stat) / (self.xp.abs(obs_stat) + 1e-9))**2) / obs_stat.size
                        else:
                            logging.debug(f"İstatistik '{stat_name}' şekil uyuşmazlığı: gözlenen={obs_stat.shape}, simüle={sim_stat.shape}")
                            continue
                    else: 
                        continue
                    distance += dist_val
                    num_valid_stats +=1
                    
            current_distance = self.xp.sqrt(distance / num_valid_stats) if num_valid_stats > 0 else float('inf')
            current_tolerance = tolerance_schedule[0] if tolerance_schedule else default_tolerance
            if current_distance < current_tolerance:
                param_entry = simulated_params.copy()
                param_entry['distance'] = float(cp.asnumpy(current_distance)) if self.use_gpu else float(current_distance)
                accepted_parameters_list.append(param_entry)
                
            if (sim_idx + 1) % (self.abc_simulation_count // 10 or 1) == 0:
                logging.info(f"ABC Simülasyonu: {sim_idx+1}/{self.abc_simulation_count}, Kabul Edilen: {len(accepted_parameters_list)}")
                
        return pd.DataFrame(accepted_parameters_list)

    def _calculate_summary_stats_for_abc(self, genetic_profiles: List[GeneticProfile]) -> Dict[str, Any]:
        """ABC için özet istatistik hesapla"""
        summary_stats = {}
        if not genetic_profiles:
            for stat_name in self.active_abc_stats: 
                summary_stats[stat_name] = None
            return summary_stats
            
        for stat_name in self.active_abc_stats:
            calculator = self.abc_summary_statistics_calculators.get(stat_name)
            if calculator:
                try: 
                    summary_stats[stat_name] = calculator(genetic_profiles)
                except Exception as e:
                    logging.error(f"Özet istatistik '{stat_name}' hesaplanırken hata: {e}")
                    summary_stats[stat_name] = None
            else: 
                summary_stats[stat_name] = None
        return summary_stats

    def _get_sequences_from_profiles(self, genetic_profiles: List[GeneticProfile]) -> List[str]:
        """Genetik profillerden sekansları al"""
        sequences = [p.genome_sequence for p in genetic_profiles if hasattr(p, 'genome_sequence') and p.genome_sequence]
        if not sequences: 
            logging.warning("Hiçbir genetik profilde 'genome_sequence' bulunamadı.")
        return sequences

    def _calculate_afs(self, genetic_profiles: List[GeneticProfile]) -> Optional[np.ndarray]:
        """Allele Frequency Spectrum hesapla"""
        sequences = self._get_sequences_from_profiles(genetic_profiles)
        if not sequences: 
            return None
        num_sequences = len(sequences)
        if num_sequences < 2: 
            return None

        seq_len = 0
        if sequences:
            seq_len = len(sequences[0])
            if not all(len(s) == seq_len for s in sequences):
                logging.warning("AFS: Sekans uzunlukları farklı.")
                seq_len = min(len(s) for s in sequences)

        if seq_len == 0: 
            return None

        afs = self.xp.zeros(num_sequences, dtype=int)

        for i in range(seq_len):
            try:
                ancestral_allele = sequences[0][i]
                derived_allele_count = 0
                for seq_idx in range(num_sequences):
                    if sequences[seq_idx][i] != ancestral_allele:
                        derived_allele_count += 1
                
                if 0 <= derived_allele_count < num_sequences:
                    afs[derived_allele_count] += 1
            except IndexError:
                continue

        return cp.asnumpy(afs) if self.use_gpu else afs

    def _calculate_tajimas_d(self, genetic_profiles: List[GeneticProfile]) -> Optional[float]:
        """Tajima's D hesapla"""
        sequences = self._get_sequences_from_profiles(genetic_profiles)
        if len(sequences) < 4:
            return None

        # Basitleştirilmiş Tajima's D
        try:
            seq_len = min(len(s) for s in sequences)
            n = len(sequences)
            
            # π (nucleotide diversity)
            total_differences = 0
            comparisons = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    differences = sum(sequences[i][k] != sequences[j][k] for k in range(seq_len))
                    total_differences += differences
                    comparisons += 1
            
            pi = total_differences / (comparisons * seq_len) if comparisons > 0 else 0
            
            # S (segregating sites)
            S = 0
            for pos in range(seq_len):
                alleles = set(seq[pos] for seq in sequences)
                if len(alleles) > 1:
                    S += 1
            
            # Basit Tajima's D yaklaşımı
            a1 = sum(1/i for i in range(1, n))
            theta_w = S / a1 if a1 > 0 else 0
            
            if theta_w > 0:
                D = (pi - theta_w) / np.sqrt(theta_w)
            else:
                D = 0
                
            return float(D)
            
        except Exception as e:
            logging.error(f"Tajima's D hesaplanırken hata: {e}")
            return None

    def _calculate_pi(self, genetic_profiles: List[GeneticProfile]) -> Optional[float]:
        """Nucleotide diversity (π) hesapla"""
        sequences = self._get_sequences_from_profiles(genetic_profiles)
        if len(sequences) < 2:
            return None

        try:
            seq_len = min(len(s) for s in sequences)
            n = len(sequences)
            
            total_differences = 0
            comparisons = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    differences = sum(sequences[i][k] != sequences[j][k] for k in range(seq_len))
                    total_differences += differences
                    comparisons += 1
            
            pi = total_differences / (comparisons * seq_len) if comparisons > 0 else 0
            return float(pi)
            
        except Exception as e:
            logging.error(f"π hesaplanırken hata: {e}")
            return None

    def load_ml_model(self):
        """ML model yükle"""
        try:
            # ML model yükleme implementasyonu
            logging.info(f"ML model {self.ml_model_path} yükleniyor...")
            # self.ml_model = joblib.load(self.ml_model_path)
        except Exception as e:
            logging.error(f"ML model yüklenirken hata: {e}")
            self.ml_model = None

    def predict_with_ml_model(self, input_data: Any) -> Optional[Any]:
        """ML model ile tahmin"""
        if self.ml_model is None:
            return None
        try:
            return self.ml_model.predict(input_data)
        except Exception as e:
            logging.error(f"ML tahmin hatası: {e}")
            return None

    def calculate_fitness_landscape_position(self, genetic_profile: GeneticProfile) -> float:
        """Fitness landscape pozisyonu hesapla"""
        if not genetic_profile.fitness_landscape_position:
            return 0.0
        return float(np.mean(genetic_profile.fitness_landscape_position))

    def calculate_genetic_diversity_metrics(self, genetic_profiles: List[GeneticProfile]) -> Dict[str, Any]:
        """Genetik çeşitlilik metriklerini hesapla"""
        if not genetic_profiles:
            return {}
        
        metrics = {}
        
        # Expected heterozygosity
        all_freqs = []
        for profile in genetic_profiles:
            all_freqs.extend(profile.allele_frequencies.values())
        
        if all_freqs:
            he = 1.0 - sum(f**2 for f in all_freqs) / len(all_freqs)
            metrics['expected_heterozygosity'] = he
        
        # Allelic richness
        all_alleles = set()
        for profile in genetic_profiles:
            all_alleles.update(profile.allele_frequencies.keys())
        metrics['allelic_richness'] = len(all_alleles)
        
        # Nucleotide diversity
        pi = self._calculate_pi(genetic_profiles)
        if pi is not None:
            metrics['nucleotide_diversity'] = pi
        
        return metrics
