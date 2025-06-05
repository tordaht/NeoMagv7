#!/usr/bin/env python3
"""
NeoMag V7 - Gerçek Performans Benchmark Aracı
Bu script tüm motorların gerçek hesaplama sürelerini ölçer
"""

import time
import sys
import numpy as np
from pathlib import Path

print('🔬 NeoMag V7 Gerçek Performans Ölçümü')
print('='*60)

def benchmark_molecular_dynamics():
    """Moleküler Dinamik Motor Performance Test"""
    try:
        from molecular_dynamics_engine import MolecularDynamicsEngine, AtomicPosition
        
        md = MolecularDynamicsEngine(temperature=310.0, dt=0.001)
        
        # Test positions - tipik bakteriyel çevre
        positions = [
            AtomicPosition([0, 0, 0], charge=1.0, mass=1e-26),
            AtomicPosition([1, 1, 1], charge=-1.0, mass=1e-26),
            AtomicPosition([2, 0, 1], charge=0.5, mass=2e-26),
            AtomicPosition([0, 2, 1], charge=-0.5, mass=1.5e-26)
        ]
        
        # 100 step hesaplama
        times = []
        for i in range(100):
            start = time.perf_counter()
            forces = md.calculate_intermolecular_forces(positions)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f'✅ Molecular Dynamics:')
        print(f'   Ortalama: {avg_time:.6f} s/step')
        print(f'   Std Dev:  {std_time:.6f} s')
        print(f'   Min/Max:  {min(times):.6f}/{max(times):.6f} s')
        
        return avg_time
        
    except Exception as e:
        print(f'❌ MD Benchmark Error: {e}')
        return None

def benchmark_population_genetics():
    """Popülasyon Genetiği Motor Performance Test"""
    try:
        from population_genetics_engine import WrightFisherModel, Population, Allele, SelectionType
        
        wf = WrightFisherModel(population_size=100, mutation_rate=1e-5)
        
        alleles = [
            Allele("A1", 0.6, 1.0),  # High fitness allele
            Allele("A2", 0.4, 0.9),  # Lower fitness allele
            Allele("A3", 0.3, 0.8)   # Lowest fitness allele
        ]
        pop = Population(size=100, alleles=alleles)
        
        # 100 generation simulation
        times = []
        for i in range(100):
            start = time.perf_counter()
            pop = wf.simulate_generation(pop, SelectionType.DIRECTIONAL, 0.01)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f'✅ Wright-Fisher Model:')
        print(f'   Ortalama: {avg_time:.6f} s/generation')
        print(f'   Std Dev:  {std_time:.6f} s')
        print(f'   Min/Max:  {min(times):.6f}/{max(times):.6f} s')
        
        return avg_time
        
    except Exception as e:
        print(f'❌ Population Genetics Benchmark Error: {e}')
        return None

def benchmark_tabpfn():
    """TabPFN ML Model Performance Test"""
    try:
        from tabpfn import TabPFNRegressor
        
        # Çeşitli ensemble boyutları test et
        ensemble_configs = [1, 8, 16]
        results = {}
        
        for n_ensemble in ensemble_configs:
            model = TabPFNRegressor(device='cpu', n_estimators=n_ensemble)
            
            # Tipik bakteri feature'ları simulate et
            X = np.random.rand(100, 6)  # 100 bacteria, 6 features
            y = np.random.rand(100)     # fitness values
            
            times = []
            for i in range(10):  # 10 prediction cycle
                start = time.perf_counter()
                model.fit(X, y)
                predictions = model.predict(X[:20])  # Predict on subset
                times.append(time.perf_counter() - start)
            
            avg_time = np.mean(times)
            results[n_ensemble] = avg_time
            
            print(f'✅ TabPFN (Ensemble={n_ensemble}):')
            print(f'   Ortalama: {avg_time:.6f} s/prediction')
            print(f'   Throughput: {1/avg_time:.1f} predictions/sec')
        
        return results
        
    except Exception as e:
        print(f'❌ TabPFN Benchmark Error: {e}')
        return None

def benchmark_reinforcement_learning():
    """RL Motor Performance Test"""
    try:
        from reinforcement_learning_engine import EcosystemManager, EcosystemState, Action
        
        ecosystem = EcosystemManager()
        
        # Tipik ecosystem state - real simulation parameters
        state = EcosystemState()
        state.avg_fitness = 0.7
        state.genetic_diversity = 0.3
        state.environmental_stress = 0.4
        state.resource_availability = 0.8
        state.time_step = 100
        
        times = []
        for i in range(100):
            start = time.perf_counter()
            action = ecosystem.get_recommended_action(state)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f'✅ RL Ecosystem Manager:')
        print(f'   Ortalama: {avg_time:.6f} s/decision')
        print(f'   Std Dev:  {std_time:.6f} s')
        print(f'   Decisions/sec: {1/avg_time:.1f}')
        
        return avg_time
        
    except Exception as e:
        print(f'❌ RL Benchmark Error: {e}')
        return None

def system_info():
    """Sistem bilgilerini topla"""
    import platform
    import psutil
    
    print('🖥️ Sistem Bilgileri:')
    print(f'   Platform: {platform.platform()}')
    print(f'   Python: {platform.python_version()}')
    print(f'   CPU: {platform.processor()}')
    print(f'   CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical')
    print(f'   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
    print()

if __name__ == "__main__":
    system_info()
    
    # Tüm benchmark'ları çalıştır
    md_time = benchmark_molecular_dynamics()
    print()
    
    wf_time = benchmark_population_genetics()
    print()
    
    tabpfn_results = benchmark_tabpfn()
    print()
    
    rl_time = benchmark_reinforcement_learning()
    print()
    
    # Özet sonuçlar
    print('📊 ÖZET PERFORMANS METRİKLERİ:')
    print('='*60)
    
    if md_time:
        print(f'Molecular Dynamics:    {md_time:.6f} s/step')
    if wf_time:
        print(f'Population Genetics:   {wf_time:.6f} s/generation')
    if tabpfn_results and 16 in tabpfn_results:
        print(f'TabPFN (16-ensemble):  {tabpfn_results[16]:.6f} s/prediction')
    if rl_time:
        print(f'RL Decision Engine:    {rl_time:.6f} s/decision')
    
    print('='*60)
    print('✅ Benchmark tamamlandı!') 