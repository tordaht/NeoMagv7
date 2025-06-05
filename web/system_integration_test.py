# NeoMag V7 - Kapsamlı Sistem Entegrasyon Testi
# Pleksus Görselliği ile Fonksiyon Bağlantıları Analizi

import sys
import os
import importlib
import inspect
import traceback
import time
import numpy as np
from pathlib import Path
import json

class SystemIntegrationTest:
    """
    3D uzay benzeri pleksus analizi ile sistem bütünlüğünü test eder
    """
    
    def __init__(self):
        self.test_results = {
            'modules': {},
            'functions': {},
            'connections': {},
            'broken_links': [],
            'missing_dependencies': [],
            'performance': {},
            'integration_score': 0
        }
        self.modules_to_test = [
            'molecular_dynamics_engine',
            'population_genetics_engine', 
            'reinforcement_learning_engine',
            'advanced_bacterium_v7',
            'web_server'
        ]
        
    def run_complete_test(self):
        """Ana test süreci - tüm analizleri çalıştır"""
        print("=" * 60)
        print("🧪 NeoMag V7 SİSTEM ENTEGRASYON TESTİ")
        print("=" * 60)
        
        # Test aşamaları
        self.test_module_imports()
        self.test_function_connectivity()
        self.test_class_instantiation()
        self.test_cross_module_dependencies()
        self.test_real_simulation_flow()
        self.test_performance_benchmarks()
        self.generate_plexus_visualization()
        self.calculate_integration_score()
        
        # Sonuçları raporla
        self.generate_final_report()
        
    def test_module_imports(self):
        """1. Modül Import Testi"""
        print("\n🔍 1. MODÜL IMPORT ANALİZİ")
        print("-" * 30)
        
        for module_name in self.modules_to_test:
            try:
                print(f"Testing {module_name}...")
                module = importlib.import_module(module_name)
                
                # Modül içeriğini analiz et
                classes = []
                functions = []
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and obj.__module__ == module_name:
                        classes.append(name)
                    elif inspect.isfunction(obj) and obj.__module__ == module_name:
                        functions.append(name)
                
                self.test_results['modules'][module_name] = {
                    'status': 'SUCCESS',
                    'classes': classes,
                    'functions': functions,
                    'total_members': len(classes) + len(functions)
                }
                
                print(f"  ✅ {module_name}: {len(classes)} class, {len(functions)} function")
                
            except Exception as e:
                self.test_results['modules'][module_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'classes': [],
                    'functions': []
                }
                print(f"  ❌ {module_name}: {e}")
                self.test_results['broken_links'].append(f"Module import: {module_name}")
    
    def test_function_connectivity(self):
        """2. Fonksiyon Bağlantılılığı Testi"""
        print("\n🔗 2. FONKSİYON BAĞLANTILILIK ANALİZİ")
        print("-" * 30)
        
        # Critical function connections to test
        critical_connections = [
            ('web_server', 'NeoMagV7WebSimulation', 'initialize_engines'),
            ('web_server', 'NeoMagV7WebSimulation', 'start_simulation'),
            ('molecular_dynamics_engine', 'MolecularDynamicsEngine', 'calculate_forces'),
            ('population_genetics_engine', 'WrightFisherModel', 'simulate_generation'),
            ('reinforcement_learning_engine', 'EcosystemManager', 'train_agent'),
            ('advanced_bacterium_v7', 'AdvancedBacteriumV7', 'update_biophysics')
        ]
        
        for module_name, class_name, method_name in critical_connections:
            try:
                if module_name in self.test_results['modules'] and \
                   self.test_results['modules'][module_name]['status'] == 'SUCCESS':
                    
                    module = importlib.import_module(module_name)
                    cls = getattr(module, class_name)
                    method = getattr(cls, method_name)
                    
                    # Method signature analizi
                    sig = inspect.signature(method)
                    params = list(sig.parameters.keys())
                    
                    connection_key = f"{module_name}.{class_name}.{method_name}"
                    self.test_results['connections'][connection_key] = {
                        'status': 'CONNECTED',
                        'parameters': params,
                        'parameter_count': len(params)
                    }
                    
                    print(f"  ✅ {connection_key}: {len(params)} parametres")
                    
                else:
                    connection_key = f"{module_name}.{class_name}.{method_name}"
                    self.test_results['connections'][connection_key] = {
                        'status': 'BROKEN - MODULE FAILED'
                    }
                    print(f"  ❌ {connection_key}: Module import failed")
                    
            except AttributeError as e:
                connection_key = f"{module_name}.{class_name}.{method_name}"
                self.test_results['connections'][connection_key] = {
                    'status': 'BROKEN - MISSING',
                    'error': str(e)
                }
                print(f"  ❌ {connection_key}: Missing - {e}")
                self.test_results['broken_links'].append(connection_key)
                
            except Exception as e:
                connection_key = f"{module_name}.{class_name}.{method_name}"
                self.test_results['connections'][connection_key] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print(f"  ⚠️ {connection_key}: Error - {e}")
    
    def test_class_instantiation(self):
        """3. Sınıf Örnekleme Testi"""
        print("\n🏗️ 3. SINIF ÖRNEKLEME TESTİ")
        print("-" * 30)
        
        instantiation_tests = [
            ('molecular_dynamics_engine', 'MolecularDynamicsEngine', {'temperature': 310.0, 'dt': 0.001}),
            ('population_genetics_engine', 'WrightFisherModel', {'population_size': 100, 'mutation_rate': 1e-5}),
            ('reinforcement_learning_engine', 'EcosystemManager', {}),
            ('advanced_bacterium_v7', 'AdvancedBacteriumV7', {'x': 0, 'y': 0, 'z': 0}),
            ('web_server', 'NeoMagV7WebSimulation', {})
        ]
        
        for module_name, class_name, kwargs in instantiation_tests:
            try:
                if module_name in self.test_results['modules'] and \
                   self.test_results['modules'][module_name]['status'] == 'SUCCESS':
                    
                    module = importlib.import_module(module_name)
                    cls = getattr(module, class_name)
                    
                    start_time = time.time()
                    instance = cls(**kwargs)
                    instantiation_time = time.time() - start_time
                    
                    # Instance metodlarını kontrol et
                    methods = [name for name, method in inspect.getmembers(instance) 
                              if inspect.ismethod(method) and not name.startswith('_')]
                    
                    self.test_results['functions'][f"{module_name}.{class_name}"] = {
                        'status': 'SUCCESS',
                        'instantiation_time': instantiation_time,
                        'methods': methods,
                        'method_count': len(methods)
                    }
                    
                    print(f"  ✅ {class_name}: {instantiation_time:.4f}s, {len(methods)} methods")
                    
                else:
                    self.test_results['functions'][f"{module_name}.{class_name}"] = {
                        'status': 'FAILED - MODULE IMPORT'
                    }
                    print(f"  ❌ {class_name}: Module import failed")
                    
            except Exception as e:
                self.test_results['functions'][f"{module_name}.{class_name}"] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"  ❌ {class_name}: {e}")
                self.test_results['broken_links'].append(f"Instantiation: {class_name}")
    
    def test_cross_module_dependencies(self):
        """4. Modüller Arası Bağımlılık Testi"""
        print("\n🔄 4. MODÜLLER ARASI BAĞIMLILIK TESTİ")
        print("-" * 30)
        
        # Web server motor entegrasyonu testi
        try:
            import web_server
            sim = web_server.NeoMagV7WebSimulation()
            
            # Motor başlatma testi
            start_time = time.time()
            engine_result = sim.initialize_engines()
            init_time = time.time() - start_time
            
            if engine_result:
                print(f"  ✅ Engine initialization: {init_time:.4f}s")
                
                # Motor bağlantılarını kontrol et
                motors = []
                if hasattr(sim, 'md_engine') and sim.md_engine:
                    motors.append('MolecularDynamics')
                if hasattr(sim, 'wf_model') and sim.wf_model:
                    motors.append('WrightFisher')
                if hasattr(sim, 'ecosystem_manager') and sim.ecosystem_manager:
                    motors.append('EcosystemManager')
                
                self.test_results['performance']['engine_init_time'] = init_time
                self.test_results['performance']['active_engines'] = motors
                
                print(f"  ✅ Active engines: {', '.join(motors)}")
                
            else:
                print(f"  ⚠️ Engine initialization returned False")
                self.test_results['broken_links'].append("Engine initialization failed")
                
        except Exception as e:
            print(f"  ❌ Cross-module dependency test failed: {e}")
            self.test_results['broken_links'].append(f"Cross-module: {e}")
    
    def test_real_simulation_flow(self):
        """5. Gerçek Simülasyon Akışı Testi"""
        print("\n⚙️ 5. SİMÜLASYON AKIŞ TESTİ")
        print("-" * 30)
        
        try:
            import web_server
            sim = web_server.NeoMagV7WebSimulation()
            
            # Engines başlat
            sim.initialize_engines()
            
            # Kısa simülasyon çalıştır
            print("  🚀 Starting mini simulation...")
            start_time = time.time()
            
            # Simülasyonu başlat
            sim_started = sim.start_simulation(initial_bacteria=10)
            
            if sim_started:
                print(f"  ✅ Simulation started with {len(sim.bacteria_population)} bacteria")
                
                # 5 saniye çalıştır
                time.sleep(2)
                
                # Veri topla
                sim_data = sim.get_simulation_data()
                
                # Durdur
                sim.stop_simulation()
                
                test_time = time.time() - start_time
                
                self.test_results['performance']['simulation_test_time'] = test_time
                self.test_results['performance']['bacteria_count'] = sim_data.get('bacteria_count', 0)
                self.test_results['performance']['simulation_steps'] = sim_data.get('time_step', 0)
                
                print(f"  ✅ Simulation flow test: {test_time:.2f}s")
                print(f"  ✅ Bacteria count: {sim_data.get('bacteria_count', 0)}")
                print(f"  ✅ Steps completed: {sim_data.get('time_step', 0)}")
                
            else:
                print(f"  ❌ Simulation failed to start")
                self.test_results['broken_links'].append("Simulation start failed")
                
        except Exception as e:
            print(f"  ❌ Simulation flow test failed: {e}")
            self.test_results['broken_links'].append(f"Simulation flow: {e}")
    
    def test_performance_benchmarks(self):
        """6. Performans Benchmark Testi"""
        print("\n⚡ 6. PERFORMANS BENCHMARK TESTİ")
        print("-" * 30)
        
        benchmarks = []
        
        # Molecular Dynamics benchmark
        try:
            import molecular_dynamics_engine
            md = molecular_dynamics_engine.MolecularDynamicsEngine()
            
            positions = [molecular_dynamics_engine.AtomicPosition(
                x=np.random.uniform(-10, 10),
                y=np.random.uniform(-10, 10),
                z=np.random.uniform(-10, 10)
            ) for _ in range(100)]
            
            start_time = time.time()
            for _ in range(10):
                forces = md.calculate_forces(positions)
            md_time = (time.time() - start_time) / 10
            
            benchmarks.append(f"MD Force Calc: {md_time:.6f}s/iteration")
            print(f"  ✅ MD Force Calculation: {md_time:.6f}s")
            
        except Exception as e:
            print(f"  ❌ MD Benchmark failed: {e}")
        
        # Population Genetics benchmark
        try:
            import population_genetics_engine
            wf = population_genetics_engine.WrightFisherModel(100, 1e-5)
            
            from population_genetics_engine import Population, Allele
            pop = Population(size=100, alleles=[
                Allele("A1", 0.6, 1.0),
                Allele("A2", 0.4, 0.9)
            ])
            
            start_time = time.time()
            for _ in range(10):
                pop = wf.simulate_generation(pop, 
                    population_genetics_engine.SelectionType.NEUTRAL, 0.0)
            wf_time = (time.time() - start_time) / 10
            
            benchmarks.append(f"WF Generation: {wf_time:.6f}s/generation")
            print(f"  ✅ Wright-Fisher Generation: {wf_time:.6f}s")
            
        except Exception as e:
            print(f"  ❌ WF Benchmark failed: {e}")
        
        # Bacterium simulation benchmark
        try:
            import advanced_bacterium_v7
            bacteria = [advanced_bacterium_v7.AdvancedBacteriumV7(
                x=np.random.uniform(-50, 50),
                y=np.random.uniform(-50, 50),
                z=np.random.uniform(-10, 10)
            ) for _ in range(50)]
            
            env_params = {
                'glucose': 0.7, 'oxygen': 0.8, 'ph': 7.0,
                'temperature': 37.0, 'toxin_level': 0.1, 'viscosity': 1e-3
            }
            
            start_time = time.time()
            for bacterium in bacteria:
                forces = bacterium.calculate_molecular_forces(bacteria, env_params)
                bacterium.update_biophysics(0.1, forces)
                bacterium.metabolic_update(env_params, 0.1)
            bacterium_time = time.time() - start_time
            
            benchmarks.append(f"Bacterium Update: {bacterium_time:.6f}s/50bacteria")
            print(f"  ✅ Bacterium Update (50): {bacterium_time:.6f}s")
            
        except Exception as e:
            print(f"  ❌ Bacterium Benchmark failed: {e}")
        
        self.test_results['performance']['benchmarks'] = benchmarks
    
    def generate_plexus_visualization(self):
        """7. Pleksus Görselleştirme - Bağlantı Ağacı"""
        print("\n🕸️ 7. PLEKSUS BAĞLANTI HARİTASI")
        print("-" * 30)
        
        # ASCII art network görselleştirmesi
        connections = []
        broken_count = 0
        
        for module_name, module_data in self.test_results['modules'].items():
            if module_data['status'] == 'SUCCESS':
                connections.append(f"🟢 {module_name}")
                for class_name in module_data['classes']:
                    connections.append(f"  ├─ 📦 {class_name}")
                for func_name in module_data['functions']:
                    connections.append(f"  ├─ ⚙️ {func_name}")
            else:
                connections.append(f"🔴 {module_name} (BROKEN)")
                broken_count += 1
        
        # Bağlantı durumu
        total_connections = len(self.test_results['connections'])
        working_connections = sum(1 for conn in self.test_results['connections'].values() 
                                if conn['status'] == 'CONNECTED')
        
        print("  System Network Map:")
        for conn in connections[:10]:  # İlk 10 satırı göster
            print(f"    {conn}")
        
        if len(connections) > 10:
            print(f"    ... ve {len(connections) - 10} tane daha")
        
        print(f"\n  📊 Network Stats:")
        print(f"    Working Connections: {working_connections}/{total_connections}")
        print(f"    Broken Links: {len(self.test_results['broken_links'])}")
        print(f"    Module Success Rate: {((len(self.test_results['modules']) - broken_count) / len(self.test_results['modules']) * 100):.1f}%")
    
    def calculate_integration_score(self):
        """8. Entegrasyon Skoru Hesaplama"""
        print("\n📊 8. ENTEGRASYON SKORU HESAPLAMA")
        print("-" * 30)
        
        scores = {
            'module_import': 0,
            'function_connectivity': 0,
            'class_instantiation': 0,
            'performance': 0,
            'simulation_flow': 0
        }
        
        # Modül import skoru
        successful_modules = sum(1 for m in self.test_results['modules'].values() 
                               if m['status'] == 'SUCCESS')
        scores['module_import'] = (successful_modules / len(self.test_results['modules'])) * 20
        
        # Fonksiyon bağlantı skoru
        working_connections = sum(1 for c in self.test_results['connections'].values() 
                                if c['status'] == 'CONNECTED')
        if self.test_results['connections']:
            scores['function_connectivity'] = (working_connections / len(self.test_results['connections'])) * 25
        
        # Sınıf örnekleme skoru
        successful_classes = sum(1 for f in self.test_results['functions'].values() 
                               if f['status'] == 'SUCCESS')
        if self.test_results['functions']:
            scores['class_instantiation'] = (successful_classes / len(self.test_results['functions'])) * 20
        
        # Performans skoru
        if 'benchmarks' in self.test_results['performance']:
            scores['performance'] = min(len(self.test_results['performance']['benchmarks']) * 5, 15)
        
        # Simülasyon akış skoru
        if 'simulation_test_time' in self.test_results['performance']:
            scores['simulation_flow'] = 20
        
        total_score = sum(scores.values())
        self.test_results['integration_score'] = total_score
        
        print(f"  Module Import Score: {scores['module_import']:.1f}/20")
        print(f"  Function Connectivity: {scores['function_connectivity']:.1f}/25")
        print(f"  Class Instantiation: {scores['class_instantiation']:.1f}/20")
        print(f"  Performance Benchmarks: {scores['performance']:.1f}/15")
        print(f"  Simulation Flow: {scores['simulation_flow']:.1f}/20")
        print(f"  " + "="*30)
        print(f"  🎯 TOTAL INTEGRATION SCORE: {total_score:.1f}/100")
        
        # Score değerlendirmesi
        if total_score >= 90:
            print(f"  🏆 EXCELLENT - System fully integrated!")
        elif total_score >= 75:
            print(f"  ✅ GOOD - Minor issues, mostly working")
        elif total_score >= 50:
            print(f"  ⚠️ MODERATE - Several integration issues")
        else:
            print(f"  ❌ POOR - Major integration problems")
    
    def generate_final_report(self):
        """9. Final Rapor Oluşturma"""
        print("\n" + "="*60)
        print("📋 FINAL ENTEGRASYON RAPORU")
        print("="*60)
        
        # Özet istatistikler
        total_modules = len(self.test_results['modules'])
        working_modules = sum(1 for m in self.test_results['modules'].values() 
                            if m['status'] == 'SUCCESS')
        
        total_functions = len(self.test_results['functions'])
        working_functions = sum(1 for f in self.test_results['functions'].values() 
                              if f['status'] == 'SUCCESS')
        
        print(f"📦 Modules: {working_modules}/{total_modules} working")
        print(f"⚙️ Classes: {working_functions}/{total_functions} instantiable")
        print(f"🔗 Connections: {len(self.test_results['connections'])} tested")
        print(f"❌ Broken Links: {len(self.test_results['broken_links'])}")
        
        if self.test_results['broken_links']:
            print(f"\n🚫 BROKEN LINKS DETECTED:")
            for link in self.test_results['broken_links']:
                print(f"   - {link}")
        
        if 'active_engines' in self.test_results['performance']:
            print(f"\n🚀 Active Engines: {', '.join(self.test_results['performance']['active_engines'])}")
        
        print(f"\n🎯 Final Integration Score: {self.test_results['integration_score']:.1f}/100")
        
        # JSON rapor kaydet
        try:
            with open('system_integration_report.json', 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved: system_integration_report.json")
        except Exception as e:
            print(f"⚠️ Could not save detailed report: {e}")

def run_integration_test():
    """Ana test çalıştırıcı fonksiyon"""
    tester = SystemIntegrationTest()
    tester.run_complete_test()
    return tester.test_results

if __name__ == "__main__":
    print("🧪 Starting NeoMag V7 System Integration Test...")
    results = run_integration_test()
    print("\n✅ Test completed!") 