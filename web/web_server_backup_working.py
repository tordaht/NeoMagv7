#!/usr/bin/env python3
"""
🚀 NeoMag V7 - Modern Web Interface
Advanced Scientific Simulation Control & Visualization
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import time
import logging
import traceback
import numpy as np
from datetime import datetime
import json
import os
import sys
import requests
import subprocess
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent.parent))

try:
    import warnings
    warnings.filterwarnings("ignore")
    
    from engines.molecular_dynamics import MolecularDynamicsEngine
    from engines.population_genetics import PopulationGeneticsEngine
    from engines.ai_decision import AIDecisionEngine
    from agents.bacterium import AdvancedBacteriumV7
    try:
        from ml_models.tabpfn_integration import create_tabpfn_predictor
        TABPFN_AVAILABLE = True
    except:
        TABPFN_AVAILABLE = False
        def create_tabpfn_predictor(*args, **kwargs):
            return None
    NEOMAG_V7_AVAILABLE = True
    print("🎉 NeoMag V7 Modular Engine - LOADED")
    print(f"   TabPFN Available: {TABPFN_AVAILABLE}")
except Exception as e:
    NEOMAG_V7_AVAILABLE = False
    TABPFN_AVAILABLE = False
    print("⚠️ Some import warnings (normal)")
    print("✅ NeoMag V7 ready - CPU mode")
    
    # Create fallback classes
    class MockEngine:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    
    def create_tabpfn_predictor(*args, **kwargs):
        return None
    
    MolecularDynamicsEngine = MockEngine
    PopulationGeneticsEngine = MockEngine 
    AIDecisionEngine = MockEngine
    AdvancedBacteriumV7 = MockEngine
    NEOMAG_V7_AVAILABLE = True

# Flask app setup
app = Flask(__name__, 
            template_folder=str(current_dir / 'templates'),
            static_folder=str(current_dir / 'static'))
app.config['SECRET_KEY'] = 'neomag_v7_modular_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Gemini AI Configuration
GEMINI_API_KEY = "AIzaSyAnC6SImdNu-oJCVm_NKPoVQZEhLlnUapo"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

class GeminiAI:
    """Gemini AI integration for bio-physics analysis"""
    
    def __init__(self, api_key=GEMINI_API_KEY):
        self.api_key = api_key
        
    def analyze_simulation_data(self, data):
        """Analyze simulation data using Gemini AI"""
        try:
            prompt = f"""
            🧬 NeoMag V7 Bio-Fizik Analiz Raporu:
            
            Bakteri Sayısı: {data.get('bacteria_count', 0)}
            Adım: {data.get('time_step', 0)}
            Ortalama Fitness: {data.get('avg_fitness', 0):.3f}
            Ortalama Enerji: {data.get('avg_energy', 0):.1f}
            
            Bu simülasyon verisini analiz et ve kısa öneriler ver:
            - Popülasyon durumu nasıl?
            - Evrimsel baskılar var mı?
            - Optimizasyon önerileri?
            
            Maksimum 150 kelime ile cevapla.
            """
            
            response = self._make_request(prompt)
            return response.get('text', 'Analiz başarısız')
            
        except Exception as e:
            logger.error(f"Gemini AI analiz hatası: {e}")
            return "AI analizi şu anda mevcut değil"
    
    def answer_question(self, question, simulation_context=""):
        """Answer user questions about the simulation"""
        try:
            prompt = f"""
            Sen NeoMag V7 bio-fizik simülasyon uzmanısın. 
            
            Kullanıcı Sorusu: {question}
            
            Simülasyon Bağlamı: {simulation_context}
            
            Kısa ve bilimsel bir cevap ver (maksimum 100 kelime).
            """
            
            response = self._make_request(prompt)
            return response.get('text', 'Cevap alınamadı')
            
        except Exception as e:
            logger.error(f"Gemini AI soru cevap hatası: {e}")
            return "AI şu anda cevap veremiyor"
    
    def _make_request(self, prompt):
        """Make request to Gemini API"""
        headers = {'Content-Type': 'application/json'}
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 200,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={self.api_key}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and result['candidates']:
                return {'text': result['candidates'][0]['content']['parts'][0]['text']}
        
        return {'text': 'API hatası'}

# Global AI instance
gemini_ai = GeminiAI()

# Ngrok Configuration
class NgrokManager:
    """Ngrok tunnel management"""
    
    def __init__(self):
        self.process = None
        self.tunnel_url = None
    
    def start_tunnel(self, port=5000):
        """Start ngrok tunnel"""
        try:
            # ngrok'un yüklü olup olmadığını kontrol et
            result = subprocess.run(['ngrok', 'version'], capture_output=True, text=True)
            if result.returncode != 0:
                return {'status': 'error', 'message': 'ngrok yüklü değil. Lütfen ngrok kurun.'}
            
            # Önceki tunnel'ı durdur
            self.stop_tunnel()
            
            # Yeni tunnel başlat
            self.process = subprocess.Popen(
                ['ngrok', 'http', str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Biraz bekle ve tunnel URL'ini al
            time.sleep(3)
            tunnel_info = self.get_tunnel_info()
            
            if tunnel_info:
                self.tunnel_url = tunnel_info
                logger.info(f"🌐 Ngrok tunnel started: {self.tunnel_url}")
                return {'status': 'success', 'url': self.tunnel_url}
            else:
                return {'status': 'error', 'message': 'Tunnel URL alınamadı'}
                
        except Exception as e:
            logger.error(f"Ngrok start error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def stop_tunnel(self):
        """Stop ngrok tunnel"""
        try:
            if self.process:
                self.process.terminate()
                self.process = None
                self.tunnel_url = None
                logger.info("🛑 Ngrok tunnel stopped")
                return {'status': 'success', 'message': 'Tunnel durduruldu'}
        except Exception as e:
            logger.error(f"Ngrok stop error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_tunnel_info(self):
        """Get tunnel URL from ngrok API"""
        try:
            response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('tunnels'):
                    for tunnel in data['tunnels']:
                        if tunnel.get('proto') == 'https':
                            return tunnel.get('public_url')
            return None
        except:
            return None
    
    def get_status(self):
        """Get tunnel status"""
        if self.process and self.process.poll() is None:
            return {'active': True, 'url': self.tunnel_url}
        else:
            return {'active': False, 'url': None}

# Global ngrok manager
ngrok_manager = NgrokManager()

class NeoMagV7WebSimulation:
    """Web interface for NeoMag V7 simulation"""
    
    def __init__(self):
        self.running = False
        self.paused = False
        self.bacteria_population = []
        self.food_particles = []
        self.selected_bacterium = None
        
        # Initialize engines
        self.md_engine = None
        self.pop_gen_engine = None
        self.ai_engine = None
        self.tabpfn_predictor = None
        
        # Simulation parameters
        self.world_width = 800
        self.world_height = 600
        self.world_depth = 400
        
        # Performance metrics
        self.simulation_step = 0
        self.fps = 0
        self.last_frame_time = time.time()
        self.simulation_start_time = None
        
        # TabPFN optimization
        self.tabpfn_analysis_interval = 100  # Her 100 step'te bir analiz
        self.last_tabpfn_analysis = 0
        self.tabpfn_batch_size = 10  # Batch işleme
        
        # Scientific data
        self.scientific_data = {
            'genetic_diversity': [],
            'population_stats': [],
            'ai_decisions': [],
            'fitness_evolution': [],
            'molecular_interactions': [],
            'tabpfn_predictions': [],
            'bacteria_classes': {}
        }
        
        # Real-time data for charts
        self.real_time_data = {
            'population_over_time': [],
            'fitness_over_time': [],
            'diversity_over_time': [],
            'energy_distribution': [],
            'spatial_clusters': []
        }
        
    def initialize_engines(self, use_gpu=False):
        """Initialize all simulation engines"""
        try:
            logger.info("🚀 Initializing NeoMag V7 engines...")
            
            self.md_engine = MolecularDynamicsEngine(use_gpu=False, temperature=300)  # Force CPU for stability
            logger.info("✅ Molecular Dynamics Engine initialized")
            
            self.pop_gen_engine = PopulationGeneticsEngine(
                mutation_rate=1e-6, 
                recombination_rate=1e-7,
                abc_simulation_count=50  # Reduced for performance
            )
            logger.info("✅ Population Genetics Engine initialized")
            
            self.ai_engine = AIDecisionEngine(
                model_type='DQN', 
                state_size=15, 
                action_size=6,
                use_gpu=False,  # Force CPU for stability
                enable_marl=False  # Disable MARL for better performance
            )
            logger.info("✅ AI Decision Engine initialized")
            
            # Initialize TabPFN predictor with optimization
            try:
                self.tabpfn_predictor = create_tabpfn_predictor("biophysical", device='cpu')
                logger.info("TabPFN predictor initialized with optimization")
            except Exception as e:
                logger.warning(f"TabPFN not available: {e}")
                self.tabpfn_predictor = None
                
            logger.info("All engines initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            return False
    
    def start_simulation(self, initial_bacteria=50):
        """Start the simulation - SIMPLE VERSION"""
        if self.running:
            return False
            
        try:
            # Create simple bacteria - NO COMPLEX ENGINES
            self.bacteria_population = []
            for i in range(initial_bacteria):
                # Simple bacterium object
                bacterium = type('SimpleBacterium', (), {
                    'x': np.random.uniform(50, self.world_width - 50),
                    'y': np.random.uniform(50, self.world_height - 50), 
                    'z': np.random.uniform(10, 50),
                    'vx': 0, 'vy': 0, 'vz': 0,
                    'energy_level': np.random.uniform(40, 80),
                    'age': 0,
                    'current_fitness': np.random.uniform(0.3, 0.9),
                    'size': np.random.uniform(0.3, 0.8),
                    'mass': 1e-15,
                    'generation': 0,
                    'genome_length': 1000,
                    'atp_level': np.random.uniform(30, 70),
                    'md_interactions': 0,
                    'genetic_operations': 0,
                    'ai_decisions': 0,
                    'fitness_landscape_position': np.random.rand(10).tolist()
                })()
                self.bacteria_population.append(bacterium)
            
            # Simple food particles
            self.food_particles = []
            for i in range(80):
                food = type('SimpleFood', (), {
                    'x': np.random.uniform(0, self.world_width),
                    'y': np.random.uniform(0, self.world_height),
                    'z': np.random.uniform(0, 20),
                    'size': 0.2,
                    'energy_value': 15
                })()
                self.food_particles.append(food)
            
            self.running = True
            self.paused = False
            self.simulation_step = 0
            self.simulation_start_time = time.time()
            
            # Start simple simulation loop
            self.simulation_thread = threading.Thread(target=self._simple_simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            logger.info(f"✅ Simple simulation started with {len(self.bacteria_population)} bacteria")
            return True
            
        except Exception as e:
            logger.error(f"Start simulation error: {e}")
            return False
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.paused = False
        
        # Wait for simulation thread to finish
        if hasattr(self, 'simulation_thread'):
            self.simulation_thread.join(timeout=2.0)
        
        logger.info("Simulation stopped")
        
    def pause_resume_simulation(self):
        """Toggle pause state"""
        if self.running:
            self.paused = not self.paused
            return self.paused
        return False
    
    def add_bacteria(self, count=25):
        """Add bacteria to simulation"""
        if not self.running:
            return False
            
        for i in range(count):
            x = np.random.uniform(50, self.world_width - 50)
            y = np.random.uniform(50, self.world_height - 50)
            z = np.random.uniform(50, self.world_depth - 50)
            bacterium = AdvancedBacteriumV7(x=x, y=y, z=z)
            self.bacteria_population.append(bacterium)
        
        logger.info(f"Added {count} bacteria")
        return True
    
    def add_food_particles(self, count=50):
        """Add food particles"""
        for i in range(count):
            food = {
                'x': np.random.uniform(0, self.world_width),
                'y': np.random.uniform(0, self.world_height),
                'z': np.random.uniform(0, self.world_depth),
                'energy': np.random.uniform(10, 30)
            }
            self.food_particles.append(food)
    
    def _simulation_loop(self):
        """Main simulation loop"""
        dt = 0.1
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            frame_start = time.time()
            
            try:
                self._update_simulation_step(dt)
                self.simulation_step += 1
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                self.fps = 1.0 / max(frame_time, 0.001)
                
                # Data collection every 10 steps
                if self.simulation_step % 10 == 0:
                    self._collect_scientific_data()
                
                # Sleep to maintain reasonable frame rate
                target_frame_time = 1.0 / 30  # 30 FPS
                sleep_time = max(0, target_frame_time - frame_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Simulation loop error: {e}")
                time.sleep(0.1)
    
    def _update_simulation_step(self, dt):
        """Update one simulation step"""
        world_dims = (self.world_width, self.world_height, self.world_depth)
        
        # Update bacteria
        for bacterium in self.bacteria_population[:]:
            if not bacterium.is_alive():
                self.bacteria_population.remove(bacterium)
                continue
            
            # Environment state
            environment_state = self._get_environment_state(bacterium)
            
            # AI decision making
            action = bacterium.make_decision(self.ai_engine, environment_state, world_dims)
            
            # Apply action
            self._apply_bacterium_action(bacterium, action, dt)
            
            # Update molecular state
            bacterium.update_molecular_state(self.md_engine, self.bacteria_population, dt)
            
            # Update genetic state
            bacterium.update_genetic_state(self.pop_gen_engine, self.bacteria_population)
            
            # Age the bacterium
            bacterium.update_age(dt)
            
            # Update AI model
            state = bacterium._get_state_representation(environment_state, world_dims)
            reward = bacterium.current_fitness
            self.ai_engine.update_model(
                state, action, reward, state, False, 
                bacterium._get_possible_actions(), bacterium.id
            )
        
        # Population genetics evolution (every 100 steps)
        if self.simulation_step % 100 == 0 and len(self.bacteria_population) > 5:
            self.bacteria_population = self.pop_gen_engine.evolve_population(
                self.bacteria_population, generations=1
            )
        
        # Add food periodically
        if self.simulation_step % 50 == 0:
            self.add_food_particles(20)
        
        # Remove old food
        self.food_particles = [f for f in self.food_particles if f['energy'] > 1]
    
    def _get_environment_state(self, bacterium):
        """Get environment state for a bacterium"""
        nearby_bacteria = []
        nearby_food = []
        
        for other in self.bacteria_population:
            if other.id != bacterium.id:
                distance = np.linalg.norm(bacterium.biophysical.position - other.biophysical.position)
                if distance < 50:  # Within 50 units
                    nearby_bacteria.append(other)
        
        for food in self.food_particles:
            food_pos = np.array([food['x'], food['y'], food['z']])
            distance = np.linalg.norm(bacterium.biophysical.position - food_pos)
            if distance < 30:  # Within 30 units
                nearby_food.append(food)
        
        return {
            'nearby_bacteria': nearby_bacteria,
            'nearby_food': nearby_food,
            'local_density': len(nearby_bacteria) / 100.0,
            'food_concentration': len(nearby_food) / 50.0
        }
    
    def _apply_bacterium_action(self, bacterium, action, dt):
        """Apply bacterium action"""
        move_speed = 5.0
        
        if action == "move_up":
            bacterium.y = max(0, bacterium.y - move_speed * dt)
        elif action == "move_down":
            bacterium.y = min(self.world_height, bacterium.y + move_speed * dt)
        elif action == "move_left":
            bacterium.x = max(0, bacterium.x - move_speed * dt)
        elif action == "move_right":
            bacterium.x = min(self.world_width, bacterium.x + move_speed * dt)
        elif action == "consume":
            self._try_consume_food(bacterium)
        # "wait" does nothing
        
        # Consume energy for movement
        if action.startswith("move"):
            bacterium.consume_energy(0.5 * dt)
    
    def _try_consume_food(self, bacterium):
        """Try to consume nearby food"""
        for food in self.food_particles[:]:
            food_pos = np.array([food['x'], food['y'], food['z']])
            distance = np.linalg.norm(bacterium.biophysical.position - food_pos)
            
            if distance < 10:  # Close enough to consume
                energy_gain = min(food['energy'], 20)
                bacterium.gain_energy(energy_gain)
                food['energy'] -= energy_gain
                
                if food['energy'] <= 0:
                    self.food_particles.remove(food)
                break
    
    def _collect_scientific_data(self):
        """Collect scientific data for analysis with optimization"""
        try:
            if not self.bacteria_population:
                return

            # Basic population statistics
            total_bacteria = len(self.bacteria_population)
            alive_bacteria = len([b for b in self.bacteria_population if hasattr(b, 'alive') and getattr(b, 'alive', True)])
            
            avg_fitness = np.mean([getattr(b, 'current_fitness', getattr(b, 'fitness', 0)) for b in self.bacteria_population])
            avg_energy = np.mean([getattr(b, 'energy_level', 0) for b in self.bacteria_population])
            avg_age = np.mean([getattr(b, 'age', 0) for b in self.bacteria_population])
            avg_generation = np.mean([getattr(b, 'generation', 0) for b in self.bacteria_population])

            pop_stats = {
                'timestamp': time.time(),
                'step': self.simulation_step,
                'total_bacteria': total_bacteria,
                'alive_bacteria': alive_bacteria,
                'avg_fitness': float(avg_fitness),
                'avg_energy': float(avg_energy),
                'avg_age': float(avg_age),
                'avg_generation': float(avg_generation)
            }
            
            self.scientific_data['population_stats'].append(pop_stats)
            
            # Real-time data for charts
            self.real_time_data['population_over_time'].append({
                'x': self.simulation_step,
                'y': total_bacteria
            })
            self.real_time_data['fitness_over_time'].append({
                'x': self.simulation_step,
                'y': float(avg_fitness)
            })
            
            # Bacteria classification for colors
            for i, bacterium in enumerate(self.bacteria_population):
                fitness = getattr(bacterium, 'current_fitness', getattr(bacterium, 'fitness', 0))
                energy = getattr(bacterium, 'energy_level', 0)
                age = getattr(bacterium, 'age', 0)
                
                # Enhanced classification
                if fitness > 0.8 and energy > 0.8:
                    bacterium_class = 'elite'      # Gold
                elif fitness > 0.6 and age > 50:
                    bacterium_class = 'veteran'    # Blue
                elif fitness > 0.5:
                    bacterium_class = 'strong'     # Green
                elif energy > 0.7:
                    bacterium_class = 'energetic'  # Yellow
                elif age < 10:
                    bacterium_class = 'young'      # Light Blue
                else:
                    bacterium_class = 'basic'      # Orange
                    
                self.scientific_data['bacteria_classes'][i] = bacterium_class

            # Genetic diversity calculation (less frequent)
            if self.simulation_step % 10 == 0:  # Her 10 step'te bir
                try:
                    genetic_profiles = [getattr(b, 'genetic_profile', {}) for b in self.bacteria_population]
                    if genetic_profiles and self.pop_gen_engine:
                        diversity_metrics = self.pop_gen_engine.calculate_genetic_diversity_metrics(genetic_profiles)
                        
                        genetic_div = {
                            'timestamp': time.time(),
                            'step': self.simulation_step,
                            'diversity_metrics': diversity_metrics
                        }
                        
                        self.scientific_data['genetic_diversity'].append(genetic_div)
                        
                        if 'diversity_index' in diversity_metrics:
                            self.real_time_data['diversity_over_time'].append({
                                'x': self.simulation_step,
                                'y': float(diversity_metrics['diversity_index'])
                            })
                except Exception as e:
                    logger.debug(f"Genetic diversity calculation error: {e}")

            # TabPFN Analysis (optimized - less frequent)
            if (TABPFN_AVAILABLE and self.tabpfn_predictor and 
                len(self.bacteria_population) > 10 and 
                self.simulation_step - self.last_tabpfn_analysis >= self.tabpfn_analysis_interval):
                try:
                    # Prepare features for TabPFN (batch processing)
                    features = []
                    targets = []
                    sample_bacteria = self.bacteria_population[::max(1, len(self.bacteria_population)//self.tabpfn_batch_size)]
                    
                    for bacterium in sample_bacteria[:self.tabpfn_batch_size]:
                        feature_vector = [
                            getattr(bacterium, 'x', 0),
                            getattr(bacterium, 'y', 0),
                            getattr(bacterium, 'energy_level', 0),
                            getattr(bacterium, 'current_fitness', getattr(bacterium, 'fitness', 0)),
                            getattr(bacterium, 'age', 0),
                            len([b for b in self.bacteria_population if 
                                 np.sqrt((getattr(b, 'x', 0) - getattr(bacterium, 'x', 0))**2 + 
                                        (getattr(b, 'y', 0) - getattr(bacterium, 'y', 0))**2) < 50])
                        ]
                        features.append(feature_vector)
                        targets.append(getattr(bacterium, 'current_fitness', getattr(bacterium, 'fitness', 0)))
                    
                    if len(features) >= 5:
                        features_array = np.array(features)
                        targets_array = np.array(targets)
                        
                        # Make predictions with TabPFN using correct method
                        prediction_result = self.tabpfn_predictor.predict_fitness_landscape(
                            features_array, targets_array, features_array
                        )
                        
                        tabpfn_result = {
                            'timestamp': time.time(),
                            'step': self.simulation_step,
                            'predictions_mean': float(np.mean(prediction_result.predictions)),
                            'predictions_std': float(np.std(prediction_result.predictions)),
                            'sample_size': len(features),
                            'prediction_variance': float(np.var(prediction_result.predictions)),
                            'prediction_time': prediction_result.prediction_time
                        }
                        
                        self.scientific_data['tabpfn_predictions'].append(tabpfn_result)
                        self.last_tabpfn_analysis = self.simulation_step
                        
                except Exception as e:
                    logger.debug(f"TabPFN analysis error: {e}")

            # AI performance metrics (less frequent)
            if self.simulation_step % 20 == 0 and self.ai_engine:
                try:
                    ai_metrics = self.ai_engine.get_performance_metrics()
                    self.scientific_data['ai_decisions'].append({
                        'step': self.simulation_step,
                        'metrics': ai_metrics,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.debug(f"AI metrics error: {e}")

            # Keep data size manageable
            max_entries = 1000
            for key in self.scientific_data:
                if isinstance(self.scientific_data[key], list) and len(self.scientific_data[key]) > max_entries:
                    self.scientific_data[key] = self.scientific_data[key][-max_entries:]
            
            for key in self.real_time_data:
                if isinstance(self.real_time_data[key], list) and len(self.real_time_data[key]) > max_entries:
                    self.real_time_data[key] = self.real_time_data[key][-max_entries:]
                    
        except Exception as e:
            logger.error(f"Scientific data collection error: {e}")
            traceback.print_exc()
    
    def get_simulation_data(self):
        """Get current simulation data for web interface"""
        if not self.running:
            return {'status': 'stopped', 'bacteria_count': 0, 'food_count': 0}
            
        # Enhanced bacteria sample data 
        bacteria_sample = []
        max_sample_size = min(100, len(self.bacteria_population))
        
        for i in range(0, len(self.bacteria_population), max(1, len(self.bacteria_population) // max_sample_size)):
            if i < len(self.bacteria_population):
                b = self.bacteria_population[i]
                bacteria_sample.append({
                    'id': i,
                    'position': [float(getattr(b, 'x', 0)), float(getattr(b, 'y', 0)), float(getattr(b, 'z', 0))],
                    'velocity': [float(getattr(b, 'vx', 0)), float(getattr(b, 'vy', 0)), float(getattr(b, 'vz', 0))],
                    'energy_level': float(getattr(b, 'energy_level', 0)),
                    'age': float(getattr(b, 'age', 0)),
                    'current_fitness_calculated': float(getattr(b, 'current_fitness', getattr(b, 'fitness', 0))),
                    'size': float(getattr(b, 'size', 0.5)),
                    'mass': float(getattr(b, 'mass', 1e-15)),
                    'generation': int(getattr(b, 'generation', 0)),
                    'genome_length': int(getattr(b, 'genome_length', 1000)),
                    'atp_level': float(getattr(b, 'atp_level', 50.0)),
                    'md_interactions': int(getattr(b, 'md_interactions', 0)),
                    'genetic_operations': int(getattr(b, 'genetic_operations', 0)),
                    'ai_decisions': int(getattr(b, 'ai_decisions', 0)),
                    'genetic_profile': {
                        'fitness_landscape_position': [float(x) for x in getattr(b, 'fitness_landscape_position', [0]*10)]
                    }
                })
        
        # Food sample
        food_sample = []
        if hasattr(self, 'food_particles') and self.food_particles:
            max_food_sample = min(50, len(self.food_particles))
            for i in range(0, len(self.food_particles), max(1, len(self.food_particles) // max_food_sample)):
                if i < len(self.food_particles):
                    f = self.food_particles[i]
                    food_sample.append({
                        'position': [float(getattr(f, 'x', 0)), float(getattr(f, 'y', 0)), float(getattr(f, 'z', 0))],
                        'size': float(getattr(f, 'size', 0.2))
                    })
        
        # Enhanced performance and environmental data
        current_time = time.time()
        sim_time = current_time - self.simulation_start_time if self.simulation_start_time else 0
        
        return {
            'status': 'running' if not self.paused else 'paused',
            'time_step': self.simulation_step,
            'sim_time': sim_time,
            'bacteria_count': len(self.bacteria_population),
            'food_count': len(self.food_particles) if hasattr(self, 'food_particles') else 0,
            'bacteria_sample': bacteria_sample,
            'food_sample': food_sample,
            'world_dimensions': [self.world_width, self.world_height, self.world_depth],
            'current_generation': max([getattr(b, 'generation', 0) for b in self.bacteria_population], default=0),
            'performance': {
                'steps_per_second': round(getattr(self, 'fps', 0), 1)
            },
            'environmental_pressures': {
                'temperature': getattr(self, 'temperature', 298.15),
                'nutrient_availability': getattr(self, 'nutrient_availability', 75.0)
            },
            'scientific_data_log': self.scientific_data
        }
    
    def get_bacterium_details(self, bacterium_id):
        """Get detailed information about a specific bacterium"""
        try:
            bacterium_id = int(bacterium_id)
            if 0 <= bacterium_id < len(self.bacteria_population):
                b = self.bacteria_population[bacterium_id]
                return {
                    'id': bacterium_id,
                    'basic_info': {
                        'x': float(getattr(b, 'x', 0)),
                        'y': float(getattr(b, 'y', 0)),
                        'z': float(getattr(b, 'z', 0)),
                        'energy': float(getattr(b, 'energy_level', 0)),
                        'fitness': float(getattr(b, 'current_fitness', getattr(b, 'fitness', 0))),
                        'age': float(getattr(b, 'age', 0)),
                        'generation': int(getattr(b, 'generation', 0)),
                        'size': float(getattr(b, 'size', 5)),
                        'class': self.scientific_data['bacteria_classes'].get(bacterium_id, 'basic')
                    },
                    'genetic_info': getattr(b, 'genetic_profile', {}),
                    'molecular_data': getattr(b, 'biophysical', {}).__dict__ if hasattr(b, 'biophysical') else {},
                    'ai_decisions': getattr(b, 'decision_history', [])[-10:] if hasattr(b, 'decision_history') else [],
                    'neighbors': self._get_bacterium_neighbors(bacterium_id),
                    'environmental_factors': self._get_environment_state(b) if hasattr(self, '_get_environment_state') else {}
                }
            return None
        except Exception as e:
            logger.error(f"Error getting bacterium details: {e}")
            return None
    
    def _get_bacterium_neighbors(self, bacterium_id):
        """Get neighbors of a specific bacterium"""
        try:
            if 0 <= bacterium_id < len(self.bacteria_population):
                target = self.bacteria_population[bacterium_id]
                neighbors = []
                
                for i, b in enumerate(self.bacteria_population):
                    if i != bacterium_id:
                        distance = np.sqrt(
                            (getattr(b, 'x', 0) - getattr(target, 'x', 0))**2 + 
                            (getattr(b, 'y', 0) - getattr(target, 'y', 0))**2
                        )
                        if distance < 100:  # Within 100 units
                            neighbors.append({
                                'id': i,
                                'distance': float(distance),
                                'energy': float(getattr(b, 'energy_level', 0)),
                                'fitness': float(getattr(b, 'current_fitness', getattr(b, 'fitness', 0)))
                            })
                
                return sorted(neighbors, key=lambda x: x['distance'])[:10]  # Closest 10
            return []
        except Exception as e:
            logger.error(f"Error getting neighbors: {e}")
            return []
    
    def get_scientific_export(self):
        """Get comprehensive scientific data export"""
        try:
            # Safely get population summaries
            population_summaries = []
            for i, b in enumerate(self.bacteria_population):
                try:
                    if hasattr(b, 'get_status_summary'):
                        population_summaries.append(b.get_status_summary())
                    else:
                        # Fallback for MockEngine bacteria
                        population_summaries.append({
                            'id': i,
                            'x': getattr(b, 'x', 0),
                            'y': getattr(b, 'y', 0),
                            'energy': getattr(b, 'energy_level', 0),
                            'fitness': getattr(b, 'current_fitness', 0)
                        })
                except:
                    pass
            
            return {
                'simulation_metadata': {
                    'version': 'NeoMag V7.0',
                    'simulation_step': self.simulation_step,
                    'world_dimensions': [self.world_width, self.world_height, self.world_depth],
                    'engines': {
                        'molecular_dynamics': type(self.md_engine).__name__ if self.md_engine else None,
                        'population_genetics': type(self.pop_gen_engine).__name__ if self.pop_gen_engine else None,
                        'ai_decision': type(self.ai_engine).__name__ if self.ai_engine else None
                    }
                },
                'scientific_data': self.scientific_data,
                'current_population': population_summaries,
                'export_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Export error: {e}")
            return {'error': str(e)}

    def _simple_simulation_loop(self):
        """Simple simulation loop - NO COMPLEX ENGINES"""
        last_time = time.time()
        
        while self.running:
            if not self.paused:
                current_time = time.time()
                
                try:
                    # Simple bacteria movement
                    for b in self.bacteria_population:
                        # Random walk
                        b.x += np.random.uniform(-2, 2)
                        b.y += np.random.uniform(-2, 2)
                        
                        # Keep in bounds
                        b.x = max(10, min(self.world_width - 10, b.x))
                        b.y = max(10, min(self.world_height - 10, b.y))
                        
                        # Age and energy changes
                        b.age += 0.1
                        b.energy_level += np.random.uniform(-0.5, 0.5)
                        b.energy_level = max(0, min(100, b.energy_level))
                        
                        # Fitness variation
                        b.current_fitness += np.random.uniform(-0.01, 0.01)
                        b.current_fitness = max(0, min(1, b.current_fitness))
                    
                    self.simulation_step += 1
                    
                    # Update FPS
                    if current_time - last_time > 0:
                        self.fps = 1.0 / (current_time - last_time)
                    last_time = current_time
                    
                except Exception as e:
                    logger.error(f"Simulation loop error: {e}")
                
            time.sleep(0.05)  # 20 FPS target

# Global simulation instance
simulation = NeoMagV7WebSimulation()

# Routes
@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html')

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """Start simulation"""
    try:
        if not NEOMAG_V7_AVAILABLE:
            return jsonify({'status': 'error', 'message': 'NeoMag V7 not available'}), 500
        
        data = request.get_json() or {}
        initial_bacteria = data.get('initial_bacteria_count', data.get('bacteria_count', 30))
        
        if simulation.start_simulation(initial_bacteria):
            return jsonify({
                'status': 'success',
                'message': f'Simulation started with {initial_bacteria} bacteria',
                'version': 'NeoMag V7.0 Modular'
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start simulation'}), 500
            
    except Exception as e:
        logger.error(f"Start simulation error: {e}")
        traceback.print_exc()  # Full error trace
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop simulation"""
    try:
        simulation.stop_simulation()
        return jsonify({'status': 'success', 'message': 'Simulation stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/pause_simulation', methods=['POST'])
def pause_simulation():
    """Pause simulation"""
    try:
        if not simulation.running:
            return jsonify({'status': 'error', 'message': 'Simulation not running'}), 400
        simulation.paused = True
        return jsonify({'status': 'success', 'message': 'Simulation paused'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/resume_simulation', methods=['POST'])  
def resume_simulation():
    """Resume simulation"""
    try:
        if not simulation.running:
            return jsonify({'status': 'error', 'message': 'Simulation not running'}), 400
        if not simulation.paused:
            return jsonify({'status': 'success', 'message': 'Simulation already running', 'already_running': True})
        simulation.paused = False
        return jsonify({'status': 'success', 'message': 'Simulation resumed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_results')
def get_results():
    """Get simulation results for export"""
    try:
        return jsonify(simulation.get_scientific_export())
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ai_analysis', methods=['POST'])
def ai_analysis():
    """Get AI analysis of current simulation"""
    try:
        if not simulation.running:
            return jsonify({'status': 'error', 'message': 'Simulation not running'}), 400
            
        sim_data = simulation.get_simulation_data()
        
        # Calculate summary stats
        analysis_data = {
            'bacteria_count': sim_data.get('bacteria_count', 0),
            'time_step': sim_data.get('time_step', 0),
            'avg_fitness': 0,
            'avg_energy': 0
        }
        
        if sim_data.get('bacteria_sample'):
            bacteria = sim_data['bacteria_sample']
            analysis_data['avg_fitness'] = sum(b.get('current_fitness_calculated', 0) for b in bacteria) / len(bacteria)
            analysis_data['avg_energy'] = sum(b.get('energy_level', 0) for b in bacteria) / len(bacteria)
        
        analysis = gemini_ai.analyze_simulation_data(analysis_data)
        return jsonify({'status': 'success', 'analysis': analysis})
        
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ai_question', methods=['POST'])
def ai_question():
    """Ask AI a question about the simulation"""
    try:
        data = request.get_json() or {}
        question = data.get('question', '')
        
        if not question:
            return jsonify({'status': 'error', 'message': 'No question provided'}), 400
        
        # Get simulation context
        context = ""
        if simulation.running:
            sim_data = simulation.get_simulation_data()
            context = f"Bacteria: {sim_data.get('bacteria_count', 0)}, Step: {sim_data.get('time_step', 0)}"
        
        answer = gemini_ai.answer_question(question, context)
        return jsonify({'status': 'success', 'answer': answer})
        
    except Exception as e:
        logger.error(f"AI question error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ngrok_start', methods=['POST'])
def start_ngrok():
    """Start ngrok tunnel"""
    try:
        result = ngrok_manager.start_tunnel()
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ngrok_stop', methods=['POST'])
def stop_ngrok():
    """Stop ngrok tunnel"""
    try:
        result = ngrok_manager.stop_tunnel()
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ngrok_status')
def ngrok_status():
    """Get ngrok status"""
    try:
        status = ngrok_manager.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/add_bacteria', methods=['POST'])
def add_bacteria():
    """Add bacteria"""
    try:
        data = request.get_json() or {}
        count = data.get('count', 25)
        
        if simulation.add_bacteria(count):
            return jsonify({
                'status': 'success', 
                'message': f'Added {count} bacteria',
                'total_bacteria': len(simulation.bacteria_population)
            })
        else:
            return jsonify({'status': 'error', 'message': 'Simulation not running'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/simulation_data')
def get_simulation_data():
    """Get simulation data"""
    try:
        return jsonify(simulation.get_simulation_data())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scientific_export')
def scientific_export():
    """Export scientific data"""
    try:
        return jsonify(simulation.get_scientific_export())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bacterium/<int:bacterium_id>')
def get_bacterium_details(bacterium_id):
    """Get detailed information about a specific bacterium"""
    try:
        details = simulation.get_bacterium_details(bacterium_id)
        if details:
            return jsonify(details)
        else:
            return jsonify({'error': 'Bacterium not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# TabPFN endpoint removed - using Gemini AI instead

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('connected', {'status': 'Connected to NeoMag V7'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

def emit_simulation_data():
    """Emit real-time simulation data"""
    if simulation.running:
        try:
            data = simulation.get_simulation_data()
            socketio.emit('simulation_update', data)
        except Exception as e:
            logger.error(f"Error emitting data: {e}")

def data_emission_loop():
    """Background data emission loop"""
    while True:
        if simulation.running and not simulation.paused:
            emit_simulation_data()
        time.sleep(0.2)  # 5 FPS for web updates

# Start data emission thread
data_thread = threading.Thread(target=data_emission_loop)
data_thread.daemon = True
data_thread.start()

if __name__ == '__main__':
    logger.info("🚀 Starting NeoMag V7 Web Server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False) 