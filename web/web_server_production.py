#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeoMag V7 - Production Web Server with TabPFN GPU Integration
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import logging
import sys
import os
import time
import numpy as np
from pathlib import Path
from tabpfn_integration import get_tabpfn_service

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=== NeoMag V7 Production Server Starting ===")

# Flask app setup
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'neomag_v7_production_2024'
CORS(app, origins=['*'])
socketio = SocketIO(app, cors_allowed_origins="*")

# Import modules
try:
    from advanced_bacterium_v7 import AdvancedBacteriumV7
    from molecular_dynamics_engine import MolecularDynamicsEngine
    from population_genetics_engine import PopulationGeneticsEngine  
    from reinforcement_learning_engine import AIDecisionEngine
    print("✅ Core engines loaded")
except ImportError as e:
    print(f"⚠️ Engine import warning: {e}")

# TabPFN GPU Integration - Modular service
print("=== TabPFN Modular Service Loading ===")
tabpfn_service = get_tabpfn_service()
print("✅ TabPFN service module loaded")

class NeoMagV7Simulation:
    """Main simulation class with TabPFN GPU integration"""
    
    def __init__(self):
        self.bacteria = []
        self.food_particles = []
        self.running = False
        self.paused = False
        self.time_step = 0
        self.simulation_data = {"bacteria_count": 0, "avg_fitness": 0.0}
        
        print("✅ NeoMag V7 Simulation initialized")
    
    def start_simulation(self, initial_bacteria=50):
        """Start bacterial simulation with TabPFN prediction"""
        print(f"=== Starting simulation with {initial_bacteria} bacteria ===")
        
        self.bacteria = []
        for i in range(initial_bacteria):
            bacterium = AdvancedBacteriumV7(
                x=np.random.uniform(0, 100),
                y=np.random.uniform(0, 100),
                z=np.random.uniform(0, 10)
            )
            self.bacteria.append(bacterium)
        
        self.running = True
        self.time_step = 0
        
        # TabPFN Prediction Test
        if tabpfn_service.is_initialized:
            self._run_tabpfn_prediction()
        
        # Start simulation loop in background
        import threading
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        return {"status": "simulation_started", "bacteria_count": len(self.bacteria)}
    
    def _run_tabpfn_prediction(self):
        """Run TabPFN GPU prediction on bacterial data using modular service"""
        try:
            print("=== TabPFN Bacterial Prediction Starting ===")
            
            if len(self.bacteria) < 35:
                print(f"⚠️ Not enough bacteria for prediction: {len(self.bacteria)} < 35")
                return None
            
            # Prepare bacterial feature data - gerçek genetics kullan
            X_train = np.array([[
                b.genetics.fitness, b.genetics.mutation_rate, b.genetics.adaptation_speed,
                b.genetics.stress_resistance, b.genetics.metabolic_efficiency,
                b.energy, b.size, b.atp_level, b.age, b.biophysics.mass
            ] for b in self.bacteria[:30]])  # 30 samples
            
            y_train = np.array([b.genetics.fitness for b in self.bacteria[:30]])
            
            X_test = np.array([[
                b.genetics.fitness, b.genetics.mutation_rate, b.genetics.adaptation_speed,
                b.genetics.stress_resistance, b.genetics.metabolic_efficiency,
                b.energy, b.size, b.atp_level, b.age, b.biophysics.mass
            ] for b in self.bacteria[30:35]])  # 5 test
            
            print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
            
            # Use modular TabPFN service
            result = tabpfn_service.predict_bacterial_features(
                X_train=X_train,
                y_train=y_train, 
                X_test=X_test,
                task_type='regression'
            )
            
            if result["status"] == "success":
                print(f"✅ TabPFN prediction completed in {result['prediction_time']:.4f}s")
                print(f"🔥 GPU Acceleration: {result['gpu_acceleration']:.2f}x")
                print(f"🔥 Device: {result['device_used']}")
            else:
                print(f"❌ TabPFN prediction failed: {result.get('message', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ TabPFN prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_simulation_data(self):
        """Get current simulation state"""
        if not self.bacteria:
            return {"bacteria_count": 0, "avg_fitness": 0.0, "time_step": self.time_step}
        
        avg_fitness = np.mean([b.genetics.fitness for b in self.bacteria])
        avg_energy = np.mean([b.energy for b in self.bacteria])
        
        return {
            "bacteria_count": len(self.bacteria),
            "avg_fitness": float(avg_fitness),
            "avg_energy": float(avg_energy), 
            "time_step": self.time_step,
            "running": self.running,
            "paused": self.paused
        }
    
    def _simulation_loop(self):
        """Background simulation loop with Socket.IO updates"""
        print("=== Simulation Loop Started ===")
        
        while self.running and not self.paused:
            try:
                # Update simulation step
                self.time_step += 1
                
                # Update bacteria (realistic simulation)
                survivors = []
                for bacterium in self.bacteria:
                    # Energy consumption
                    bacterium.energy -= np.random.uniform(0.05, 0.2)
                    bacterium.age += 1
                    
                    # Movement
                    bacterium.position[0] += np.random.uniform(-2, 2)
                    bacterium.position[1] += np.random.uniform(-2, 2)
                    
                    # Fitness variation based on energy
                    if bacterium.energy > 5.0:
                        bacterium.genetics.fitness += np.random.uniform(-0.01, 0.02)
                    else:
                        bacterium.genetics.fitness -= np.random.uniform(0.01, 0.03)
                    
                    # Keep fitness in bounds
                    bacterium.genetics.fitness = max(0.1, min(1.0, bacterium.genetics.fitness))
                    
                    # Survival check
                    if bacterium.energy > 0.1:
                        survivors.append(bacterium)
                
                # Update bacteria list
                self.bacteria = survivors
                
                # Reproduction if population is low
                if len(self.bacteria) < 30 and len(self.bacteria) > 5:
                    # Select best bacteria for reproduction
                    best_bacteria = sorted(self.bacteria, key=lambda b: b.genetics.fitness, reverse=True)[:5]
                    
                    for parent in best_bacteria:
                        if np.random.random() < 0.3:  # 30% reproduction chance
                            offspring = AdvancedBacteriumV7(np.random.uniform(0, 100), np.random.uniform(0, 100))
                            # Inherit parent traits with mutation
                            offspring.genetics.fitness = parent.genetics.fitness + np.random.uniform(-0.05, 0.05)
                            offspring.genetics.fitness = max(0.1, min(1.0, offspring.genetics.fitness))
                            offspring.energy = parent.energy * 0.6  # Child gets 60% of parent energy
                            self.bacteria.append(offspring)
                
                # Get simulation data
                sim_data = self.get_simulation_data()
                
                # Emit to frontend via Socket.IO
                socketio.emit('simulation_update', sim_data)
                
                print(f"Step {self.time_step}: {len(self.bacteria)} bacteria, avg_fitness: {sim_data['avg_fitness']:.3f}")
                
                # Sleep for 1 second
                import time
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Simulation loop error: {e}")
                break
        
        print("=== Simulation Loop Ended ===")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        return {"status": "simulation_stopped"}
    
    def pause_resume_simulation(self):
        """Pause/Resume the simulation"""
        self.paused = not self.paused
        return {"status": "paused" if self.paused else "resumed", "paused": self.paused}

# Global simulation instance
simulation = NeoMagV7Simulation()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """Start bacterial simulation with TabPFN GPU acceleration"""
    try:
        print("=== API /start_simulation called ===")
        data = request.get_json() or {}
        initial_bacteria = data.get('initial_bacteria', 50)
        
        print(f"Starting simulation with {initial_bacteria} bacteria")
        result = simulation.start_simulation(initial_bacteria)
        print(f"Simulation started successfully: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ start_simulation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/simulation_data')
def get_simulation_data():
    """Get current simulation data"""
    try:
        print("=== API /simulation_data called ===")
        data = simulation.get_simulation_data()
        return jsonify(data)
    except Exception as e:
        print(f"❌ simulation_data error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop the simulation"""
    try:
        print("=== API /stop_simulation called ===")
        result = simulation.stop_simulation()
        return jsonify(result)
    except Exception as e:
        print(f"❌ stop_simulation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/pause_simulation', methods=['POST'])
def pause_simulation():
    """Pause/Resume the simulation"""
    try:
        print("=== API /pause_simulation called ===")
        result = simulation.pause_resume_simulation()
        return jsonify(result)
    except Exception as e:
        print(f"❌ pause_simulation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/tabpfn_test', methods=['POST'])
def tabpfn_test():
    """Test TabPFN GPU performance"""
    try:
        print("=== API /tabpfn_test called ===")
        
        if not tabpfn_service.is_initialized:
            return jsonify({"status": "error", "message": "TabPFN service not initialized"}), 500
        
        # Use modular service test
        result = tabpfn_service.test_performance(n_samples=100, n_features=10)
        
        if result["status"] == "success":
            return jsonify({
                "status": "success",
                "test_time": result["prediction_time"],
                "gpu_acceleration": result["gpu_acceleration"],
                "device": result["device_used"],
                "message": "TabPFN GPU test completed"
            })
        else:
            return jsonify({"status": "error", "message": result.get("message", "Test failed")}), 500
        
    except Exception as e:
        print(f"❌ tabpfn_test error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/gpu_status')
def gpu_status():
    """Get GPU status and TabPFN availability"""
    try:
        print("=== API /gpu_status called ===")
        status = tabpfn_service.get_status()
        return jsonify(status)
        
    except Exception as e:
        print(f"❌ gpu_status error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Initialize TabPFN service first
    try:
        print("=== TabPFN Service Initialization ===")
        tabpfn_service.initialize()
        print("✅ TabPFN service initialized successfully")
        
        print("=== NeoMag V7 Production Server Ready ===")
        status = tabpfn_service.get_status()
        print(f"🔥 TabPFN Status: {status['status']}")
        print(f"🔥 Device: {status.get('device', 'unknown')}")
        if status.get('gpu_name'):
            print(f"🔥 GPU: {status['gpu_name']}")
        print("🚀 Starting server on http://localhost:5000")
        
        # Configure detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("production_server_debug.log", encoding='utf-8')
            ]
        )
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"❌ TabPFN initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print("Server will continue without TabPFN GPU acceleration")
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    
    except KeyboardInterrupt:
        print("\n=== Server stopped by user ===") 