#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeoMag V7 - Advanced Server with Modern Features
Enhanced with Security, Environment Variables, Background Threads
"""

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import time
import threading
import random
import math
import logging
import io
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# Environment Variables Support
def load_env_variables():
    """Load environment variables from config.env file"""
    env_path = Path(__file__).parent / 'config.env'
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                    os.environ[key.strip()] = value.strip()
    return env_vars

# Load environment variables
ENV_VARS = load_env_variables()

# Professional Logging Setup
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.environ.get('LOG_FILE', 'logs/neomag_server.log')

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Flask App with Security
app = Flask(__name__)
# IMPORTANT: For production, ensure SECRET_KEY in config.env is a strong, cryptographically random string.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback_secret_key')

# CORS Configuration - ALLOW BOTH 5000 AND 5001 PORTS
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000').split(',')
logger.info(f"CORS origins configured: {CORS_ORIGINS}") # Log configured origins
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}}) # Use origins from env

# SocketIO for Real-time Communication - ENHANCED DEBUG LOGGING
socketio = SocketIO(app, 
                   cors_allowed_origins=CORS_ORIGINS, # Use origins from env
                   async_mode='threading', 
                   ping_timeout=60, 
                   ping_interval=25,
                   logger=True,        # SOCKET.IO LOGGER AKTIF
                   engineio_logger=True)  # ENGINE.IO LOGGER AKTIF

# Security Headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

logger.info("🚀 NeoMag V7 Advanced Server başlatılıyor...")

# --- Thread Safety Implementation ---
import threading

# Global thread locks for critical sections
simulation_lock = threading.RLock()  # Re-entrant lock for nested calls
data_lock = threading.RLock()  # For data read/write operations

# Global simulation state with thread safety
simulation_state = {
    "running": False,
    "paused": False,
    "bacteria": [],
    "food": [],
    "generation": 1,
    "step": 0,
    "total_bacteria": 0,
    "avg_fitness": 0.0,
    "avg_energy": 0.0,
    "mutations": 0,
    "deaths": 0,
    "births": 0,
    "last_update": time.time(),
    "fps": 0,
    "history": {
        "population": [],
        "fitness": [],
        "energy": [],
        "mutations": [],
        "timestamp": []
    }
}

# Thread-safe getter/setter functions
def get_simulation_state(key=None):
    """Thread-safe getter for simulation state"""
    with data_lock:
        if key:
            return simulation_state.get(key)
        return simulation_state.copy()  # Return copy to prevent external modifications

def set_simulation_state(key, value):
    """Thread-safe setter for simulation state"""
    with data_lock:
        simulation_state[key] = value

def update_simulation_state(updates):
    """Thread-safe bulk update for simulation state"""
    with data_lock:
        simulation_state.update(updates)

# Simulation parameters from environment
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
MAX_BACTERIA = int(os.environ.get('MAX_BACTERIA_COUNT', 100))
FOOD_COUNT = 25
MUTATION_RATE = 0.1
ENERGY_DECAY = 0.3
SIMULATION_FPS = int(os.environ.get('SIMULATION_FPS', 25))

# --- Gemini AI Service ---
import requests

class GeminiAI:
    """Gemini AI integration for simulation analysis"""
    def __init__(self):
        self.api_key = os.environ.get('GEMINI_API_KEY', '')
        self.model = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        if not self.api_key or self.api_key == 'your_actual_gemini_api_key_here':
            logger.warning("⚠️ Gemini API anahtarı ayarlanmamış. AI özellikleri çalışmayacak.")

    def _make_request(self, prompt_text, max_tokens=200, temperature=0.7):
        """Make API request to Gemini"""
        if not self.api_key:
            return {'text': 'Gemini API anahtarı yapılandırılmamış.', 'error': True}

        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}
        }
        
        try:
            response = requests.post(f"{self.api_url}?key={self.api_key}", 
                                   headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            if result.get('candidates') and result['candidates'][0].get('content', {}).get('parts'):
                return {'text': result['candidates'][0]['content']['parts'][0]['text'], 'error': False}
            else:
                logger.error(f"Gemini API'den beklenmedik yanıt: {result}")
                return {'text': 'AI modelinden geçerli bir yanıt alınamadı.', 'error': True}
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API isteği başarısız: {e}")
            return {'text': f"AI servisine ulaşılamadı: {e}", 'error': True}
        except Exception as e:
            logger.error(f"Gemini yanıtı işlenirken hata: {e}")
            return {'text': f"AI yanıtı işlenirken hata: {e}", 'error': True}

    def analyze_population(self, bacteria_data):
        """Analyze bacterial population"""
        total_count = len(bacteria_data)
        avg_fitness = sum(b.get('fitness', 0) for b in bacteria_data) / max(total_count, 1)
        avg_energy = sum(b.get('energy', 0) for b in bacteria_data) / max(total_count, 1)
        
        # Classification distribution
        classifications = {}
        for bacterium in bacteria_data:
            cls = bacterium.get('classification', 'basic')
            classifications[cls] = classifications.get(cls, 0) + 1
        
        prompt = f"""
        NeoMag V7 bakteriyel simülasyon analisti olarak aşağıdaki popülasyon verilerini analiz edin:
        
        Popülasyon Özeti:
        - Toplam Bakteri: {total_count}
        - Ortalama Fitness: {avg_fitness:.3f}
        - Ortalama Enerji: {avg_energy:.1f}
        - Sınıf Dağılımı: {classifications}
        
        Lütfen kısa (max 120 kelime), bilimsel ve anlaşılır bir analiz sunun. 
        Popülasyonun sağlığını, evrimi ve optimizasyon önerilerini içerin.
        """
        
        return self._make_request(prompt).get('text', "AI analizi şu anda yapılamıyor.")

# Initialize AI service
gemini_ai = GeminiAI()

class AdvancedBacterium:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.energy = random.uniform(60, 120)
        self.size = random.uniform(8, 20)
        self.speed = random.uniform(0.5, 4.0)
        self.direction = random.uniform(0, 2 * math.pi)
        self.age = 0
        self.generation = 1 if parent is None else parent.generation + 1
        
        # Advanced properties - SET INTELLIGENCE FIRST!
        self.intelligence = random.uniform(0.1, 1.0)
        self.aggression = random.uniform(0.0, 0.8)
        self.reproduction_cooldown = 0
        self.vision_range = random.uniform(20, 80)
        self.fitness = self.calculate_initial_fitness()  # MOVED AFTER intelligence
        
        # Genetics
        self.dna = self.generate_dna(parent)
        self.classification = self.get_classification()
        
        # Stats tracking
        self.food_eaten = 0
        self.distance_traveled = 0
        self.offspring_count = 0
        
    def generate_dna(self, parent):
        if parent is None:
            return {
                'speed_gene': random.uniform(0.0, 1.0),
                'energy_gene': random.uniform(0.0, 1.0),
                'size_gene': random.uniform(0.0, 1.0),
                'intelligence_gene': random.uniform(0.0, 1.0),
                'vision_gene': random.uniform(0.0, 1.0)
            }
        else:
            # Inherit from parent with mutations
            dna = {}
            for gene, value in parent.dna.items():
                if random.random() < MUTATION_RATE:
                    # Mutation
                    mutation = random.uniform(-0.2, 0.2)
                    dna[gene] = max(0.0, min(1.0, value + mutation))
                else:
                    dna[gene] = value
            return dna
    
    def calculate_initial_fitness(self):
        # Base fitness calculation
        base_fitness = (self.energy / 120) * 0.4 + (self.speed / 4.0) * 0.3 + (self.intelligence) * 0.3
        return max(0.1, min(1.0, base_fitness))
    
    def update_fitness(self):
        # Dynamic fitness based on performance
        age_factor = min(1.0, self.age / 1000)  # Experience bonus
        food_factor = min(1.0, self.food_eaten / 10)  # Survival bonus
        offspring_factor = min(1.0, self.offspring_count / 3)  # Reproduction bonus
        
        self.fitness = (
            (self.energy / 120) * 0.3 +
            food_factor * 0.25 +
            age_factor * 0.2 +
            offspring_factor * 0.15 +
            (self.intelligence) * 0.1
        )
        self.fitness = max(0.1, min(1.0, self.fitness))
    
    def get_classification(self):
        if self.fitness > 0.8 and self.generation >= 3:
            return "elite"
        elif self.fitness > 0.65:
            return "veteran"
        elif self.fitness > 0.45:
            return "strong"
        elif self.energy > 80:
            return "energetic"
        elif self.age < 100:
            return "young"
        else:
            return "basic"
    
    def find_nearest_food(self, food_list):
        if not food_list:
            return None
            
        nearest_food = None
        min_distance = self.vision_range
        
        for food_item in food_list:
            distance = math.sqrt((self.x - food_item["x"])**2 + (self.y - food_item["y"])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_food = food_item
                
        return nearest_food
    
    def move_towards_food(self, food_item):
        if food_item:
            target_x = food_item["x"]
            target_y = food_item["y"]
            
            # Calculate direction to food
            dx = target_x - self.x
            dy = target_y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Move towards food with intelligence factor
                move_efficiency = 0.5 + (self.intelligence * 0.5)
                self.direction = math.atan2(dy, dx) + random.uniform(-0.3, 0.3) * (1 - self.intelligence)
                return True
        return False
    
    def move(self, food_list):
        old_x, old_y = self.x, self.y
        
        # Try to find and move towards food
        if not self.move_towards_food(self.find_nearest_food(food_list)):
            # Random movement if no food found
            if random.random() < 0.15:
                self.direction += random.uniform(-0.8, 0.8)
        
        # Apply movement
        speed_factor = self.dna['speed_gene'] * self.speed
        self.x += math.cos(self.direction) * speed_factor
        self.y += math.sin(self.direction) * speed_factor
        
        # Boundary collision with bouncing
        if self.x < self.size or self.x > CANVAS_WIDTH - self.size:
            self.direction = math.pi - self.direction
            self.x = max(self.size, min(CANVAS_WIDTH - self.size, self.x))
            
        if self.y < self.size or self.y > CANVAS_HEIGHT - self.size:
            self.direction = -self.direction
            self.y = max(self.size, min(CANVAS_HEIGHT - self.size, self.y))
        
        # Track distance traveled
        self.distance_traveled += math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
        
        # Energy consumption based on activity
        energy_cost = ENERGY_DECAY * (1 + speed_factor * 0.1)
        self.energy -= energy_cost
        
        # Age and cooldowns
        self.age += 1
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
            
        # Update classification
        self.classification = self.get_classification()
        self.update_fitness()
    
    def can_reproduce(self):
        return (self.energy > 90 and 
                self.reproduction_cooldown == 0 and 
                self.age > 50 and
                self.fitness > 0.4)
    
    def reproduce(self):
        if self.can_reproduce():
            # Create offspring near parent
            offset_x = random.uniform(-30, 30)
            offset_y = random.uniform(-30, 30)
            new_x = max(20, min(CANVAS_WIDTH - 20, self.x + offset_x))
            new_y = max(20, min(CANVAS_HEIGHT - 20, self.y + offset_y))
            
            offspring = AdvancedBacterium(new_x, new_y, parent=self)
            
            # Reproduction cost
            self.energy -= 40
            self.reproduction_cooldown = 200
            self.offspring_count += 1
            
            return offspring
        return None
    
    def to_dict(self):
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "size": round(self.size, 2),
            "energy": round(self.energy, 2),
            "fitness": round(self.fitness, 3),
            "age": self.age,
            "generation": self.generation,
            "classification": self.classification,
            "intelligence": round(self.intelligence, 3),
            "speed": round(self.speed, 3),
            "food_eaten": self.food_eaten,
            "offspring_count": self.offspring_count
        }

def create_food():
    return {
        "x": random.uniform(15, CANVAS_WIDTH - 15),
        "y": random.uniform(15, CANVAS_HEIGHT - 15),
        "energy": random.uniform(25, 50),
        "quality": random.uniform(0.5, 1.5)
    }

def initialize_simulation(bacteria_count=30):
    """Thread-safe simulation initialization"""
    with simulation_lock:
        logger.info(f"🔄 Initializing simulation with {bacteria_count} bacteria")
        
        # Create initial bacteria
        bacteria = []
        for _ in range(bacteria_count):
            x = random.uniform(50, CANVAS_WIDTH - 50)
            y = random.uniform(50, CANVAS_HEIGHT - 50)
            bacteria.append(AdvancedBacterium(x, y))
        
        # Create food
        food = [create_food() for _ in range(FOOD_COUNT)]
        
        # Thread-safe state update
        update_simulation_state({
            "bacteria": bacteria,
            "food": food,
            "step": 0,
            "mutations": 0,
            "deaths": 0,
            "births": 0,
            "total_bacteria": len(bacteria),
            "avg_fitness": sum(b.fitness for b in bacteria) / len(bacteria) if bacteria else 0,
            "avg_energy": sum(b.energy for b in bacteria) / len(bacteria) if bacteria else 0,
            "last_update": time.time()
        })
        logger.info(f"✅ Simulation initialized: {len(bacteria)} bacteria, {len(food)} food")

def simulation_step():
    """Thread-safe simulation step execution"""
    with simulation_lock:
        # Check simulation state safely
        if not get_simulation_state("running") or get_simulation_state("paused"):
            return
        
        start_time = time.time()
        bacteria = get_simulation_state("bacteria")
        food = get_simulation_state("food")
        
        if not bacteria:  # Safety check
            logger.warning("⚠️ No bacteria in simulation step")
            return
    
    births_this_step = 0
    deaths_this_step = 0
    
    # Move bacteria and handle interactions
    for bacterium in bacteria[:]:
        bacterium.move(food)
        
        # Check food consumption
        for food_item in food[:]:
            distance = math.sqrt((bacterium.x - food_item["x"])**2 + (bacterium.y - food_item["y"])**2)
            if distance < bacterium.size:
                # Consume food
                energy_gain = food_item["energy"] * food_item["quality"]
                bacterium.energy += energy_gain
                bacterium.food_eaten += 1
                food.remove(food_item)
                break
        
        # Check for death
        if bacterium.energy <= 0:
            bacteria.remove(bacterium)
            deaths_this_step += 1
            continue
            
        # Check for reproduction
        if len(bacteria) < MAX_BACTERIA:
            offspring = bacterium.reproduce()
            if offspring:
                bacteria.append(offspring)
                births_this_step += 1
    
        # Maintain food supply
        while len(food) < FOOD_COUNT:
            food.append(create_food())
        
        # Thread-safe statistics update
        current_step = get_simulation_state("step")
        update_simulation_state({
            "bacteria": bacteria,
            "food": food,
            "step": current_step + 1,
            "deaths": get_simulation_state("deaths") + deaths_this_step,
            "births": get_simulation_state("births") + births_this_step
        })
        
        if bacteria:
            stats_update = {
                "total_bacteria": len(bacteria),
                "avg_fitness": sum(b.fitness for b in bacteria) / len(bacteria),
                "avg_energy": sum(b.energy for b in bacteria) / len(bacteria),
            }
            update_simulation_state(stats_update)
            
            # Record history every 10 steps (thread-safe)
            new_step = current_step + 1
            if new_step % 10 == 0:
                history = get_simulation_state("history")
                history["population"].append(len(bacteria))
                history["fitness"].append(stats_update["avg_fitness"])
                history["energy"].append(stats_update["avg_energy"])
                history["mutations"].append(get_simulation_state("mutations"))
                history["timestamp"].append(time.time())
                
                # Keep only last 100 records
                for key in history:
                    if len(history[key]) > 100:
                        history[key] = history[key][-100:]
                
                set_simulation_state("history", history)
        
        # Calculate FPS
        step_time = time.time() - start_time
        update_simulation_state({
            "fps": round(1.0 / max(step_time, 0.001), 1),
            "last_update": time.time()
        })

def simulation_loop():
    while True:
        if simulation_state["running"] and not simulation_state["paused"]:
            simulation_step()
        time.sleep(1.0 / SIMULATION_FPS)  # Configurable FPS

# --- Enhanced Background Data Emitter with Thread Safety ---
def background_data_emitter():
    """Enhanced thread-safe real-time data emitter with error handling"""
    logger.info("📡 Enhanced background data emitter started")
    error_count = 0
    max_errors = 5
    
    while True:
        if get_simulation_state("running") and not get_simulation_state("paused"):
            try:
                # Thread-safe data acquisition
                with data_lock:
                    bacteria = get_simulation_state("bacteria")
                    food = get_simulation_state("food")
                    
                    if not bacteria:  # Safety check
                        logger.debug("📡 No bacteria data to emit")
                        time.sleep(1.0 / 10.0)
                        continue
                
                # Prepare optimized data for emission
                bacteria_sample = [b.to_dict() for b in bacteria[:50]]  # Limit sample
                food_sample = food[:30] if food else []  # Safety check for food
                
                # Classification distribution (thread-safe)
                classifications = {}
                for bacterium in bacteria:
                    cls = bacterium.classification
                    classifications[cls] = classifications.get(cls, 0) + 1
                
                data = {
                    'status': 'running',
                    'step': get_simulation_state("step"),
                    'bacteria_count': len(bacteria),
                    'avg_fitness': round(get_simulation_state("avg_fitness"), 3),
                    'avg_energy': round(get_simulation_state("avg_energy"), 1),
                    'generation': get_simulation_state("generation"),
                    'fps': get_simulation_state("fps"),
                    'bacteria_sample': bacteria_sample,
                    'food_sample': food_sample,
                    'classifications': classifications,
                    'history': get_simulation_state("history"),
                    'world_dimensions': [CANVAS_WIDTH, CANVAS_HEIGHT],
                    'timestamp': time.time()
                }
                
                # Emit to all connected clients with error handling
                socketio.emit('simulation_update', data)
                error_count = 0  # Reset error count on success
                logger.debug(f"📡 Data emitted: step {data['step']}, bacteria {data['bacteria_count']}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Background emitter error #{error_count}: {e}")
                
                # Emit error to clients if persistent issues
                if error_count >= max_errors:
                    socketio.emit('simulation_error', {
                        'message': f'Veri yayını hatası: {str(e)}',
                        'error_count': error_count,
                        'timestamp': time.time()
                    })
                    logger.critical(f"🚨 Max errors reached ({max_errors}), notifying clients")
                    error_count = 0  # Reset after notification
        
        time.sleep(1.0 / 10.0)  # 10 FPS emission rate

# Start simulation thread
simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
simulation_thread.start()

# Start background data emitter thread
emitter_thread = threading.Thread(target=background_data_emitter, daemon=True)
emitter_thread.start()
logger.info("📡 Background data emitter thread started")

@app.route('/')
def index():
    logger.info("🔍 RENDERING: advanced_index.html template")
    return render_template('advanced_index.html')

@app.route('/favicon.ico')
def favicon():
    """Prevent favicon 404 errors"""
    return '', 204  # No Content

@app.route('/api/status')
def get_status():
    return jsonify({
        "running": simulation_state["running"],
        "paused": simulation_state["paused"],
        "step": simulation_state["step"],
        "generation": simulation_state["generation"],
        "total_bacteria": simulation_state["total_bacteria"],
        "avg_fitness": round(simulation_state["avg_fitness"], 3),
        "avg_energy": round(simulation_state["avg_energy"], 1),
        "fps": simulation_state["fps"],
        "mutations": simulation_state["mutations"],
        "deaths": simulation_state["deaths"],
        "births": simulation_state["births"],
        "last_update": simulation_state["last_update"]
    })

@app.route('/api/data')
def get_simulation_data():
    bacteria_data = [b.to_dict() for b in simulation_state["bacteria"]]
    
    # Classification counts
    classifications = {}
    for bacterium in simulation_state["bacteria"]:
        cls = bacterium.classification
        classifications[cls] = classifications.get(cls, 0) + 1
    
    return jsonify({
        "bacteria": bacteria_data,
        "food": simulation_state["food"],
        "stats": {
            "total_bacteria": simulation_state["total_bacteria"],
            "avg_fitness": round(simulation_state["avg_fitness"], 3),
            "avg_energy": round(simulation_state["avg_energy"], 1),
            "step": simulation_state["step"],
            "generation": simulation_state["generation"],
            "fps": simulation_state["fps"],
            "classifications": classifications
        },
        "history": simulation_state["history"]
    })

@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Thread-safe simulation start/resume"""
    with simulation_lock:
        try:
            data = request.get_json() or {}
            bacteria_count = data.get('bacteria_count', 30)
            
            if not get_simulation_state("running"):
                initialize_simulation(bacteria_count)
                update_simulation_state({"running": True, "paused": False})
                logger.info(f"✅ Simulation started with {bacteria_count} bacteria")
                message = "Simülasyon başlatıldı"
            else:
                set_simulation_state("paused", False)
                logger.info("▶️ Simulation resumed")
                message = "Simülasyon devam ettirildi"
            
            return jsonify({"status": "started", "message": message})
        except Exception as e:
            logger.error(f"❌ Error starting simulation: {e}")
            return jsonify({"status": "error", "message": f"Başlatma hatası: {str(e)}"}), 500

@app.route('/api/pause', methods=['POST'])
def pause_simulation():
    set_simulation_state("paused", True) # Use thread-safe setter
    logger.info("⏸️ Simulation paused")
    
    return jsonify({"status": "paused", "message": "Simülasyon duraklatıldı"})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    update_simulation_state({"running": False, "paused": False}) # Use thread-safe update
    logger.info("⏹️ Simulation stopped")
    
    return jsonify({"status": "stopped", "message": "Simülasyon durduruldu"})

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    data = request.get_json() or {}
    bacteria_count = data.get('bacteria_count', 30)
    
    update_simulation_state({"running": False, "paused": False}) # Use thread-safe update
    initialize_simulation(bacteria_count) # This is already thread-safe
    logger.info("🔄 Simulation reset")
    
    return jsonify({"status": "reset", "message": "Simülasyon sıfırlandı"})

@app.route('/api/add_bacteria', methods=['POST'])
def add_bacteria():
    """Thread-safe bacteria addition"""
    with simulation_lock:
        try:
            if not get_simulation_state("running"):
                return jsonify({"status": "error", "message": "Simülasyon çalışmıyor"})
            
            data = request.get_json() or {}
            count = data.get('count', 10)
            
            bacteria = get_simulation_state("bacteria") # Get a copy
            current_bacteria_list = list(bacteria) # Ensure it's a mutable list if it's a tuple from get_simulation_state
            
            added_count = 0
            
            for _ in range(count):
                if len(current_bacteria_list) >= MAX_BACTERIA:
                    break
                x = random.uniform(50, CANVAS_WIDTH - 50)
                y = random.uniform(50, CANVAS_HEIGHT - 50)
                current_bacteria_list.append(AdvancedBacterium(x, y))
                added_count += 1
            
            # Update state with new bacteria list
            set_simulation_state("bacteria", current_bacteria_list)
            
            logger.info(f"➕ Added {added_count} bacteria (total: {len(current_bacteria_list)})")
            return jsonify({
                "status": "success", 
                "message": f"{added_count} bakteri eklendi",
                "total_bacteria": len(current_bacteria_list),
                "added": added_count
            })
        except Exception as e:
            logger.error(f"❌ Error adding bacteria: {e}")
            return jsonify({"status": "error", "message": f"Bakteri ekleme hatası: {str(e)}"}), 500

@app.route('/api/export', methods=['GET'])
def export_data():
    bacteria_data = []
    for bacterium in simulation_state["bacteria"]:
        data = bacterium.to_dict()
        data.update({
            "distance_traveled": round(bacterium.distance_traveled, 2),
            "dna": bacterium.dna
        })
        bacteria_data.append(data)
    
    # Create CSV
    output = io.StringIO()
    if bacteria_data:
        writer = csv.DictWriter(output, fieldnames=bacteria_data[0].keys())
        writer.writeheader()
        writer.writerows(bacteria_data)
    
    # Create response
    csv_data = output.getvalue()
    output.close()
    
    response = app.response_class(
        response=csv_data,
        status=200,
        mimetype='text/csv'
    )
    response.headers["Content-Disposition"] = f"attachment; filename=neomag_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return response

@app.route('/api/ai_analysis', methods=['GET'])
def ai_analysis():
    bacteria = simulation_state["bacteria"]
    
    if not bacteria:
        return jsonify({"status": "error", "message": "Analiz için veri yok"})
    
    # Enhanced AI analysis using Gemini
    bacteria_data = [b.to_dict() for b in bacteria]
    ai_analysis_text = gemini_ai.analyze_population(bacteria_data)
    
    # Basic statistics
    classifications = {}
    for bacterium in bacteria:
        cls = bacterium.classification
        classifications[cls] = classifications.get(cls, 0) + 1
    
    analysis = {
        "ai_analysis": ai_analysis_text,
        "population_health": "İyi" if simulation_state["avg_fitness"] > 0.6 else "Kötü",
        "diversity": len(set(b.classification for b in bacteria)),
        "classifications": classifications,
        "top_performers": sorted([{
            "fitness": b.fitness,
            "energy": b.energy,
            "generation": b.generation,
            "classification": b.classification
        } for b in bacteria], key=lambda x: x["fitness"], reverse=True)[:5],
        "recommendations": [
            "Popülasyon stabil" if len(bacteria) > 20 else "Daha fazla bakteri ekleyin",
            "Fitness seviyesi iyi" if simulation_state["avg_fitness"] > 0.5 else "Çevre koşullarını iyileştirin",
            "Genetik çeşitlilik yeterli" if len(set(b.generation for b in bacteria)) > 2 else "Yeni genetik çeşitlilik gerekli"
        ]
    }
    
    return jsonify({
        "status": "success",
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/ai_question', methods=['POST'])
def ai_question():
    """Answer user questions about the simulation using Gemini AI"""
    try:
        data = request.get_json() or {}
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'status': 'error', 'message': 'Soru belirtilmedi.'}), 400
        
        # Prepare simulation context
        bacteria_count = len(simulation_state["bacteria"])
        context = f"""
        Mevcut simülasyon durumu:
        - Adım: {simulation_state["step"]}
        - Bakteri sayısı: {bacteria_count}
        - Ortalama fitness: {simulation_state["avg_fitness"]:.3f}
        - Ortalama enerji: {simulation_state["avg_energy"]:.1f}
        - Nesil: {simulation_state["generation"]}
        - Durum: {"Çalışıyor" if simulation_state["running"] else "Durdurulmuş"}
        """
        
        # Create detailed prompt for Gemini
        prompt = f"""
        NeoMag V7 bakteriyel simülasyon uzmanı olarak kullanıcının sorusunu yanıtlayın.
        
        {context}
        
        Kullanıcı sorusu: "{question}"
        
        Lütfen bilimsel temellere dayanan, anlaşılır ve faydalı bir yanıt verin.
        Cevabınızı Türkçe olarak 100-150 kelime arasında tutun.
        """
        
        answer = gemini_ai._make_request(prompt, max_tokens=180)
        
        return jsonify({
            'status': 'success',
            'question': question,
            'answer': answer.get('text', 'AI yanıt veremiyor.'),
            'simulation_context': context.strip()
        })
        
    except Exception as e:
        logger.error(f"AI question error: {e}")
        return jsonify({'status': 'error', 'message': f"AI soru hatası: {str(e)}"}), 500

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    logger.info(f"🔌 Client connected: {request.sid}")
    emit('connection_ack', {'status': 'NeoMag V7 sunucusuna başarıyla bağlanıldı.'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"🔌 Client disconnected: {request.sid}")

@socketio.on('request_data')
def handle_data_request():
    """Handle explicit data requests from clients"""
    if simulation_state["bacteria"]:
        bacteria_sample = [b.to_dict() for b in simulation_state["bacteria"][:50]]
        data = {
            'status': 'running' if simulation_state["running"] else 'stopped',
            'step': simulation_state["step"],
            'bacteria_count': len(simulation_state["bacteria"]),
            'bacteria_sample': bacteria_sample,
            'food_sample': simulation_state["food"][:30],
            'timestamp': time.time()
        }
        emit('simulation_update', data)

if __name__ == '__main__':
    SERVER_PORT = int(os.environ.get('SERVER_PORT', 5000))  # DEFAULT 5000
    SERVER_HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
    DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("🧬 NeoMag V7 Advanced Server Starting...")
    print(f"🌍 Server: {SERVER_HOST}:{SERVER_PORT}")
    print(f"🔧 Debug Mode: {DEBUG_MODE}")
    print(f"📊 Max Bacteria: {MAX_BACTERIA}")
    print(f"⚡ Simulation FPS: {SIMULATION_FPS}")
    
    # Initialize simulation with default settings
    initialize_simulation()
    
    # Start server with SocketIO
    socketio.run(app, host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE, use_reloader=False) 