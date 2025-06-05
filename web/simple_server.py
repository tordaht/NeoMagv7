#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeoMag V7 - Simple HTTP Server (Socket.IO yerine)
"""

from flask import Flask, jsonify, render_template, request
import json
import time
import threading
import random
import math
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global simulation state
simulation_state = {
    "running": False,
    "bacteria": [],
    "food": [],
    "generation": 1,
    "total_bacteria": 0,
    "avg_fitness": 0.0,
    "last_update": time.time()
}

# Simulation parameters
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
MAX_BACTERIA = 50
FOOD_COUNT = 20

class SimpleBacterium:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = random.uniform(50, 100)
        self.size = random.uniform(8, 15)
        self.speed = random.uniform(1, 3)
        self.direction = random.uniform(0, 2 * math.pi)
        self.fitness = random.uniform(0.3, 0.9)
        self.age = 0
        self.classification = self.get_classification()
    
    def get_classification(self):
        if self.fitness > 0.8:
            return "elite"
        elif self.fitness > 0.6:
            return "veteran"
        elif self.fitness > 0.4:
            return "strong"
        else:
            return "basic"
    
    def move(self):
        # Simple movement
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed
        
        # Boundary check
        if self.x < 0 or self.x > CANVAS_WIDTH:
            self.direction = math.pi - self.direction
        if self.y < 0 or self.y > CANVAS_HEIGHT:
            self.direction = -self.direction
            
        self.x = max(0, min(CANVAS_WIDTH, self.x))
        self.y = max(0, min(CANVAS_HEIGHT, self.y))
        
        # Energy consumption
        self.energy -= 0.5
        self.age += 1
        
        # Random direction change
        if random.random() < 0.1:
            self.direction += random.uniform(-0.5, 0.5)
    
    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "energy": self.energy,
            "fitness": self.fitness,
            "age": self.age,
            "classification": self.classification
        }

def create_food():
    return {
        "x": random.uniform(10, CANVAS_WIDTH - 10),
        "y": random.uniform(10, CANVAS_HEIGHT - 10),
        "energy": random.uniform(20, 40)
    }

def initialize_simulation():
    global simulation_state
    
    # Create initial bacteria
    bacteria = []
    for _ in range(20):
        x = random.uniform(50, CANVAS_WIDTH - 50)
        y = random.uniform(50, CANVAS_HEIGHT - 50)
        bacteria.append(SimpleBacterium(x, y))
    
    # Create food
    food = [create_food() for _ in range(FOOD_COUNT)]
    
    simulation_state.update({
        "bacteria": bacteria,
        "food": food,
        "total_bacteria": len(bacteria),
        "avg_fitness": sum(b.fitness for b in bacteria) / len(bacteria),
        "last_update": time.time()
    })

def simulation_step():
    global simulation_state
    
    if not simulation_state["running"]:
        return
    
    bacteria = simulation_state["bacteria"]
    food = simulation_state["food"]
    
    # Move bacteria
    for bacterium in bacteria[:]:
        bacterium.move()
        
        # Check food consumption
        for food_item in food[:]:
            distance = math.sqrt((bacterium.x - food_item["x"])**2 + (bacterium.y - food_item["y"])**2)
            if distance < bacterium.size:
                bacterium.energy += food_item["energy"]
                food.remove(food_item)
                break
        
        # Remove dead bacteria
        if bacterium.energy <= 0:
            bacteria.remove(bacterium)
    
    # Add new food
    while len(food) < FOOD_COUNT:
        food.append(create_food())
    
    # Reproduction
    if len(bacteria) < MAX_BACTERIA:
        for bacterium in bacteria[:]:
            if bacterium.energy > 80 and random.random() < 0.05:
                # Create offspring
                new_x = bacterium.x + random.uniform(-20, 20)
                new_y = bacterium.y + random.uniform(-20, 20)
                new_x = max(0, min(CANVAS_WIDTH, new_x))
                new_y = max(0, min(CANVAS_HEIGHT, new_y))
                
                offspring = SimpleBacterium(new_x, new_y)
                offspring.fitness = bacterium.fitness + random.uniform(-0.1, 0.1)
                offspring.fitness = max(0.1, min(1.0, offspring.fitness))
                
                bacteria.append(offspring)
                bacterium.energy -= 30
    
    # Update statistics
    if bacteria:
        simulation_state.update({
            "total_bacteria": len(bacteria),
            "avg_fitness": sum(b.fitness for b in bacteria) / len(bacteria),
            "last_update": time.time()
        })

def simulation_loop():
    while True:
        if simulation_state["running"]:
            simulation_step()
        time.sleep(0.2)  # 5 FPS - OPTIMIZED

# Start simulation thread
simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
simulation_thread.start()

@app.route('/')
def index():
    return render_template('simple_index.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        "running": simulation_state["running"],
        "generation": simulation_state["generation"],
        "total_bacteria": simulation_state["total_bacteria"],
        "avg_fitness": round(simulation_state["avg_fitness"], 3),
        "last_update": simulation_state["last_update"]
    })

@app.route('/api/data')
def get_simulation_data():
    bacteria_data = [b.to_dict() for b in simulation_state["bacteria"]]
    
    return jsonify({
        "bacteria": bacteria_data,
        "food": simulation_state["food"],
        "stats": {
            "total_bacteria": simulation_state["total_bacteria"],
            "avg_fitness": round(simulation_state["avg_fitness"], 3),
            "generation": simulation_state["generation"]
        }
    })

@app.route('/api/start', methods=['POST'])
def start_simulation():
    global simulation_state
    
    if not simulation_state["running"]:
        initialize_simulation()
        simulation_state["running"] = True
        logger.info("Simulation started")
    
    return jsonify({"status": "started", "message": "Simülasyon başlatıldı"})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    global simulation_state
    simulation_state["running"] = False
    logger.info("Simulation stopped")
    
    return jsonify({"status": "stopped", "message": "Simülasyon durduruldu"})

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    global simulation_state
    simulation_state["running"] = False
    initialize_simulation()
    logger.info("Simulation reset")
    
    return jsonify({"status": "reset", "message": "Simülasyon sıfırlandı"})

if __name__ == '__main__':
    print("🧬 NeoMag V7 Simple Server Starting...")
    initialize_simulation()
    app.run(host='0.0.0.0', port=5000, debug=True) 
