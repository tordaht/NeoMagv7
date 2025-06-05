"""
NeoMag V7 - Global Constants
Proje genelinde kullanılan sabit değerler
"""

# Canvas dimensions
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 600

# Simulation parameters
MAX_BACTERIA = 500
MIN_BACTERIA = 10
DEFAULT_BACTERIA_COUNT = 30
DEFAULT_FOOD_COUNT = 100

# Physics constants
DEFAULT_SPEED = 2.0
MAX_SPEED = 5.0
MIN_SPEED = 0.5
COLLISION_RADIUS = 15

# Energy constants
INITIAL_ENERGY = 100.0
MAX_ENERGY = 150.0
MIN_ENERGY = 0.0
ENERGY_CONSUMPTION_RATE = 0.5
FOOD_ENERGY_VALUE = 20.0

# Genetic constants
MUTATION_RATE = 0.05
DNA_LENGTH = 10
REPRODUCTION_THRESHOLD = 80.0

# Visualization colors
COLORS = {
    'elite': '#ffd700',
    'veteran': '#4169e1',
    'strong': '#32cd32',
    'energetic': '#ff6347',
    'young': '#00bfff',
    'basic': '#ff8c00'
}

# Server constants
DEFAULT_PORT = 5000
DEFAULT_HOST = '0.0.0.0'
SOCKET_TIMEOUT = 60
PING_INTERVAL = 25

# AI/ML constants
TABPFN_ENSEMBLE_SIZE = 8
MIN_CONFIDENCE_THRESHOLD = 0.7
ANALYSIS_INTERVAL = 30  # seconds
