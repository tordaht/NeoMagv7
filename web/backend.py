"""
NeoMag V7 - Web Backend Services
Web uygulaması için backend servisleri
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import redis
import sqlite3
from datetime import datetime, timedelta
import json
import logging
import os
from typing import Dict, List, Optional, Any
import jwt
import bcrypt
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Veritabanı yönetimi
    """
    def __init__(self, db_path: str = "neomag_v7.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Veritabanı tablolarını oluştur"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Simülasyon kayıtları
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    name TEXT NOT NULL,
                    config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'created'
                )
            """)
            
            # Simülasyon verileri
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER,
                    step INTEGER,
                    data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
                )
            """)
            
            # Analiz sonuçları
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER,
                    analysis_type TEXT,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
                )
            """)
            
            # Kullanıcılar
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # API anahtarları
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    key TEXT UNIQUE NOT NULL,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def save_simulation(self, user_id: int, name: str, config: Dict) -> int:
        """Simülasyonu kaydet"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO simulations (user_id, name, config) VALUES (?, ?, ?)",
                (user_id, name, json.dumps(config))
            )
            conn.commit()
            return cursor.lastrowid
    
    def save_simulation_data(self, simulation_id: int, step: int, data: Dict):
        """Simülasyon verisini kaydet"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO simulation_data (simulation_id, step, data) VALUES (?, ?, ?)",
                (simulation_id, step, json.dumps(data))
            )
            conn.commit()
    
    def get_simulation_history(self, simulation_id: int, limit: int = 100) -> List[Dict]:
        """Simülasyon geçmişini al"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT step, data, timestamp FROM simulation_data 
                   WHERE simulation_id = ? 
                   ORDER BY step DESC LIMIT ?""",
                (simulation_id, limit)
            )
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'step': row[0],
                    'data': json.loads(row[1]),
                    'timestamp': row[2]
                })
            
            return results
    
    def save_analysis_result(self, simulation_id: int, analysis_type: str, result: Dict):
        """Analiz sonucunu kaydet"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analysis_results (simulation_id, analysis_type, result) VALUES (?, ?, ?)",
                (simulation_id, analysis_type, json.dumps(result))
            )
            conn.commit()


class CacheManager:
    """
    Redis tabanlı önbellek yönetimi
    """
    def __init__(self, redis_url: str = None):
        try:
            self.redis_client = redis.from_url(
                redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
            )
            self.redis_client.ping()
            self.enabled = True
            logger.info("Redis cache connected")
        except:
            logger.warning("Redis not available, caching disabled")
            self.enabled = False
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Önbellekten değer al"""
        if not self.enabled:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Önbelleğe değer kaydet"""
        if not self.enabled:
            return
        
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Önbellekten sil"""
        if not self.enabled:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def clear_pattern(self, pattern: str):
        """Pattern'e uyan anahtarları temizle"""
        if not self.enabled:
            return
        
        try:
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")


class AuthManager:
    """
    Kimlik doğrulama yönetimi
    """
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv('JWT_SECRET_KEY', 'your-secret-key')
        self.db = DatabaseManager()
    
    def hash_password(self, password: str) -> str:
        """Şifreyi hashle"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hash: str) -> bool:
        """Şifreyi doğrula"""
        return bcrypt.checkpw(password.encode('utf-8'), hash.encode('utf-8'))
    
    def generate_token(self, user_id: int) -> str:
        """JWT token oluştur"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=7)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[int]:
        """JWT token doğrula"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload.get('user_id')
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
        
        return None
    
    def create_user(self, username: str, email: str, password: str) -> Optional[int]:
        """Yeni kullanıcı oluştur"""
        try:
            password_hash = self.hash_password(password)
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, password_hash)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            logger.error("User already exists")
            return None
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Kullanıcı kimlik doğrulama"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, password_hash, email FROM users WHERE username = ? AND is_active = 1",
                (username,)
            )
            
            row = cursor.fetchone()
            if row and self.verify_password(password, row[1]):
                # Son giriş zamanını güncelle
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (row[0],)
                )
                conn.commit()
                
                return {
                    'id': row[0],
                    'username': username,
                    'email': row[2],
                    'token': self.generate_token(row[0])
                }
        
        return None


def require_auth(f):
    """Kimlik doğrulama gerektiren decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        auth_manager = AuthManager()
        user_id = auth_manager.verify_token(token)
        
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.user_id = user_id
        return f(*args, **kwargs)
    
    return decorated_function


class BackendAPI:
    """
    Backend API servisi
    """
    def __init__(self, app: Flask = None):
        self.app = app or Flask(__name__)
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.auth = AuthManager()
        
        # Rate limiting
        self.limiter = Limiter(
            self.app,
            key_func=get_remote_address,
            default_limits=["100 per hour"]
        )
        
        # CORS
        CORS(self.app, resources={r"/api/*": {"origins": "*"}})
        
        # Flask caching
        self.flask_cache = Cache(self.app, config={
            'CACHE_TYPE': 'simple',
            'CACHE_DEFAULT_TIMEOUT': 300
        })
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Routes
        self.setup_routes()
    
    def setup_routes(self):
        """API route'larını kur"""
        
        @self.app.route('/api/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            })
        
        @self.app.route('/api/auth/register', methods=['POST'])
        @self.limiter.limit("3 per hour")
        def register():
            data = request.get_json()
            
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            
            if not all([username, email, password]):
                return jsonify({'error': 'Missing required fields'}), 400
            
            user_id = self.auth.create_user(username, email, password)
            
            if user_id:
                return jsonify({
                    'user_id': user_id,
                    'token': self.auth.generate_token(user_id)
                }), 201
            else:
                return jsonify({'error': 'User already exists'}), 409
        
        @self.app.route('/api/auth/login', methods=['POST'])
        def login():
            data = request.get_json()
            
            username = data.get('username')
            password = data.get('password')
            
            if not all([username, password]):
                return jsonify({'error': 'Missing credentials'}), 400
            
            user = self.auth.authenticate(username, password)
            
            if user:
                return jsonify(user)
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
        
        @self.app.route('/api/simulations', methods=['GET'])
        @require_auth
        @self.flask_cache.cached(timeout=60)
        def get_simulations():
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT id, name, config, created_at, status 
                       FROM simulations 
                       WHERE user_id = ? 
                       ORDER BY created_at DESC""",
                    (request.user_id,)
                )
                
                simulations = []
                for row in cursor.fetchall():
                    simulations.append({
                        'id': row[0],
                        'name': row[1],
                        'config': json.loads(row[2]),
                        'created_at': row[3],
                        'status': row[4]
                    })
                
                return jsonify(simulations)
        
        @self.app.route('/api/simulations/<int:sim_id>/data', methods=['GET'])
        @require_auth
        def get_simulation_data(sim_id):
            # Cache kontrolü
            cache_key = f"sim_data_{sim_id}_{request.user_id}"
            cached = self.cache.get(cache_key)
            
            if cached:
                return jsonify(cached)
            
            # Veritabanından al
            data = self.db.get_simulation_history(sim_id)
            
            # Cache'e kaydet
            self.cache.set(cache_key, data, ttl=60)
            
            return jsonify(data)
        
        @self.app.route('/api/simulations/<int:sim_id>/analyze', methods=['POST'])
        @require_auth
        @self.limiter.limit("10 per hour")
        def analyze_simulation(sim_id):
            data = request.get_json()
            analysis_type = data.get('type', 'basic')
            
            # Async analiz başlat
            future = self.executor.submit(
                self._run_analysis,
                sim_id,
                analysis_type
            )
            
            return jsonify({
                'message': 'Analysis started',
                'task_id': id(future)
            }), 202
        
        @self.app.route('/api/export/<int:sim_id>', methods=['GET'])
        @require_auth
        def export_simulation(sim_id):
            format_type = request.args.get('format', 'json')
            
            # Simülasyon verilerini al
            data = self.db.get_simulation_history(sim_id, limit=10000)
            
            if format_type == 'csv':
                # CSV formatında export
                import csv
                import io
                
                output = io.StringIO()
                if data:
                    writer = csv.DictWriter(output, fieldnames=data[0]['data'].keys())
                    writer.writeheader()
                    for record in data:
                        writer.writerow(record['data'])
                
                response = Response(
                    output.getvalue(),
                    mimetype='text/csv',
                    headers={
                        'Content-Disposition': f'attachment; filename=simulation_{sim_id}.csv'
                    }
                )
                return response
            else:
                # JSON formatında export
                return jsonify(data)
    
    def _run_analysis(self, simulation_id: int, analysis_type: str):
        """Async analiz çalıştır"""
        try:
            # Simülasyon verilerini al
            data = self.db.get_simulation_history(simulation_id)
            
            # Analiz yap (basit örnek)
            result = {
                'type': analysis_type,
                'total_steps': len(data),
                'summary': 'Analysis completed successfully',
                'metrics': {}
            }
            
            # Sonucu kaydet
            self.db.save_analysis_result(simulation_id, analysis_type, result)
            
            # Cache'i temizle
            self.cache.delete(f"sim_data_{simulation_id}_*")
            
            logger.info(f"Analysis completed for simulation {simulation_id}")
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")


def create_app() -> Flask:
    """Flask uygulaması oluştur"""
    app = Flask(__name__)
    
    # Konfigürasyon
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True
    )
    
    # Backend API'yi başlat
    backend = BackendAPI(app)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5001)
