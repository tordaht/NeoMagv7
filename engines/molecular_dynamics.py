"""NeoMag V7 - Molecular Dynamics Engine - PERFORMANCE OPTIMIZED"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any
import time
from collections import deque
import sys
sys.path.append('..')

# Optional imports
try:
    import cupy as cp
    # Test if cupy actually works
    cp.array([1, 2, 3])
    HAS_CUPY = True
    logging.info("🚀 CuPy GPU acceleration ENABLED")
except Exception as e:
    cp = None
    HAS_CUPY = False
    logging.info("⚠️ CuPy not available - Using CPU mode")

try:
    from numba import jit, prange
    HAS_NUMBA = True
    logging.info("🚀 Numba JIT acceleration ENABLED")
except ImportError:
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
    HAS_NUMBA = False
    logging.info("⚠️ Numba not available - Using pure Python")

try:
    from ..core.biophysical_properties import BiophysicalProperties, MolecularForceType
    from ..agents.bacterium import AdvancedBacteriumV7
except ImportError:
    # Fallback for direct execution
    from core.biophysical_properties import BiophysicalProperties, MolecularForceType

# Bilimsel Sabitler
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
AVOGADRO_NUMBER = 6.02214076e23   # mol^-1
GAS_CONSTANT = 8.314462618        # J/(mol·K)
ELEMENTARY_CHARGE = 1.602176634e-19  # C

class MolecularDynamicsEngine:
    """GPU-accelerated molecular dynamics engine - PERFORMANCE OPTIMIZED"""
    def __init__(self, time_step=1e-15, temperature=300, pressure=1,
                 force_field='AMBER', ensemble='NVE', thermostat_params=None, barostat_params=None,
                 use_gpu=True, pme_parameters=None, cg_model_parameters=None,
                 lj_epsilon=1.0, lj_sigma=3.0, electrostatic_constant=332.0637, cutoff=12.0):
        self.time_step = time_step
        self.temperature = temperature
        self.pressure = pressure
        self.force_field = force_field
        self.ensemble = ensemble
        self.thermostat_params = thermostat_params if thermostat_params else {}
        self.barostat_params = barostat_params if barostat_params else {}
        
        # HYBRID GPU/CPU Setup
        self.prefer_gpu = use_gpu
        self.use_gpu = False  # Force CPU mode for stability
        if use_gpu and HAS_CUPY:
            try:
                if cp.cuda.is_available():
                    self.use_gpu = True
                    logging.info("🚀 GPU mode enabled")
            except Exception as e:
                logging.info(f"⚠️ GPU unavailable, using CPU: {e}")
                self.use_gpu = False
        self.xp = cp if self.use_gpu else np
        
        # Auto-fallback to CPU for small systems
        self.force_threshold_for_gpu = 100  # Switch to GPU for >100 particles
        
        self.pme_parameters = pme_parameters if pme_parameters else {
            'fft_grid_spacing': 0.12, 'interpolation_order': 4, 'cutoff_distance': 1.0
        }
        self.cg_model_parameters = cg_model_parameters if cg_model_parameters else {
            'type': 'Martini3', 'solvent_model': 'polarizable_water', 'bead_definitions': {}
        }
        self.lj_epsilon = lj_epsilon
        self.lj_sigma = lj_sigma
        self.electrostatic_constant = electrostatic_constant
        self.cutoff = cutoff
        
        # Performance tracking
        self.forces_calculated = 0
        self.last_force_calculation_time = 0.0
        self.calculation_times = deque(maxlen=100)
        self.gpu_memory_usage = 0.0
        
        # Pre-allocated arrays for performance
        self._temp_forces = None
        self._temp_positions = None
        
        logging.info(f"🚀 MD Engine initialized: {'GPU' if self.use_gpu else 'CPU'} mode")
        logging.info(f"   GPU Available: {HAS_CUPY and cp.cuda.is_available() if HAS_CUPY else False}")
        logging.info(f"   Numba JIT: {HAS_NUMBA}")
        logging.info(f"   Force threshold for GPU: {self.force_threshold_for_gpu}")

    def calculate_forces(self, bacterium1, bacterium2) -> np.ndarray:
        """İki bakteri arasındaki toplam kuvveti hesaplar (HYBRID GPU/CPU)"""
        start_time = time.time()
        
        position1 = self.xp.asarray(bacterium1.biophysical.position)
        position2 = self.xp.asarray(bacterium2.biophysical.position)
        
        # Smart array allocation
        if self._temp_forces is None or self._temp_forces.shape != position1.shape:
            self._temp_forces = self.xp.zeros_like(position1)
        else:
            self._temp_forces.fill(0)

        # Calculate distance vector
        dist_vector = position1 - position2
        dist_sq = self.xp.sum(dist_vector**2)
        distance = self.xp.sqrt(dist_sq)

        if distance < self.cutoff and distance > 1e-9:
            # Lennard-Jones forces
            sigma_over_r = self.lj_sigma / distance
            sigma_over_r6 = sigma_over_r**6
            sigma_over_r12 = sigma_over_r6**2
            lj_force_magnitude = 24 * self.lj_epsilon / distance * (2 * sigma_over_r12 - sigma_over_r6)
            self._temp_forces += (lj_force_magnitude / distance) * dist_vector

            # Coulomb forces
            q1 = bacterium1.biophysical.charge
            q2 = bacterium2.biophysical.charge
            if q1 != 0 and q2 != 0:
                coulomb_force_magnitude = self.electrostatic_constant * q1 * q2 / dist_sq
                self._temp_forces += (coulomb_force_magnitude / distance) * dist_vector
        else:
            # Add small random force for numerical stability
            self._temp_forces += self.xp.random.randn(*position1.shape) * 0.0001

        result = cp.asnumpy(self._temp_forces) if self.use_gpu else self._temp_forces
        
        # Performance tracking
        calc_time = time.time() - start_time
        self.calculation_times.append(calc_time)
        self.forces_calculated += 1
        
        return result

    def update_system(self, bacteria_population: List) -> List:
        """MD sistemi güncelleme - HYBRID PERFORMANCE"""
        if not bacteria_population:
            return bacteria_population

        n_bacteria = len(bacteria_population)
        
        # Smart GPU/CPU switching based on population size
        current_use_gpu = self.use_gpu and (n_bacteria >= self.force_threshold_for_gpu)
        current_xp = cp if current_use_gpu else np
        
        if current_use_gpu != self.use_gpu:
            logging.info(f"Switching to {'GPU' if current_use_gpu else 'CPU'} for {n_bacteria} bacteria")
        
        try:
            # Batch force calculation for better performance
            if n_bacteria > 50:
                return self._update_system_batched(bacteria_population, current_xp, current_use_gpu)
            else:
                return self._update_system_simple(bacteria_population)
        except Exception as e:
            logging.error(f"MD system update error: {e}")
            return bacteria_population

    def _update_system_simple(self, bacteria_population: List) -> List:
        """Simple update for small populations"""
        for i, bacterium in enumerate(bacteria_population):
            net_force = np.zeros(3)
            
            # Calculate forces with nearby bacteria only
            for j, other_bacterium in enumerate(bacteria_population):
                if i != j:
                    distance = np.linalg.norm(
                        bacterium.biophysical.position - other_bacterium.biophysical.position
                    )
                    if distance < self.cutoff * 2:  # Only calculate for nearby bacteria
                        force = self.calculate_forces(bacterium, other_bacterium)
                        net_force += force
            
            # Apply force and thermostat
            bacterium.biophysical.apply_force(net_force)
            self._apply_thermostat(bacterium)
        
        return bacteria_population

    def _update_system_batched(self, bacteria_population: List, xp, use_gpu: bool) -> List:
        """Batched update for large populations using GPU"""
        n_bacteria = len(bacteria_population)
        
        # Extract positions for vectorized operations
        positions = xp.array([b.biophysical.position for b in bacteria_population])
        forces = xp.zeros_like(positions)
        
        # Vectorized distance calculations
        if use_gpu and HAS_CUPY:
            forces = self._calculate_forces_gpu_batched(positions, xp)
        else:
            forces = self._calculate_forces_cpu_batched(positions)
        
        # Apply forces back to bacteria
        forces_cpu = cp.asnumpy(forces) if use_gpu else forces
        for i, bacterium in enumerate(bacteria_population):
            bacterium.biophysical.apply_force(forces_cpu[i])
            self._apply_thermostat(bacterium)
        
        # Track GPU memory
        if use_gpu and HAS_CUPY:
            self.gpu_memory_usage = cp.cuda.memory_pool.used_bytes() / 1024**2
        
        return bacteria_population

    def _calculate_forces_gpu_batched(self, positions, xp):
        """GPU-optimized batch force calculation"""
        n_particles = positions.shape[0]
        forces = xp.zeros_like(positions)
        
        # Use advanced GPU kernels for large systems
        if n_particles > 200:
            return self._calculate_forces_advanced_gpu(positions, xp)
        
        # Standard pairwise calculation
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                r_vec = positions[i] - positions[j]
                r_sq = xp.sum(r_vec**2)
                r = xp.sqrt(r_sq)
                
                if r < self.cutoff and r > 1e-10:
                    # LJ + Coulomb forces
                    force = self._calculate_pairwise_force_gpu(r_vec, r, xp)
                    forces[i] += force
                    forces[j] -= force
        
        return forces

    def _calculate_forces_advanced_gpu(self, positions, xp):
        """Advanced GPU force calculation with optimizations"""
        # This could use CUDA kernels for maximum performance
        # For now, use optimized CuPy operations
        n_particles = positions.shape[0]
        forces = xp.zeros_like(positions)
        
        # Neighbor list optimization
        # (Implementation would go here for production)
        
        return self._calculate_forces_gpu_batched(positions, xp)

    def _calculate_pairwise_force_gpu(self, r_vec, r, xp):
        """Calculate pairwise force on GPU"""
        # Lennard-Jones
        sigma_over_r = self.lj_sigma / r
        sigma_over_r6 = sigma_over_r**6
        sigma_over_r12 = sigma_over_r6**2
        
        lj_force_magnitude = 24 * self.lj_epsilon / r * (2 * sigma_over_r12 - sigma_over_r6)
        force = (lj_force_magnitude / r) * r_vec
        
        return force

    def _calculate_forces_cpu_batched(self, positions):
        """CPU-optimized batch force calculation with Numba"""
        if HAS_NUMBA:
            return self._calculate_forces_cpu_numba(positions)
        else:
            return self._calculate_forces_cpu_numpy(positions)
    
    def _calculate_forces_cpu_numpy(self, positions):
        """NumPy-based force calculation"""
        n_particles = positions.shape[0]
        forces = np.zeros_like(positions)
        
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                r_vec = positions[i] - positions[j]
                r_sq = np.sum(r_vec**2)
                r = np.sqrt(r_sq)
                
                if r < self.cutoff and r > 1e-10:
                    force = self._calculate_pairwise_force_cpu(r_vec, r)
                    forces[i] += force
                    forces[j] -= force
        
        return forces
    
    def _calculate_forces_cpu_numba(self, positions):
        """Numba-optimized force calculation"""
        return self._calculate_forces_cpu_numba_impl(positions, self.lj_epsilon, self.lj_sigma, self.cutoff)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_forces_cpu_numba_impl(positions, lj_epsilon, lj_sigma, cutoff):
        """Numba implementation"""
        n_particles = positions.shape[0]
        forces = np.zeros_like(positions)
        
        for i in prange(n_particles):
            for j in range(i+1, n_particles):
                r_vec = positions[i] - positions[j]
                r_sq = np.sum(r_vec**2)
                r = np.sqrt(r_sq)
                
                if r < cutoff and r > 1e-10:
                    # LJ force calculation
                    sigma_over_r = lj_sigma / r
                    sigma_over_r6 = sigma_over_r**6
                    sigma_over_r12 = sigma_over_r6**2
                    
                    lj_force_magnitude = 24 * lj_epsilon / r * (2 * sigma_over_r12 - sigma_over_r6)
                    force = (lj_force_magnitude / r) * r_vec
                    
                    forces[i] += force
                    forces[j] -= force
        
        return forces
    
    def _calculate_pairwise_force_cpu(self, r_vec, r):
        """Calculate pairwise force on CPU"""
        # Lennard-Jones
        sigma_over_r = self.lj_sigma / r
        sigma_over_r6 = sigma_over_r**6
        sigma_over_r12 = sigma_over_r6**2
        
        lj_force_magnitude = 24 * self.lj_epsilon / r * (2 * sigma_over_r12 - sigma_over_r6)
        force = (lj_force_magnitude / r) * r_vec
        
        return force
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            'forces_calculated': self.forces_calculated,
            'using_gpu': self.use_gpu,
            'gpu_available': HAS_CUPY and cp.cuda.is_available() if HAS_CUPY else False,
            'numba_available': HAS_NUMBA,
            'force_threshold': self.force_threshold_for_gpu
        }
        
        if self.calculation_times:
            metrics['avg_calc_time'] = np.mean(self.calculation_times)
            metrics['calc_fps'] = 1.0 / np.mean(self.calculation_times) if np.mean(self.calculation_times) > 0 else 0
        
        if self.use_gpu and HAS_CUPY:
            metrics['gpu_memory_mb'] = self.gpu_memory_usage
        
        return metrics

    def _apply_thermostat(self, bacterium):
        """Sıcaklık kontrolü (Berendsen termostat)"""
        target_kinetic = 1.5 * BOLTZMANN_CONSTANT * self.temperature
        current_kinetic = bacterium.biophysical.calculate_kinetic_energy()
        
        if current_kinetic > 0:
            scaling_factor = np.sqrt(target_kinetic / current_kinetic)
            bacterium.biophysical.velocity *= scaling_factor * 0.1  # Yumuşak scaling

    def _apply_barostat(self, population: List):
        """Basınç kontrolü"""
        pass  # Şimdilik basit implementasyon

    def calculate_van_der_waals_forces(self, positions: np.ndarray, atom_types: Optional[np.ndarray]=None) -> np.ndarray:
        """Van der Waals kuvvetlerini hesapla"""
        if self.use_gpu:
            positions_gpu = cp.asarray(positions)
            forces = self._calculate_vdw_forces_gpu(positions_gpu)
            return cp.asnumpy(forces)
        else:
            return self._calculate_vdw_forces_cpu(positions)

    def _calculate_vdw_forces_gpu(self, positions) -> np.ndarray:
        """GPU-accelerated VdW force calculation"""
        n_atoms = positions.shape[0]
        forces = cp.zeros_like(positions)
        
        # CUDA kernel burada olabilir, şimdilik basit döngü
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_vec = positions[i] - positions[j]
                r_sq = cp.sum(r_vec**2)
                r = cp.sqrt(r_sq)
                
                if r < self.cutoff and r > 1e-10:
                    sigma_over_r = self.lj_sigma / r
                    sigma_over_r6 = sigma_over_r**6
                    sigma_over_r12 = sigma_over_r6**2
                    
                    force_magnitude = 24 * self.lj_epsilon / r * (2 * sigma_over_r12 - sigma_over_r6)
                    force_vec = (force_magnitude / r) * r_vec
                    
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        return forces

    def _calculate_vdw_forces_cpu(self, positions: np.ndarray) -> np.ndarray:
        """CPU VdW force calculation with Numba acceleration"""
        return self._vdw_forces_numba(positions, self.lj_epsilon, self.lj_sigma, self.cutoff)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _vdw_forces_numba(positions, epsilon, sigma, cutoff):
        """Numba-accelerated VdW force calculation"""
        n_atoms = positions.shape[0]
        forces = np.zeros_like(positions)
        
        for i in prange(n_atoms):
            for j in range(i+1, n_atoms):
                r_vec = positions[i] - positions[j]
                r_sq = np.sum(r_vec**2)
                r = np.sqrt(r_sq)
                
                if r < cutoff and r > 1e-10:
                    sigma_over_r = sigma / r
                    sigma_over_r6 = sigma_over_r**6
                    sigma_over_r12 = sigma_over_r6**2
                    
                    force_magnitude = 24 * epsilon / r * (2 * sigma_over_r12 - sigma_over_r6)
                    force_vec = (force_magnitude / r) * r_vec
                    
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        return forces

    def calculate_electrostatic_forces_pme(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """PME elektrostatik kuvvet hesabı (basitleştirilmiş)"""
        # Gerçek PME yerine basitleştirilmiş Coulomb
        return self._simplified_electrostatic_forces(positions, charges)

    def _simplified_electrostatic_forces(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """Basitleştirilmiş elektrostatik kuvvetler"""
        n_atoms = positions.shape[0]
        forces = np.zeros_like(positions)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if charges[i] == 0 or charges[j] == 0:
                    continue
                    
                r_vec = positions[i] - positions[j]
                r_sq = np.sum(r_vec**2)
                r = np.sqrt(r_sq)
                
                if r < self.cutoff and r > 1e-10:
                    force_magnitude = self.electrostatic_constant * charges[i] * charges[j] / r_sq
                    force_vec = (force_magnitude / r) * r_vec
                    
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        return forces
