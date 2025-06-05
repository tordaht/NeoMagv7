"""NeoMag V7 - Biophysical Properties Core Module"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

class MolecularForceType(Enum):
    """Moleküler kuvvet türleri"""
    VAN_DER_WAALS = "van_der_waals"
    ELECTROSTATIC = "electrostatic"
    HYDROGEN_BOND = "hydrogen_bond"
    COVALENT = "covalent"
    HYDROPHOBIC = "hydrophobic"

@dataclass
class BiophysicalProperties:
    """Biyofiziksel özellikler"""
    membrane_potential: float = -70.0  # mV
    ion_concentrations: Dict[str, float] = field(default_factory=lambda: {
        'Na+': 145.0,   # mM extracellular
        'K+': 5.0,      # mM extracellular
        'Ca2+': 2.5,    # mM extracellular
        'Cl-': 110.0,   # mM extracellular
        'ATP': 5.0,     # mM intracellular
        'ADP': 0.5      # mM intracellular
    })
    ph_gradient: float = 7.4
    osmotic_pressure: float = 300.0  # mOsm/kg
    surface_tension: float = 0.072   # N/m
    viscosity: float = 1.0e-3        # Pa·s
    diffusion_coefficients: Dict[str, float] = field(default_factory=lambda: {
        'glucose': 6.7e-10,    # m²/s
        'oxygen': 2.1e-9,      # m²/s
        'co2': 1.9e-9,         # m²/s
        'water': 2.3e-9        # m²/s
    })
    mass: float = 1.0 # atomik kütle birimi (amu) veya CG bead kütlesi
    charge: float = 0.0 # temel yük (e)
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # Angstrom veya nm
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # Angstrom/ps veya nm/ps
    force: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # kJ/mol/nm veya kcal/mol/Angstrom
    hydrophobicity: float = 0.0 # Hidrofobisite ölçeği (örn: Kyte-Doolittle)
    binding_affinity: Dict[str, float] = field(default_factory=dict) # Diğer molekül türlerine bağlanma afinitesi
    cg_bead_type: Optional[str] = None
    cg_interaction_params: Dict[str, Any] = field(default_factory=dict)
    current_forces: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # Anlık kuvvetler

    def __post_init__(self):
        """BiophysicalProperties post-initialization"""
        # NumPy array'lerin dtype'ını kontrol et
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float64)
        if not isinstance(self.force, np.ndarray):
            self.force = np.array(self.force, dtype=np.float64)
        if not isinstance(self.current_forces, np.ndarray):
            self.current_forces = np.array(self.current_forces, dtype=np.float64)

    def update_position(self, dt: float):
        """Pozisyonu güncelle"""
        self.position += self.velocity * dt

    def update_velocity(self, acceleration: np.ndarray, dt: float):
        """Hızı güncelle"""
        self.velocity += acceleration * dt

    def apply_force(self, force: np.ndarray):
        """Kuvvet uygula"""
        self.current_forces += force

    def reset_forces(self):
        """Kuvvetleri sıfırla"""
        self.current_forces = np.zeros(3)

    def calculate_kinetic_energy(self) -> float:
        """Kinetik enerji hesapla"""
        return 0.5 * self.mass * np.sum(self.velocity ** 2)

    def get_state_vector(self) -> np.ndarray:
        """Durum vektörü al (ML/AI için)"""
        state = np.concatenate([
            self.position,
            self.velocity,
            self.current_forces,
            [self.membrane_potential, self.ph_gradient, self.osmotic_pressure,
             self.surface_tension, self.viscosity, self.mass, self.charge, 
             self.hydrophobicity]
        ])
        return state

    def get_tabpfn_features(self) -> np.ndarray:
        """TabPFN için optimize edilmiş feature vektörü"""
        features = []
        
        # Pozisyon ve hız bilgileri
        features.extend(self.position.tolist())
        features.extend(self.velocity.tolist())
        
        # Temel biyofiziksel özellikler
        features.extend([
            self.membrane_potential,
            self.ph_gradient, 
            self.osmotic_pressure,
            self.mass,
            self.charge,
            self.hydrophobicity
        ])
        
        # İyon konsantrasyonları (seçili olanlar)
        features.extend([
            self.ion_concentrations.get('Na+', 0.0),
            self.ion_concentrations.get('K+', 0.0),
            self.ion_concentrations.get('ATP', 0.0)
        ])
        
        # Diffusion coefficients (seçili olanlar)
        features.extend([
            self.diffusion_coefficients.get('glucose', 0.0),
            self.diffusion_coefficients.get('oxygen', 0.0)
        ])
        
        return np.array(features, dtype=np.float32)
