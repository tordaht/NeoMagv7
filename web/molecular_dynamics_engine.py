# Moleküler Dinamik Motor - Gerçek Biyofiziksel Hesaplamalar
# Based on research: Moleküler Dinamik Algoritmalar ve Bakteriyel Sistemlerde Biyofiziksel Simülasyonlar

import numpy as np
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class AtomicPosition:
    x: float
    y: float 
    z: float
    mass: float = 1.0
    charge: float = 0.0

class MolecularDynamicsEngine:
    """
    Gerçek moleküler dinamik hesaplamalar için motor
    Van der Waals ve elektrostatik kuvvetleri hesaplar
    """
    
    def __init__(self, temperature=310.0, dt=0.001):
        self.temperature = temperature  # Kelvin (bakteriyel yaşam sıcaklığı)
        self.dt = dt  # Zaman adımı (pikosaniye)
        self.kB = 1.38064852e-23  # Boltzmann sabiti
        self.epsilon_0 = 8.854187817e-12  # Vakum dielektrik sabiti
        self.cutoff_distance = 12.0  # Angstrom
        
    def calculate_van_der_waals_force(self, pos1: AtomicPosition, pos2: AtomicPosition, 
                                    sigma=3.4, epsilon=0.995) -> Tuple[float, float, float]:
        """
        Van der Waals kuvvetlerini Lennard-Jones potansiyeli ile hesaplar
        F = 4*epsilon * [12*(sigma/r)^13 - 6*(sigma/r)^7] * (r_vec/r)
        """
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        dz = pos2.z - pos1.z
        
        r = math.sqrt(dx*dx + dy*dy + dz*dz)
        if r < 0.1 or r > self.cutoff_distance:
            return 0.0, 0.0, 0.0
            
        sr6 = (sigma/r)**6
        sr12 = sr6 * sr6
        
        force_magnitude = 4 * epsilon * (12*sr12 - 6*sr6) / r
        
        fx = force_magnitude * dx / r
        fy = force_magnitude * dy / r  
        fz = force_magnitude * dz / r
        
        return fx, fy, fz
    
    def calculate_electrostatic_force(self, pos1: AtomicPosition, pos2: AtomicPosition) -> Tuple[float, float, float]:
        """
        Elektrostatik kuvvetleri Coulomb yasası ile hesaplar
        PME (Particle Mesh Ewald) yöntemi yaklaşımı
        """
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        dz = pos2.z - pos1.z
        
        r = math.sqrt(dx*dx + dy*dy + dz*dz)
        if r < 0.1:
            return 0.0, 0.0, 0.0
            
        # Coulomb kuvveti: F = k * q1 * q2 / r^2
        k_coulomb = 1.0 / (4 * math.pi * self.epsilon_0)
        force_magnitude = k_coulomb * pos1.charge * pos2.charge / (r * r * r)
        
        fx = force_magnitude * dx
        fy = force_magnitude * dy
        fz = force_magnitude * dz
        
        return fx, fy, fz
    
    def calculate_forces(self, positions: List[AtomicPosition]) -> List[Tuple[float, float, float]]:
        """
        Tüm atomlar arası kuvvetleri hesaplar - alias for calculate_total_forces
        """
        return self.calculate_total_forces(positions)
    
    def calculate_total_forces(self, positions: List[AtomicPosition]) -> List[Tuple[float, float, float]]:
        """
        Tüm atomlar arası kuvvetleri hesaplar
        """
        n = len(positions)
        forces = [(0.0, 0.0, 0.0) for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                # Van der Waals kuvveti
                fx_vdw, fy_vdw, fz_vdw = self.calculate_van_der_waals_force(positions[i], positions[j])
                
                # Elektrostatik kuvvet
                fx_elec, fy_elec, fz_elec = self.calculate_electrostatic_force(positions[i], positions[j])
                
                # Toplam kuvvet
                fx_total = fx_vdw + fx_elec
                fy_total = fy_vdw + fy_elec
                fz_total = fz_vdw + fz_elec
                
                # Newton'un 3. yasası: i'ye etki eden kuvvet
                forces[i] = (forces[i][0] + fx_total, forces[i][1] + fy_total, forces[i][2] + fz_total)
                # j'ye etki eden kuvvet (zıt yönde)
                forces[j] = (forces[j][0] - fx_total, forces[j][1] - fy_total, forces[j][2] - fz_total)
        
        return forces
    
    def update_positions_verlet(self, positions: List[AtomicPosition], 
                              velocities: List[Tuple[float, float, float]],
                              forces: List[Tuple[float, float, float]]) -> Tuple[List[AtomicPosition], List[Tuple[float, float, float]]]:
        """
        Verlet algoritması ile pozisyonları günceller
        """
        new_positions = []
        new_velocities = []
        
        for i, pos in enumerate(positions):
            # Hızlanma hesabı: a = F/m
            ax = forces[i][0] / pos.mass
            ay = forces[i][1] / pos.mass
            az = forces[i][2] / pos.mass
            
            # Yeni hız: v(t+dt) = v(t) + a*dt
            vx_new = velocities[i][0] + ax * self.dt
            vy_new = velocities[i][1] + ay * self.dt
            vz_new = velocities[i][2] + az * self.dt
            
            # Yeni pozisyon: r(t+dt) = r(t) + v*dt + 0.5*a*dt^2
            x_new = pos.x + velocities[i][0] * self.dt + 0.5 * ax * self.dt * self.dt
            y_new = pos.y + velocities[i][1] * self.dt + 0.5 * ay * self.dt * self.dt
            z_new = pos.z + velocities[i][2] * self.dt + 0.5 * az * self.dt * self.dt
            
            new_positions.append(AtomicPosition(x_new, y_new, z_new, pos.mass, pos.charge))
            new_velocities.append((vx_new, vy_new, vz_new))
        
        return new_positions, new_velocities
    
    def calculate_kinetic_energy(self, velocities: List[Tuple[float, float, float]], 
                               masses: List[float]) -> float:
        """
        Kinetik enerji hesabı: KE = 0.5 * m * v^2
        """
        total_ke = 0.0
        for i, vel in enumerate(velocities):
            v_squared = vel[0]**2 + vel[1]**2 + vel[2]**2
            total_ke += 0.5 * masses[i] * v_squared
        return total_ke
    
    def calculate_potential_energy(self, positions: List[AtomicPosition]) -> float:
        """
        Potansiyel enerji hesabı (Van der Waals + Elektrostatik)
        """
        total_pe = 0.0
        n = len(positions)
        
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[j].x - positions[i].x
                dy = positions[j].y - positions[i].y
                dz = positions[j].z - positions[i].z
                r = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                if r < self.cutoff_distance:
                    # Lennard-Jones potansiyeli
                    sigma = 3.4
                    epsilon = 0.995
                    sr6 = (sigma/r)**6
                    lj_potential = 4 * epsilon * (sr6*sr6 - sr6)
                    
                    # Coulomb potansiyeli
                    k_coulomb = 1.0 / (4 * math.pi * self.epsilon_0)
                    coulomb_potential = k_coulomb * positions[i].charge * positions[j].charge / r
                    
                    total_pe += lj_potential + coulomb_potential
        
        return total_pe

# Test fonksiyonu
def test_molecular_dynamics():
    """
    Moleküler dinamik motorunu test eder
    """
    engine = MolecularDynamicsEngine()
    
    # Basit 2-atom sistemi
    positions = [
        AtomicPosition(0.0, 0.0, 0.0, mass=12.0, charge=0.1),
        AtomicPosition(4.0, 0.0, 0.0, mass=16.0, charge=-0.1)
    ]
    
    velocities = [(0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)]
    
    print("Moleküler Dinamik Test:")
    print(f"Başlangıç pozisyonları: {[(p.x, p.y, p.z) for p in positions]}")
    
    # 10 adım simülasyon
    for step in range(10):
        forces = engine.calculate_total_forces(positions)
        positions, velocities = engine.update_positions_verlet(positions, velocities, forces)
        
        ke = engine.calculate_kinetic_energy(velocities, [p.mass for p in positions])
        pe = engine.calculate_potential_energy(positions)
        
        if step % 5 == 0:
            print(f"Adım {step}: KE={ke:.4f}, PE={pe:.4f}, Total={ke+pe:.4f}")
    
    print(f"Final pozisyonlar: {[(p.x, p.y, p.z) for p in positions]}")

if __name__ == "__main__":
    test_molecular_dynamics() 