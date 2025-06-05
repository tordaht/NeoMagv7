# Reinforcement Learning Ecosystem Intervention Engine
# Based on research: Advancements in Artificial Intelligence for Ecological Systems

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

class ActionType(Enum):
    ADD_NUTRIENTS = "add_nutrients"
    REMOVE_TOXINS = "remove_toxins"
    ADJUST_PH = "adjust_ph"
    CONTROL_TEMPERATURE = "control_temperature"
    ADD_ANTIBIOTICS = "add_antibiotics"
    MODIFY_OXYGEN = "modify_oxygen"

@dataclass
class EnvironmentState:
    nutrient_level: float
    toxin_level: float
    ph_level: float
    temperature: float
    oxygen_level: float
    bacteria_count: int
    diversity_index: float
    time_step: int

@dataclass
class Action:
    action_type: ActionType
    intensity: float  # 0.0 to 1.0
    duration: int    # time steps

@dataclass
class Reward:
    bacteria_survival: float
    diversity_maintenance: float
    ecosystem_stability: float
    resource_efficiency: float
    total_reward: float

class EcosystemState:
    """
    Bakteriyel ecosystem state representation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Initialize ecosystem to default state"""
        self.state = EnvironmentState(
            nutrient_level=0.5,
            toxin_level=0.1,
            ph_level=7.0,
            temperature=37.0,
            oxygen_level=0.6,
            bacteria_count=100,
            diversity_index=0.8,
            time_step=0
        )
        return self.get_state_vector()
    
    def get_state_vector(self) -> np.ndarray:
        """Convert state to numerical vector for RL agent"""
        return np.array([
            self.state.nutrient_level,
            self.state.toxin_level,
            (self.state.ph_level - 7.0) / 7.0,  # Normalize pH
            (self.state.temperature - 37.0) / 50.0,  # Normalize temp
            self.state.oxygen_level,
            self.state.bacteria_count / 1000.0,  # Normalize count
            self.state.diversity_index
        ])
    
    def apply_action(self, action: Action) -> Tuple[np.ndarray, Reward, bool]:
        """
        Apply action to ecosystem and return new state, reward, done
        """
        # Apply action effects
        if action.action_type == ActionType.ADD_NUTRIENTS:
            self.state.nutrient_level += action.intensity * 0.2
            self.state.nutrient_level = min(1.0, self.state.nutrient_level)
            
        elif action.action_type == ActionType.REMOVE_TOXINS:
            self.state.toxin_level -= action.intensity * 0.3
            self.state.toxin_level = max(0.0, self.state.toxin_level)
            
        elif action.action_type == ActionType.ADJUST_PH:
            ph_change = (action.intensity - 0.5) * 2.0  # -1 to +1
            self.state.ph_level += ph_change
            self.state.ph_level = np.clip(self.state.ph_level, 4.0, 10.0)
            
        elif action.action_type == ActionType.CONTROL_TEMPERATURE:
            temp_change = (action.intensity - 0.5) * 20.0  # -10 to +10
            self.state.temperature += temp_change
            self.state.temperature = np.clip(self.state.temperature, 20.0, 50.0)
            
        elif action.action_type == ActionType.ADD_ANTIBIOTICS:
            self.state.toxin_level += action.intensity * 0.1  # Antibiotics as mild toxin
            
        elif action.action_type == ActionType.MODIFY_OXYGEN:
            oxygen_change = (action.intensity - 0.5) * 0.4
            self.state.oxygen_level += oxygen_change
            self.state.oxygen_level = np.clip(self.state.oxygen_level, 0.0, 1.0)
        
        # Update ecosystem dynamics
        self._update_ecosystem_dynamics()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        self.state.time_step += 1
        
        return self.get_state_vector(), reward, done
    
    def _update_ecosystem_dynamics(self):
        """
        Update ecosystem based on current conditions
        """
        # Bacteria growth based on conditions
        growth_factor = 1.0
        
        # Nutrient effect
        if self.state.nutrient_level > 0.3:
            growth_factor *= (1.0 + self.state.nutrient_level)
        else:
            growth_factor *= 0.8
        
        # Toxin effect
        growth_factor *= (1.0 - self.state.toxin_level)
        
        # pH effect (optimal around 7.0)
        ph_factor = 1.0 - abs(self.state.ph_level - 7.0) / 7.0
        growth_factor *= ph_factor
        
        # Temperature effect (optimal around 37°C)
        temp_factor = 1.0 - abs(self.state.temperature - 37.0) / 50.0
        growth_factor *= temp_factor
        
        # Oxygen effect
        growth_factor *= self.state.oxygen_level
        
        # Update bacteria count
        self.state.bacteria_count = int(self.state.bacteria_count * growth_factor)
        self.state.bacteria_count = max(1, min(2000, self.state.bacteria_count))
        
        # Update diversity (affected by extreme conditions)
        if growth_factor < 0.5:
            self.state.diversity_index *= 0.98
        elif growth_factor > 1.2:
            self.state.diversity_index *= 1.01
        
        self.state.diversity_index = np.clip(self.state.diversity_index, 0.1, 1.0)
        
        # Natural processes
        self.state.toxin_level += random.uniform(-0.01, 0.02)  # Random toxin accumulation
        self.state.toxin_level = max(0.0, self.state.toxin_level)
        
        self.state.nutrient_level -= 0.01  # Nutrient consumption
        self.state.nutrient_level = max(0.0, self.state.nutrient_level)
    
    def _calculate_reward(self) -> Reward:
        """
        Calculate multi-objective reward for RL agent
        """
        # Bacteria survival reward
        bacteria_reward = min(self.state.bacteria_count / 500.0, 1.0)
        
        # Diversity maintenance reward
        diversity_reward = self.state.diversity_index
        
        # Ecosystem stability reward (penalize extreme values)
        stability_penalty = 0.0
        if self.state.toxin_level > 0.5:
            stability_penalty += (self.state.toxin_level - 0.5)
        if abs(self.state.ph_level - 7.0) > 2.0:
            stability_penalty += abs(self.state.ph_level - 7.0) / 10.0
        if abs(self.state.temperature - 37.0) > 10.0:
            stability_penalty += abs(self.state.temperature - 37.0) / 50.0
            
        stability_reward = max(0.0, 1.0 - stability_penalty)
        
        # Resource efficiency (penalize excessive interventions)
        resource_reward = 1.0  # Can be modified based on action cost
        
        # Combined reward
        total_reward = (
            0.4 * bacteria_reward +
            0.3 * diversity_reward +
            0.2 * stability_reward +
            0.1 * resource_reward
        )
        
        return Reward(
            bacteria_survival=bacteria_reward,
            diversity_maintenance=diversity_reward,
            ecosystem_stability=stability_reward,
            resource_efficiency=resource_reward,
            total_reward=total_reward
        )
    
    def _is_episode_done(self) -> bool:
        """
        Check if episode should terminate
        """
        # Episode ends if ecosystem collapses or reaches max time
        if self.state.bacteria_count < 10:
            return True
        if self.state.diversity_index < 0.2:
            return True
        if self.state.time_step >= 200:  # Max episode length
            return True
        return False

class DQNAgent:
    """
    Deep Q-Network agent for ecosystem management
    Simplified version for demonstration
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.memory_size = 10000
        
        # Q-table approximation (in real implementation, use neural network)
        self.q_table = {}
        
    def discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretize continuous state for Q-table
        """
        discretized = []
        for i, value in enumerate(state):
            discrete_value = int(value * 10) / 10.0  # 0.1 precision
            discretized.append(discrete_value)
        return tuple(discretized)
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """
        Store experience in memory
        """
        experience = (state, action, reward, next_state, done)
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(experience)
    
    def replay(self, batch_size: int = 32):
        """
        Train the agent on a batch of experiences
        """
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.discretize_state(state)
            next_state_key = self.discretize_state(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            target = reward
            if not done:
                target += 0.95 * np.amax(self.q_table[next_state_key])
            
            self.q_table[state_key][action] = target
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class EcosystemManager:
    """
    High-level ecosystem management using RL
    """
    
    def __init__(self):
        self.ecosystem = EcosystemState()
        self.agent = DQNAgent(state_size=7, action_size=len(ActionType))
        self.action_map = list(ActionType)
        
    def action_to_ecosystem_action(self, action_index: int) -> Action:
        """
        Convert agent action to ecosystem action
        """
        action_type = self.action_map[action_index]
        intensity = random.uniform(0.3, 0.8)  # Random intensity
        duration = 1  # Single time step
        
        return Action(action_type, intensity, duration)
    
    def train_agent(self, episodes: int = 100):
        """
        Train RL agent to manage ecosystem
        """
        scores = []
        
        for episode in range(episodes):
            state = self.ecosystem.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 200:
                # Agent chooses action
                action_index = self.agent.act(state)
                ecosystem_action = self.action_to_ecosystem_action(action_index)
                
                # Apply action to ecosystem
                next_state, reward_obj, done = self.ecosystem.apply_action(ecosystem_action)
                
                # Store experience and train
                self.agent.remember(state, action_index, reward_obj.total_reward, 
                                  next_state, done)
                self.agent.replay()
                
                state = next_state
                total_reward += reward_obj.total_reward
                steps += 1
            
            scores.append(total_reward)
            
            if episode % 20 == 0:
                avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
                print(f"Episode {episode}, Average Score: {avg_score:.3f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
        
        return scores
    
    def evaluate_agent(self, episodes: int = 10):
        """
        Evaluate trained agent performance
        """
        self.agent.epsilon = 0.0  # No exploration during evaluation
        
        evaluation_results = []
        
        for episode in range(episodes):
            state = self.ecosystem.reset()
            episode_rewards = []
            episode_actions = []
            done = False
            steps = 0
            
            while not done and steps < 200:
                action_index = self.agent.act(state)
                ecosystem_action = self.action_to_ecosystem_action(action_index)
                
                state, reward_obj, done = self.ecosystem.apply_action(ecosystem_action)
                
                episode_rewards.append(reward_obj.total_reward)
                episode_actions.append(ecosystem_action.action_type.value)
                steps += 1
            
            evaluation_results.append({
                'total_reward': sum(episode_rewards),
                'episode_length': steps,
                'final_bacteria_count': self.ecosystem.state.bacteria_count,
                'final_diversity': self.ecosystem.state.diversity_index,
                'actions_taken': episode_actions
            })
        
        return evaluation_results

def test_reinforcement_learning():
    """
    Test RL ecosystem management
    """
    print("=== Reinforcement Learning Ecosystem Test ===")
    
    manager = EcosystemManager()
    
    print("Training agent...")
    scores = manager.train_agent(episodes=50)
    
    print(f"\nTraining completed. Final average score: {np.mean(scores[-10:]):.3f}")
    
    print("\nEvaluating agent...")
    results = manager.evaluate_agent(episodes=5)
    
    for i, result in enumerate(results):
        print(f"Episode {i+1}: Reward={result['total_reward']:.3f}, "
              f"Length={result['episode_length']}, "
              f"Bacteria={result['final_bacteria_count']}, "
              f"Diversity={result['final_diversity']:.3f}")
    
    avg_reward = np.mean([r['total_reward'] for r in results])
    print(f"\nAverage evaluation reward: {avg_reward:.3f}")

if __name__ == "__main__":
    test_reinforcement_learning() 