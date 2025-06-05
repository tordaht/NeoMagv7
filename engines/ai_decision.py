"""NeoMag V7 - AI Decision Engine with Advanced RL"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import sys
sys.path.append('..')

class ExperienceReplayBuffer:
    """Deep RL için deneyim replay buffer'ı - OPTIMIZED"""
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def push(self, state, action, reward, next_state, done):
        """Deneyimi buffer'a ekle"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Rastgele batch örnekle - TENSOR OPTIMIZATION"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # OPTIMIZED: Convert to numpy first, then tensor
        states_np = np.array([e[0] for e in batch], dtype=np.float32)
        actions_np = np.array([e[1] for e in batch], dtype=np.int64)
        rewards_np = np.array([e[2] for e in batch], dtype=np.float32)
        next_states_np = np.array([e[3] for e in batch], dtype=np.float32)
        dones_np = np.array([e[4] for e in batch], dtype=bool)
        
        # Convert to tensors efficiently
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    """Policy Gradient için neural network - GPU OPTIMIZED"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights for better convergence
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for better training"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class ActorNetwork(nn.Module):
    """Actor-Critic için Actor network - OPTIMIZED"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class CriticNetwork(nn.Module):
    """Actor-Critic için Critic network - OPTIMIZED"""
    def __init__(self, state_size: int, hidden_size: int = 128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class MultiAgentCoordinator:
    """MARL için multi-agent koordinasyon sistemi"""
    def __init__(self, num_agents: int, communication_enabled: bool = True):
        self.num_agents = num_agents
        self.communication_enabled = communication_enabled
        self.agent_messages = {}
        self.global_reward_history = deque(maxlen=1000)
        
    def coordinate_agents(self, agent_states: Dict[int, np.ndarray], agent_rewards: Dict[int, float]):
        """Ajanları koordine et"""
        if not self.communication_enabled:
            return {}
            
        coordination_signals = {}
        for agent_id, state in agent_states.items():
            neighbor_states = [s for aid, s in agent_states.items() if aid != agent_id]
            if neighbor_states:
                avg_neighbor_state = np.mean(neighbor_states, axis=0)
                coordination_signals[agent_id] = avg_neighbor_state
            else:
                coordination_signals[agent_id] = np.zeros_like(state)
                
        return coordination_signals
    
    def update_global_reward(self, total_reward: float):
        """Global ödülü güncelle"""
        self.global_reward_history.append(total_reward)
    
    def get_cooperation_bonus(self, agent_id: int, individual_reward: float) -> float:
        """İşbirliği bonusu hesapla"""
        if len(self.global_reward_history) < 10:
            return 0.0
            
        recent_global_avg = np.mean(list(self.global_reward_history)[-10:])
        cooperation_bonus = 0.1 * recent_global_avg
        return cooperation_bonus

class AIDecisionEngine:
    """AI-driven decision making with advanced RL algorithms - PERFORMANCE OPTIMIZED"""
    def __init__(self, learning_rate=0.001, discount_factor=0.99, exploration_rate=0.1,
                 model_type='DQN', use_gpu=True, rl_algorithm_params=None,
                 state_size=15, action_size=6, enable_marl=True):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.initial_exploration_rate = exploration_rate
        self.model_type = model_type
        # Force CPU for stability initially
        self.use_gpu = False
        if use_gpu:
            try:
                if torch.cuda.is_available():
                    self.use_gpu = True
                    logging.info("🚀 AI using GPU")
                else:
                    logging.info("⚠️ CUDA not available, AI using CPU")
            except Exception as e:
                logging.info(f"⚠️ GPU error, AI using CPU: {e}")
                self.use_gpu = False
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.rl_algorithm_params = rl_algorithm_params if rl_algorithm_params else {}
        self.state_size = state_size
        self.action_size = action_size
        self.enable_marl = enable_marl

        # Base RL Components
        self.q_table = {}
        self.dqn_model = None
        self.target_dqn_model = None
        self.policy_model = None
        self.optimizer = None
        self.loss_fn = None
        
        # Advanced RL Components
        self.experience_buffer = ExperienceReplayBuffer(capacity=50000)
        self.actor_network = None
        self.critic_network = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        
        # MARL Components
        self.multi_agent_coordinator = MultiAgentCoordinator(num_agents=100) if enable_marl else None
        
        # Training metrics
        self.training_step = 0
        self.target_update_frequency = 100
        self.batch_size = 32
        self.policy_log_probs = []
        self.rewards_history = deque(maxlen=1000)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Performance tracking
        self.update_times = deque(maxlen=100)
        self.gpu_memory_usage = 0.0

        self._initialize_model()
        logging.info(f"AI Decision Engine initialized: {self.model_type}, Device: {self.device}")

    def _initialize_model(self):
        """RL modelini başlat - OPTIMIZED"""
        if self.model_type == 'SimpleQ':
            self.q_table = {}
            
        elif self.model_type == 'DQN':
            self.dqn_model = nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.action_size)
            ).to(self.device)
            
            self.target_dqn_model = nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.action_size)
            ).to(self.device)
            
            # Copy weights to target network
            self.target_dqn_model.load_state_dict(self.dqn_model.state_dict())
            
            self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            self.loss_fn = nn.MSELoss()
            
        elif self.model_type == 'DDQN':
            self._initialize_ddqn()
            
        elif self.model_type == 'A3C' or self.model_type == 'Actor-Critic':
            self._initialize_a3c()
            
        elif self.model_type == 'Policy Gradient':
            self.policy_model = PolicyNetwork(self.state_size, self.action_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
            
        else:
            logging.warning(f"Desteklenmeyen model tipi: {self.model_type}")
            self.model_type = 'DQN'
            self._initialize_model()

    def _initialize_ddqn(self):
        """Double DQN initialization"""
        self.dqn_model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.action_size)
        )
        
        self.target_dqn_model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.action_size)
        )
        
        if self.use_gpu:
            self.dqn_model.cuda()
            self.target_dqn_model.cuda()
            
        self.target_dqn_model.load_state_dict(self.dqn_model.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()

    def _initialize_a3c(self):
        """A3C initialization"""
        self.actor_network = ActorNetwork(self.state_size, self.action_size, hidden_size=256)
        self.critic_network = CriticNetwork(self.state_size, hidden_size=256)
        
        if self.use_gpu:
            self.actor_network.cuda()
            self.critic_network.cuda()
            
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

    def select_action(self, state_representation: np.ndarray, possible_actions: List[str], agent_id: Optional[int] = None) -> str:
        """Action selection with epsilon-greedy exploration"""
        try:
            # Epsilon decay
            if self.training_step > 0:
                self.exploration_rate = max(self.epsilon_min, 
                                          self.initial_exploration_rate * (self.epsilon_decay ** self.training_step))
            
            if self.model_type == 'SimpleQ':
                return self._select_action_simple_q(state_representation, possible_actions)
            elif self.model_type == 'DQN' or self.model_type == 'DDQN':
                return self._select_action_dqn(state_representation, possible_actions)
            elif self.model_type == 'PolicyGradient':
                return self._select_action_policy_gradient(state_representation, possible_actions)
            elif self.model_type == 'ActorCritic' or self.model_type == 'A3C':
                return self._select_action_actor_critic(state_representation, possible_actions)
            else:
                return np.random.choice(possible_actions)
                
        except Exception as e:
            logging.error(f"Action selection error: {e}")
            return np.random.choice(possible_actions) if possible_actions else "wait"

    def _select_action_simple_q(self, state_representation: np.ndarray, possible_actions: List[str]) -> str:
        """Simple Q-Learning action selection"""
        state_key = tuple(np.round(state_representation, 2))
        
        if np.random.random() < self.exploration_rate:
            return np.random.choice(possible_actions)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in possible_actions}
        
        q_values = self.q_table[state_key]
        best_action = max(q_values, key=q_values.get)
        return best_action

    def _select_action_dqn(self, state_representation: np.ndarray, possible_actions: List[str]) -> str:
        """DQN/DDQN action selection"""
        if np.random.random() < self.exploration_rate:
            return np.random.choice(possible_actions)
        
        state_tensor = torch.FloatTensor(state_representation).unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            self.dqn_model.eval()  # Set to evaluation mode
            q_values = self.dqn_model(state_tensor)
            action_idx = q_values.argmax().item()
            
        if action_idx < len(possible_actions):
            return possible_actions[action_idx]
        else:
            return np.random.choice(possible_actions)

    def _select_action_policy_gradient(self, state_representation: np.ndarray, possible_actions: List[str]) -> str:
        """Policy Gradient action selection"""
        state_tensor = torch.FloatTensor(state_representation).unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            self.policy_model.eval()
            action_probs = self.policy_model(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample().item()
            
        if action_idx < len(possible_actions):
            return possible_actions[action_idx]
        else:
            return np.random.choice(possible_actions)

    def _select_action_actor_critic(self, state_representation: np.ndarray, possible_actions: List[str]) -> str:
        """Actor-Critic action selection"""
        state_tensor = torch.FloatTensor(state_representation).unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            self.actor_network.eval()
            action_probs = self.actor_network(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample().item()
            
        if action_idx < len(possible_actions):
            return possible_actions[action_idx]
        else:
            return np.random.choice(possible_actions)

    def update_model(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, 
                    done: bool, possible_actions: List[str], agent_id: Optional[int] = None):
        """Model güncelleme"""
        try:
            if self.model_type == 'SimpleQ':
                self._update_simple_q(state, action, reward, next_state, done, possible_actions)
            elif self.model_type == 'DQN':
                self._update_dqn(state, action, reward, next_state, done, possible_actions)
            elif self.model_type == 'DDQN':
                self._update_ddqn(state, action, reward, next_state, done, possible_actions)
            elif self.model_type == 'PolicyGradient':
                self._update_policy_gradient(state, action, reward, next_state, done, possible_actions)
            elif self.model_type == 'ActorCritic' or self.model_type == 'A3C':
                self._update_actor_critic(state, action, reward, next_state, done, possible_actions)
                
            self.training_step += 1
            self.rewards_history.append(reward)
            
            # Update target network periodically
            if self.training_step % self.target_update_frequency == 0:
                self._update_target_network()
                
        except Exception as e:
            logging.error(f"Model update error: {e}")

    def _update_simple_q(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, 
                        done: bool, possible_actions: List[str]):
        """Simple Q-Learning update"""
        state_key = tuple(np.round(state, 2))
        next_state_key = tuple(np.round(next_state, 2))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in possible_actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in possible_actions}
            
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values()) if not done else 0.0
        
        target_q = reward + self.discount_factor * max_next_q
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)

    def _update_dqn(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, 
                   done: bool, possible_actions: List[str]):
        """DQN update - PERFORMANCE OPTIMIZED"""
        start_time = time.time()
        
        action_idx = possible_actions.index(action) if action in possible_actions else 0
        self.experience_buffer.push(state, action_idx, reward, next_state, done)
        
        if len(self.experience_buffer) < self.batch_size:
            return
            
        batch = self.experience_buffer.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = batch
        
        try:
            self.dqn_model.train()
            
            # Forward pass with gradient clipping
            current_q_values = self.dqn_model(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            # Target calculation with no_grad for efficiency
            with torch.no_grad():
                next_q_values = self.target_dqn_model(next_states).max(1)[0]
                target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
            
            loss = self.loss_fn(current_q_values, target_q_values)
            
            # Optimized backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dqn_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track GPU memory
            if self.use_gpu:
                self.gpu_memory_usage = torch.cuda.memory_allocated() / 1024**2
                
        except Exception as e:
            logging.error(f"DQN update error: {e}")
        
        # Track update time
        update_time = time.time() - start_time
        self.update_times.append(update_time)

    def _update_ddqn(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, 
                    done: bool, possible_actions: List[str]):
        """Double DQN update"""
        action_idx = possible_actions.index(action) if action in possible_actions else 0
        self.experience_buffer.push(state, action_idx, reward, next_state, done)
        
        if len(self.experience_buffer) < self.batch_size:
            return
            
        batch = self.experience_buffer.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = batch
        if self.use_gpu:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()
            
        current_q_values = self.dqn_model(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select action, target network to evaluate
        next_actions = self.dqn_model(next_states).max(1)[1].detach()
        next_q_values = self.target_dqn_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _update_policy_gradient(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, 
                               done: bool, possible_actions: List[str]):
        """Policy Gradient update"""
        # Store experience for episode update
        action_idx = possible_actions.index(action) if action in possible_actions else 0
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if self.use_gpu:
            state_tensor = state_tensor.cuda()
            
        action_probs = self.policy_model(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(torch.tensor(action_idx))
        
        self.policy_log_probs.append((log_prob, reward))

    def _update_actor_critic(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, 
                            done: bool, possible_actions: List[str]):
        """Actor-Critic update"""
        action_idx = possible_actions.index(action) if action in possible_actions else 0
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        if self.use_gpu:
            state_tensor = state_tensor.cuda()
            next_state_tensor = next_state_tensor.cuda()
            
        # Critic update
        current_value = self.critic_network(state_tensor)
        next_value = self.critic_network(next_state_tensor) if not done else torch.tensor([[0.0]])
        if self.use_gpu and not done:
            next_value = next_value.cuda()
            
        target_value = reward + self.discount_factor * next_value.item()
        critic_loss = F.mse_loss(current_value, torch.tensor([[target_value]]).cuda() if self.use_gpu else torch.tensor([[target_value]]))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        action_probs = self.actor_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(torch.tensor(action_idx))
        
        advantage = target_value - current_value.item()
        actor_loss = -log_prob * advantage
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def _update_target_network(self):
        """Target network güncelleme"""
        if self.target_dqn_model is not None:
            self.target_dqn_model.load_state_dict(self.dqn_model.state_dict())
    
    def _soft_update_target_network(self, tau=0.005):
        """Soft target network update for stability"""
        if self.target_dqn_model is not None:
            for target_param, local_param in zip(self.target_dqn_model.parameters(), self.dqn_model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_possible_actions(self) -> List[str]:
        """Mümkün eylemleri döndür"""
        return ["move_up", "move_down", "move_left", "move_right", "consume", "wait"]

    def get_performance_metrics(self) -> Dict[str, float]:
        """Performans metriklerini döndür - ENHANCED"""
        if not self.rewards_history:
            return {}
        
        metrics = {
            'avg_reward': np.mean(self.rewards_history),
            'total_training_steps': self.training_step,
            'current_exploration_rate': self.exploration_rate,
            'experience_buffer_size': len(self.experience_buffer),
            'device': str(self.device)
        }
        
        # Performance metrics
        if self.update_times:
            metrics['avg_update_time'] = np.mean(self.update_times)
            metrics['update_fps'] = 1.0 / np.mean(self.update_times) if np.mean(self.update_times) > 0 else 0
        
        # GPU metrics
        if self.use_gpu:
            metrics['gpu_memory_mb'] = self.gpu_memory_usage
            metrics['gpu_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                metrics['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**2
        
        return metrics
