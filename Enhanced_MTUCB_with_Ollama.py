"""
增强版 MTUCB + Ollama LLM 集成系统
全面优化SC-QOS（语义通信服务质量），支持6个参数动态调整

新增特性：
1. 6参数全面优化：alpha, zeta, omega, compression_ratio, power_ratio, min_phi
2. 语义通信增强：语义压缩、功率分配、多级QOS
3. 智能网络状态感知：拥塞预测、负载均衡、切换优化
4. 丰富的性能指标：语义效率、能耗效率、用户体验质量
5. 多维度可视化：热力图、雷达图、趋势分析
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from typing import List, Tuple, Dict, Optional
import time
from tqdm import tqdm
from dataclasses import dataclass, asdict
import pandas as pd

# 导入原有模块
from ollama_integration import OllamaLLM
from llm_suggest import NetworkMetrics, LLMSuggestion
from sc_qos_optimizer import SCQoSOptimizer, SCQoSConfig
from llm_qos_evaluator import LLMQoSEvaluator

# 设置随机种子
np.random.seed(42)
random.seed(42)

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


@dataclass
class EnhancedNetworkMetrics:
    """增强的网络性能指标"""
    # 基础QoS指标
    avg_qos: float
    avg_effective_qos: float
    avg_objective_score: float
    semantic_accuracy: float
    semantic_rate: float
    llm_semantic_score: float
    user_satisfaction: float

    # 时延与能耗
    avg_latency_ms: float
    avg_energy_joule: float

    # 网络状态指标
    path_congestion: List[float]
    worker_load: List[float]
    switching_rate: float

    # 语义通信指标
    semantic_compression_efficiency: float  # 语义压缩效率
    power_efficiency: float                 # 功率效率
    latency_variance: float                 # 延迟方差
    bandwidth_utilization: float            # 带宽利用率

    # 用户体验指标
    service_continuity: float               # 服务连续性
    qoe_score: float                       # 用户体验质量
    fairness_index: float                  # 公平性指标

    # 系统效率指标
    resource_efficiency: float             # 资源效率
    energy_consumption: float              # 归一化能耗指标（0-1）
    system_stability: float                # 系统稳定性

    timestamp: int


class EnhancedMTUCBBaseline:
    """增强的基础MTUCB算法（固定参数）"""
    
    def __init__(self, num_users=12, num_workers=6, num_paths=4):
        self.num_users = num_users
        self.num_workers = num_workers
        self.num_paths = num_paths
        
        # SC-QOS固定参数配置
        self.config = SCQoSConfig.default()
        
        # 环境参数
        self.beta = 0.8  # 工人偏好权重

        # ✅ 异构工人建模
        self.worker_capacity_profile = np.random.randint(1, 4, size=self.num_workers)
        self.worker_capacity = int(np.round(np.mean(self.worker_capacity_profile)))  # 向后兼容
        self.timeslot_duration_ms = 120.0  # 单个时隙持续时间，用于能耗估计
        self.reference_latency_ms = 250.0
        self.objective_weights = {
            'qos': 0.6,
            'delay': 0.25,
            'energy': 0.15
        }
        
        # 初始化环境
        self._init_enhanced_environment()

        # LLM semantic QoS evaluator
        self.llm_qos_evaluator = LLMQoSEvaluator(auto_generate=True)
        # Weighting between legacy QoS and LLM semantic QoS contribution
        self.qos_weight = 0.7
        self.semantic_weight = 0.3
        self.llm_quality_factor = 1.0  # 动态质量因子，外部方法可设置
        
        # 性能记录
        self.metrics_history: List[EnhancedNetworkMetrics] = []
        self.qos_history = []
        self.effective_qos_history = []
        self.objective_score_history = []
        self.latency_history_ms = []
        self.energy_history_joule = []
        self.semantic_accuracy_history = []
        self.semantic_rate_history = []
        self.llm_semantic_score_history = []
        
        # 新增性能记录
        self.compression_efficiency_history = []
        self.power_efficiency_history = []
        self.qoe_history = []
        self.resource_efficiency_history = []
        self.energy_consumption_history = []
        
        # 匹配和选择记录
        self.R = np.zeros((num_users, num_workers, num_paths))
        self.S = np.zeros((num_users, num_workers, num_paths))
        self.historical_matches = {u: [] for u in range(num_users)}
        
        # 网络状态记录
        self.congestion_history = []
        self.load_balance_history = []
        self.switching_history = []
        
        # 参数历史（基础版也记录固定参数，便于统一可视化）
        self.parameter_history = {
            "alpha": [], "zeta": [], "omega": [],
            "compression_ratio": [], "power_ratio": [], "min_phi": []
        }

    def _init_enhanced_environment(self):
        """初始化增强的仿真环境"""
        # 基础兼容性矩阵
        self.compatibility_matrix = np.random.uniform(0.5, 1.0, 
                                                    (self.num_users, self.num_workers))
        
        # 用户支付意愿和服务需求
        self.willingness_to_pay = np.random.uniform(0.3, 1.0, self.num_users)
        self.service_priority = np.random.uniform(0.4, 1.0, self.num_users)  # 服务优先级
        
        # 路径基础质量和特征
        self.base_path_qualities = np.random.uniform(0.4, 0.9, 
                                                   (self.num_workers, self.num_paths))
        self.path_latency = np.random.uniform(10, 100, (self.num_workers, self.num_paths))  # ms
        self.path_bandwidth = np.random.uniform(50, 200, (self.num_workers, self.num_paths))  # Mbps
        
        # 语义通信特征
        self.semantic_complexity = np.random.uniform(0.3, 0.8, self.num_users)  # 语义复杂度
        self.compression_capability = np.random.uniform(0.5, 0.9, self.num_workers)  # 压缩能力
        
        # 能耗模型
        self.base_power_consumption = np.random.uniform(50, 150, self.num_workers)  # W
        self.power_scaling_factor = np.random.uniform(0.8, 1.2, self.num_workers)

        # 异构性参数
        self.worker_delay_bias = np.random.uniform(0.8, 1.3, self.num_workers)
        self.worker_energy_factor = np.random.uniform(0.9, 1.4, self.num_workers)
        self.worker_latency_jitter = np.random.uniform(5.0, 20.0, self.num_workers)  # ms

        # 参考能耗上限（用于归一化）
        reference_power = np.max(self.base_power_consumption * self.power_scaling_factor * self.worker_energy_factor)
        self.reference_energy = reference_power * (self.timeslot_duration_ms / 1000.0)
    
    def get_enhanced_path_quality(self, t: int, worker: int, path: int) -> Dict[str, float]:
        """获取增强的路径质量信息"""
        base_quality = self.base_path_qualities[worker, path]
        
        # 时间变化的网络状态
        if 50 <= t <= 100:  # 拥塞期
            congestion_factor = 0.6 + 0.3 * np.sin(0.3 * (t - 50))
            latency_penalty = 1.5
        elif 100 < t <= 150:  # 频繁切换期
            switch_factor = 0.7 + 0.4 * np.sin(0.5 * (t - 100)) * np.random.normal(1, 0.2)
            congestion_factor = switch_factor
            latency_penalty = 1.2
        else:
            congestion_factor = 1.0
            latency_penalty = 1.0
        
        # 计算各项质量指标
        quality = base_quality * congestion_factor + 0.05 * np.sin(0.1 * t + worker + path) * np.random.normal(0, 0.1)
        quality = np.clip(quality, 0.1, 1.0)
        
        # 延迟和带宽（考虑工人异构性）
        base_latency = self.path_latency[worker, path] * latency_penalty * self.worker_delay_bias[worker]
        latency_jitter = np.random.normal(0, self.worker_latency_jitter[worker])
        current_latency = max(5.0, base_latency + latency_jitter)
        current_bandwidth = self.path_bandwidth[worker, path] / latency_penalty
        
        return {
            'quality': quality,
            'latency': current_latency,
            'bandwidth': current_bandwidth,
            'congestion_level': 1 - congestion_factor
        }
    
    def calculate_semantic_compression_efficiency(self, user: int, worker: int, path: int) -> float:
        """计算语义压缩效率"""
        user_complexity = self.semantic_complexity[user]
        worker_capability = self.compression_capability[worker]
        compression_ratio = self.config.compression_ratio
        
        # 考虑用户语义复杂度和工人压缩能力
        efficiency = worker_capability * (1 - user_complexity * 0.3) * compression_ratio
        return np.clip(efficiency, 0.1, 1.0)
    
    def calculate_power_efficiency(self, worker: int, load_ratio: float) -> Tuple[float, float]:
        """计算功率效率和实际功率"""
        base_power = self.base_power_consumption[worker]
        scaling = self.power_scaling_factor[worker] * self.worker_energy_factor[worker]
        power_ratio = self.config.power_ratio

        # 功率随负载增加（使用归一化负载）
        actual_power = base_power * (1 + 0.5 * load_ratio) * scaling * power_ratio

        # 效率计算（服务质量/功耗）
        efficiency = 1.0 / (1 + actual_power / 120)  # 归一化，参考功率120W
        efficiency = np.clip(efficiency, 0.1, 1.0)
        return efficiency, actual_power
    
    def calculate_enhanced_qos(self, t: int, user: int, worker: int, path: int,
                             worker_load: int) -> Dict[str, float]:
        """?????QoS??????????????LLM?????"""
        path_info = self.get_enhanced_path_quality(t, worker, path)
        compatibility = self.compatibility_matrix[user, worker]

        base_qos = (
            self.config.alpha * path_info['quality']
            + (1 - self.config.alpha) * compatibility
        )

        # ????????????????????????????
        capacity = max(1, self.worker_capacity_profile[worker])
        raw_ratio = worker_load / capacity
        load_ratio = float(np.clip(raw_ratio, 0.2, 4.0))

        if load_ratio <= 1.0:
            relief = 1.0 - load_ratio
            load_factor = 1.0 + 0.15 * relief
        else:
            overload = load_ratio - 1.0
            load_factor = np.exp(-0.45 * overload)
        base_qos *= load_factor

        compression_eff = self.calculate_semantic_compression_efficiency(user, worker, path)
        power_eff, actual_power = self.calculate_power_efficiency(worker, load_ratio)

        semantic_rate = base_qos * compression_eff
        semantic_penalty = 0.8 if semantic_rate < self.config.min_phi else 1.0

        enhanced_qos = base_qos * (1 + 0.2 * compression_eff) * (1 + 0.1 * power_eff) * semantic_penalty
        enhanced_qos = np.clip(enhanced_qos, 0.1, 1.0)

        latency_scale = 1.0 + 0.7 * max(0.0, load_ratio - 1.0)
        latency_ms = path_info['latency'] * latency_scale
        energy_joule = actual_power * (self.timeslot_duration_ms / 1000.0)

        semantic_score = self.llm_qos_evaluator.get_semantic_score(
            service_priority=float(self.service_priority[user]),
            semantic_complexity=float(self.semantic_complexity[user]),
            path_quality=float(path_info['quality']),
            load_ratio=load_ratio,
            bandwidth_mbps=float(path_info.get('bandwidth', 0.0)),
            latency_ms=float(latency_ms),
        )
        combined_qos = self.qos_weight * enhanced_qos + self.semantic_weight * semantic_score

        effective_qos = combined_qos * np.exp(-latency_ms / max(1.0, self.reference_latency_ms))

        norm_delay = latency_ms / self.reference_latency_ms
        norm_energy = energy_joule / (self.reference_energy + 1e-6)
        objective_score = (
            self.objective_weights['qos'] * combined_qos
            - self.objective_weights['delay'] * norm_delay
            - self.objective_weights['energy'] * norm_energy
        )

        semantic_accuracy = np.clip(0.6 * enhanced_qos + 0.4 * semantic_score, 0.1, 1.0)
        if 50 <= t <= 100:
            semantic_accuracy -= 0.1 * (1 - np.cos(0.1 * (t - 50)))
        semantic_accuracy = np.clip(semantic_accuracy * 0.95, 0.1, 1.0)

        return {
            'qos': enhanced_qos,
            'effective_qos': effective_qos,
            'objective_score': objective_score,
            'semantic_accuracy': semantic_accuracy,
            'semantic_rate': semantic_rate,
            'llm_semantic_score': semantic_score,
            'combined_qos': combined_qos,
            'compression_efficiency': compression_eff,
            'power_efficiency': power_eff,
            'power_watt': actual_power,
            'energy_joule': energy_joule,
            'load_ratio': load_ratio,
            'latency': latency_ms,
            'bandwidth': path_info['bandwidth']
        }

    def collect_enhanced_metrics(self, t: int, matching: List[Tuple[int, int, int]], 
                               qos_results: List[Dict[str, float]]) -> EnhancedNetworkMetrics:
        """收集增强的网络指标"""
        if not matching or not qos_results:
            return self._get_default_metrics(t)
        
        # 基础指标
        avg_qos = np.mean([r['qos'] for r in qos_results])
        avg_effective_qos = np.mean([r['effective_qos'] for r in qos_results])
        avg_objective_score = np.mean([r['objective_score'] for r in qos_results])
        avg_semantic_accuracy = np.mean([r['semantic_accuracy'] for r in qos_results])
        avg_semantic_rate = np.mean([r['semantic_rate'] for r in qos_results])
        avg_semantic_score = np.mean([r.get('llm_semantic_score', 0.0) for r in qos_results])
        avg_latency_ms = np.mean([r['latency'] for r in qos_results])
        avg_energy_joule = np.mean([r['energy_joule'] for r in qos_results])
        
        # 语义通信指标
        compression_efficiency = np.mean([r['compression_efficiency'] for r in qos_results])
        power_efficiency = np.mean([r['power_efficiency'] for r in qos_results])
        
        # 网络性能指标
        latency_variance = np.var([r['latency'] for r in qos_results])
        bandwidth_utilization = np.mean([r['bandwidth'] for r in qos_results]) / 200  # 归一化
        
        # 路径拥塞和工人负载
        path_congestion = []
        worker_load = []
        for w in range(self.num_workers):
            worker_matches = sum(1 for _, worker, _ in matching if worker == w)
            capacity = max(1, self.worker_capacity_profile[w])
            congestion = worker_matches / capacity
            path_congestion.append(congestion)
            worker_load.append(congestion)
        
        # 切换率
        switches = 0
        total_users = 0
        for u in range(self.num_users):
            if len(self.historical_matches[u]) >= 2:
                if self.historical_matches[u][-1][1] != self.historical_matches[u][-2][1]:
                    switches += 1
                total_users += 1
        switching_rate = switches / total_users if total_users > 0 else 0
        
        # 用户满意度和体验质量
        user_satisfaction = avg_qos * 0.85 + np.random.normal(0, 0.02)
        user_satisfaction = np.clip(user_satisfaction, 0, 1)
        
        # QoE评分（考虑多个因素）
        qoe_score = (
            0.32 * avg_qos
            + 0.22 * avg_semantic_accuracy
            + 0.16 * avg_semantic_score
            + 0.18 * compression_efficiency
            + 0.12 * power_efficiency
        )
        qoe_score = np.clip(qoe_score, 0, 1)
        
        # 服务连续性（基于切换频率）
        service_continuity = max(0, 1 - switching_rate * 2)
        
        # 公平性指标（Jain's fairness index）
        qos_values = [r['qos'] for r in qos_results]
        fairness_index = (sum(qos_values) ** 2) / (len(qos_values) * sum(q ** 2 for q in qos_values))
        
        # 资源效率
        resource_efficiency = avg_qos / max(np.mean(worker_load), 0.1)
        resource_efficiency = np.clip(resource_efficiency, 0, 2)
        
        # 能耗（归一化）
        energy_norms = [r['energy_joule'] / (self.reference_energy + 1e-6) for r in qos_results]
        energy_consumption = np.clip(np.mean(energy_norms), 0, 1)
        
        # 系统稳定性
        qos_std = np.std(qos_values)
        system_stability = max(0, 1 - qos_std * 2)
        
        return EnhancedNetworkMetrics(
            avg_qos=avg_qos,
            avg_effective_qos=avg_effective_qos,
            avg_objective_score=avg_objective_score,
            semantic_accuracy=avg_semantic_accuracy,
            semantic_rate=avg_semantic_rate,
            llm_semantic_score=avg_semantic_score,
            user_satisfaction=user_satisfaction,
            avg_latency_ms=avg_latency_ms,
            avg_energy_joule=avg_energy_joule,
            path_congestion=path_congestion,
            worker_load=worker_load,
            switching_rate=switching_rate,
            semantic_compression_efficiency=compression_efficiency,
            power_efficiency=power_efficiency,
            latency_variance=latency_variance,
            bandwidth_utilization=bandwidth_utilization,
            service_continuity=service_continuity,
            qoe_score=qoe_score,
            fairness_index=fairness_index,
            resource_efficiency=resource_efficiency,
            energy_consumption=energy_consumption,
            system_stability=system_stability,
            timestamp=t
        )

    def compute_optimal_qos_for_timestep(self, t: int) -> float:
        """
        计算时隙 t 的贪心最优 QoS 上界（用于遗憾值基准）
        参考 MTUCB._calculate_optimal_qos 的核心思路：容量约束下为每个用户挑选最优工人-路径组合
        """
        best_total_qos = 0.0
        worker_loads = {w: 0 for w in range(self.num_workers)}

        for u in range(self.num_users):
            best_qos_for_user = 0.0
            best_worker = None

            for w in range(self.num_workers):
                capacity = max(1, self.worker_capacity_profile[w])
                if worker_loads[w] >= capacity:
                    continue

                for p in range(self.num_paths):
                    qos_val = self.calculate_enhanced_qos(t, u, w, p, worker_loads[w])
                    if isinstance(qos_val, dict):
                        qos_scalar = qos_val.get('qos', 0.0)
                    else:
                        qos_scalar = float(qos_val)

                    if qos_scalar > best_qos_for_user:
                        best_qos_for_user = qos_scalar
                        best_worker = w

            if best_worker is not None and best_qos_for_user > 0:
                worker_loads[best_worker] += 1
            best_total_qos += best_qos_for_user

        return float(best_total_qos)

    def compute_optimal_objective_for_timestep(self, t: int) -> float:
        """
        使用与实时决策一致的综合 objective_score 计算贪心最优上界。

        逻辑与 compute_optimal_qos_for_timestep 一致，但基于 objective_score
       （即 QoS − delay_penalty − energy_penalty），确保 reward/regret 口径统一。
        """
        best_total_objective = 0.0
        worker_loads = {w: 0 for w in range(self.num_workers)}

        for u in range(self.num_users):
            best_obj_for_user = float('-inf')
            best_worker = None

            for w in range(self.num_workers):
                capacity = max(1, self.worker_capacity_profile[w])
                if worker_loads[w] >= capacity:
                    continue

                for p in range(self.num_paths):
                    qos_val = self.calculate_enhanced_qos(t, u, w, p, worker_loads[w])
                    if isinstance(qos_val, dict):
                        obj_scalar = qos_val.get('objective_score')
                    else:
                        obj_scalar = None

                    if obj_scalar is None:
                        continue

                    if obj_scalar > best_obj_for_user:
                        best_obj_for_user = obj_scalar
                        best_worker = w

            if best_worker is not None and best_obj_for_user > float('-inf'):
                worker_loads[best_worker] += 1
                best_total_objective += best_obj_for_user

        return float(best_total_objective)

    def _get_default_metrics(self, t: int) -> EnhancedNetworkMetrics:
        """获取默认指标（当没有匹配时）"""
        return EnhancedNetworkMetrics(
            avg_qos=0.0,
            avg_effective_qos=0.0,
            avg_objective_score=0.0,
            semantic_accuracy=0.0,
            semantic_rate=0.0,
            llm_semantic_score=0.0,
            user_satisfaction=0.0,
            avg_latency_ms=0.0,
            avg_energy_joule=0.0,
            path_congestion=[0.0] * self.num_workers,
            worker_load=[0.0] * self.num_workers,
            switching_rate=0.0,
            semantic_compression_efficiency=0.0,
            power_efficiency=0.0,
            latency_variance=0.0,
            bandwidth_utilization=0.0,
            service_continuity=0.0,
            qoe_score=0.0,
            fairness_index=0.0,
            resource_efficiency=0.0,
            energy_consumption=0.0,
            system_stability=0.0,
            timestamp=t
        )
    
    def calculate_preference(self, t: int, user: int, worker: int) -> float:
        """计算用户对工人的偏好"""
        total_reward = sum(self.R[user, worker, p] for p in range(self.num_paths))
        total_selections = sum(self.S[user, worker, p] for p in range(self.num_paths))
        
        if total_selections > 0:
            avg_reward = total_reward / total_selections
        else:
            avg_reward = self.compatibility_matrix[user, worker]
        
        ucb_term = np.sqrt(self.config.zeta * np.log(t + 1) / (total_selections + 1))
        
        last_match = self.historical_matches[user][-1][1] if self.historical_matches[user] else None
        switching_cost = self.config.omega if (last_match is not None and last_match != worker) else 0
        
        return avg_reward + ucb_term - switching_cost
    
    def stable_matching(self, t: int, users: Optional[List[int]] = None) -> List[Tuple[int, int]]:
        """Gale-Shapley稳定匹配（可选用户子集）"""
        free_users = list(users) if users is not None else list(range(self.num_users))
        matches = {w: [] for w in range(self.num_workers)}
        proposals = {u: set() for u in range(self.num_users)}
        
        while free_users:
            u = free_users.pop(0)
            
            preferences = sorted(range(self.num_workers), 
                               key=lambda w: self.calculate_preference(t, u, w), 
                               reverse=True)
            
            for w in preferences:
                if w not in proposals[u]:
                    proposals[u].add(w)
                    
                    if len(matches[w]) < self.worker_capacity_profile[w]:
                        matches[w].append(u)
                        break
                    else:
                        def worker_preference(user):
                            compatibility = self.compatibility_matrix[user, w]
                            willingness = self.willingness_to_pay[user]
                            priority = self.service_priority[user]
                            return (self.beta * compatibility + 
                                   (1 - self.beta) * willingness * 0.5 + 
                                   priority * 0.3)
                        
                        worst_u = min(matches[w], key=worker_preference)
                        if worker_preference(u) > worker_preference(worst_u):
                            matches[w].remove(worst_u)
                            matches[w].append(u)
                            free_users.append(worst_u)
                            break
        
        matching = []
        for w in range(self.num_workers):
            for u in matches[w]:
                matching.append((u, w))
        
        return matching
    
    def select_path_ucb(self, t: int, user: int, worker: int) -> int:
        """增强的UCB路径选择"""
        best_score = float('-inf')
        best_path = 0
        
        for p in range(self.num_paths):
            if self.S[user, worker, p] > 0:
                avg_reward = self.R[user, worker, p] / self.S[user, worker, p]
            else:
                avg_reward = self.compatibility_matrix[user, worker]
            
            ucb_term = np.sqrt(self.config.zeta * np.log(t + 1) / (self.S[user, worker, p] + 1))
            
            # 考虑当前路径质量
            path_info = self.get_enhanced_path_quality(t, worker, p)
            quality_bonus = 0.1 * path_info['quality']
            
            score = avg_reward + ucb_term + quality_bonus
            
            if score > best_score:
                best_score = score
                best_path = p
        
        return best_path
    
    def run_simulation(self, T: int = 200):
        """运行增强仿真"""
        print(f"🚀 运行增强MTUCB仿真 (T={T})")
        
        for t in tqdm(range(T), desc="仿真进度"):
            matching = self.stable_matching(t)
            
            matching_with_paths = []
            qos_results = []
            worker_loads = {w: 0 for w in range(self.num_workers)}
            
            for (u, w) in matching:
                worker_loads[w] += 1
                path = self.select_path_ucb(t, u, w)
                qos_result = self.calculate_enhanced_qos(t, u, w, path, worker_loads[w])
                
                matching_with_paths.append((u, w, path))
                qos_results.append(qos_result)
                
                self.R[u, w, path] += qos_result['objective_score']
                self.S[u, w, path] += 1
                
                self.historical_matches[u].append((t, w))
                if len(self.historical_matches[u]) > 10:
                    self.historical_matches[u].pop(0)
            
            # 收集增强指标
            current_metrics = self.collect_enhanced_metrics(t, matching_with_paths, qos_results)
            self.metrics_history.append(current_metrics)
            
            # 记录基础历史
            self.qos_history.append(current_metrics.avg_qos)
            self.effective_qos_history.append(current_metrics.avg_effective_qos)
            self.objective_score_history.append(current_metrics.avg_objective_score)
            self.latency_history_ms.append(current_metrics.avg_latency_ms)
            self.energy_history_joule.append(current_metrics.avg_energy_joule)
            self.semantic_accuracy_history.append(current_metrics.semantic_accuracy)
            self.semantic_rate_history.append(current_metrics.semantic_rate)
            self.llm_semantic_score_history.append(current_metrics.llm_semantic_score)
            
            # 记录新增指标
            self.compression_efficiency_history.append(current_metrics.semantic_compression_efficiency)
            self.power_efficiency_history.append(current_metrics.power_efficiency)
            self.qoe_history.append(current_metrics.qoe_score)
            self.resource_efficiency_history.append(current_metrics.resource_efficiency)
            self.energy_consumption_history.append(current_metrics.energy_consumption)

            # 记录当前参数状态（每个时间步都记录，确保参数历史完整）
            self.parameter_history["alpha"].append(self.config.alpha)
            self.parameter_history["zeta"].append(self.config.zeta)
            self.parameter_history["omega"].append(self.config.omega)
            self.parameter_history["compression_ratio"].append(self.config.compression_ratio)
            self.parameter_history["power_ratio"].append(self.config.power_ratio)
            self.parameter_history["min_phi"].append(self.config.min_phi)


class EnhancedMTUCBWithOllama(EnhancedMTUCBBaseline):
    """集成Ollama LLM的增强MTUCB算法"""
    
    def __init__(self, num_users=12, num_workers=6, num_paths=4, 
                 llm_model="tinyllama", llm_period=25):
        super().__init__(num_users, num_workers, num_paths)
        
        self.llm_period = llm_period
        
        # 初始化Ollama LLM
        print(f"🤖 初始化增强Ollama LLM (模型: {llm_model})")
        self.llm = OllamaLLM(llm_model)
        
        # SC-QOS优化器
        self.sc_optimizer = SCQoSOptimizer(self.config)
        
        # 性能记录
        self.llm_suggestions = []
        self.parameter_history = {
            "alpha": [], "zeta": [], "omega": [],
            "compression_ratio": [], "power_ratio": [], "min_phi": []
        }
        self.llm_call_times = []
        self.confidence_history = []
        
        # 优化效果记录
        self.before_optimization = []  # 优化前性能
        self.after_optimization = []   # 优化后性能
    
    def run_simulation(self, T: int = 200):
        """运行包含LLM优化的增强仿真"""
        print(f"🚀 运行NetLLM增强MTUCB仿真 (T={T})")
        
        for t in tqdm(range(T), desc="仿真进度"):
            # 记录当前参数到历史
            self.parameter_history["alpha"].append(self.config.alpha)
            self.parameter_history["zeta"].append(self.config.zeta)
            self.parameter_history["omega"].append(self.config.omega)
            self.parameter_history["compression_ratio"].append(self.config.compression_ratio)
            self.parameter_history["power_ratio"].append(self.config.power_ratio)
            self.parameter_history["min_phi"].append(self.config.min_phi)
            
            # 执行匹配和路径选择
            matching = self.stable_matching(t)
            
            matching_with_paths = []
            qos_results = []
            worker_loads = {w: 0 for w in range(self.num_workers)}
            
            for (u, w) in matching:
                worker_loads[w] += 1
                path = self.select_path_ucb(t, u, w)
                qos_result = self.calculate_enhanced_qos(t, u, w, path, worker_loads[w])
                
                matching_with_paths.append((u, w, path))
                qos_results.append(qos_result)
                
                self.R[u, w, path] += qos_result['objective_score']
                self.S[u, w, path] += 1
                
                self.historical_matches[u].append((t, w))
                if len(self.historical_matches[u]) > 10:
                    self.historical_matches[u].pop(0)
            
            # 收集增强指标
            current_metrics = self.collect_enhanced_metrics(t, matching_with_paths, qos_results)
            self.metrics_history.append(current_metrics)
            
            # 记录基础历史
            self.qos_history.append(current_metrics.avg_qos)
            self.semantic_accuracy_history.append(current_metrics.semantic_accuracy)
            self.semantic_rate_history.append(current_metrics.semantic_rate)
            
            # 记录新增指标
            self.compression_efficiency_history.append(current_metrics.semantic_compression_efficiency)
            self.power_efficiency_history.append(current_metrics.power_efficiency)
            self.qoe_history.append(current_metrics.qoe_score)
            self.resource_efficiency_history.append(current_metrics.resource_efficiency)
            self.energy_consumption_history.append(current_metrics.energy_consumption)
            
            # LLM优化 - 核心新增逻辑
            if t % self.llm_period == 0 and t > 0:
                print(f"\n🤖 时隙 {t}: NetLLM参数优化...")
                self._perform_llm_optimization(t, current_metrics)
    
    def update_parameters_from_suggestion(self, suggestion: LLMSuggestion, t: int):
        """从LLM建议更新参数"""
        # 应用LLM建议到SC-QOS优化器
        self.sc_optimizer.apply_llm_suggestion(suggestion, t, confidence_threshold=0.5)
        
        # 更新当前配置
        self.config = self.sc_optimizer.config
        
        # 注意：参数历史在仿真主循环中统一记录，这里不重复记录
        
        # 记录置信度
        self.confidence_history.append(suggestion.confidence)
    
    def _build_enhanced_metrics_for_llm(self, t: int, current_metrics: "EnhancedNetworkMetrics"):
        """构建用于LLM的增强网络指标"""
        # 转换为NetworkMetrics格式（兼容原有接口）
        from llm_suggest import NetworkMetrics
        
        return NetworkMetrics(
            avg_qos=current_metrics.avg_qos,
            semantic_accuracy=current_metrics.semantic_accuracy,
            semantic_rate=current_metrics.semantic_rate,
            user_satisfaction=current_metrics.user_satisfaction,
            path_congestion=current_metrics.path_congestion,
            worker_load=current_metrics.worker_load,
            switching_rate=current_metrics.switching_rate,
            timestamp=t
        )
    
    def _perform_llm_optimization(self, t: int, current_metrics: "EnhancedNetworkMetrics"):
        """执行LLM优化"""
        start_time = time.time()
        try:
            # 构建增强的网络状态信息
            enhanced_metrics = self._build_enhanced_metrics_for_llm(t, current_metrics)

            # 区分NetLLMAdapter和其他LLM接口
            if hasattr(self.llm, '_build_deep_prompt'):  # 检查是否是NetLLMAdapter
                suggestion = self.llm.get_suggestion(
                    self.metrics_history[-10:],
                    enhanced_metrics,
                    self.config  # 传递系统当前配置
                )
            else:
                suggestion = self.llm.get_suggestion(
                    self.metrics_history[-10:],
                    enhanced_metrics
                )

            # 记录优化前参数
            old_config = asdict(self.config)

            # 应用建议
            self.sc_optimizer.apply_llm_suggestion(suggestion, t, confidence_threshold=0.5)
            self.config = self.sc_optimizer.config

            # 记录优化后参数
            new_config = asdict(self.config)

            elapsed = time.time() - start_time
            self.llm_call_times.append(elapsed)
            self.llm_suggestions.append((t, suggestion))
            self.confidence_history.append(suggestion.confidence)

            print(f"   🔧 参数优化完成 (模型: {self.llm.model_name}):")
            print(f"   α: {old_config['alpha']:.3f}→{new_config['alpha']:.3f}")
            print(f"   ζ: {old_config['zeta']:.3f}→{new_config['zeta']:.3f}")
            print(f"   ω: {old_config['omega']:.3f}→{new_config['omega']:.3f}")
            print(f"   置信度: {suggestion.confidence:.3f}")
            print(f"   推理: {suggestion.reasoning}")
        except Exception as e:
            print(f"   ❌ LLM优化失败: {e}")
            import traceback
            traceback.print_exc()


def create_comprehensive_comparison_plots(baseline, llm_enhanced):
    """创建全面的性能对比图表"""
    fig = plt.figure(figsize=(20, 16))
    
    # 创建网格布局
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    time_steps = range(len(baseline.qos_history))
    
    # 1. 主要性能对比 (左上角，2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(time_steps, baseline.qos_history, 'b-', linewidth=2.5, 
             label='固定参数MTUCB', alpha=0.8)
    ax1.plot(time_steps, llm_enhanced.qos_history, 'darkorange', linewidth=2.5, 
             label='动态参数MTUCB+Ollama', alpha=0.9)
    
    # 标记关键时期
    ax1.axvspan(50, 100, alpha=0.2, color='red', label='网络拥塞期')
    ax1.axvspan(100, 150, alpha=0.2, color='purple', label='频繁切换期')
    
    # 标记LLM调用点
    for t, _ in llm_enhanced.llm_suggestions:
        if t < len(llm_enhanced.qos_history):
            ax1.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax1.set_title('主要QoS性能对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间槽')
    ax1.set_ylabel('平均QoS')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 语义通信指标对比 (右上角)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time_steps, baseline.semantic_accuracy_history, 'b-', linewidth=2, label='固定参数')
    ax2.plot(time_steps, llm_enhanced.semantic_accuracy_history, 'darkorange', linewidth=2, label='动态参数')
    ax2.set_title('语义准确率', fontsize=12, fontweight='bold')
    ax2.set_ylabel('准确率')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(time_steps, baseline.semantic_rate_history, 'b-', linewidth=2, label='固定参数')
    ax3.plot(time_steps, llm_enhanced.semantic_rate_history, 'darkorange', linewidth=2, label='动态参数')
    ax3.set_title('语义速率', fontsize=12, fontweight='bold')
    ax3.set_ylabel('速率')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 3. 新增语义通信特性对比
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(time_steps, baseline.compression_efficiency_history, 'b-', linewidth=2, label='固定参数')
    ax4.plot(time_steps, llm_enhanced.compression_efficiency_history, 'darkorange', linewidth=2, label='动态参数')
    ax4.set_title('语义压缩效率', fontsize=12, fontweight='bold')
    ax4.set_ylabel('压缩效率')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.plot(time_steps, baseline.power_efficiency_history, 'b-', linewidth=2, label='固定参数')
    ax5.plot(time_steps, llm_enhanced.power_efficiency_history, 'darkorange', linewidth=2, label='动态参数')
    ax5.set_title('功率效率', fontsize=12, fontweight='bold')
    ax5.set_ylabel('功率效率')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 4. 6参数动态调整历史 (第三行)
    ax6 = fig.add_subplot(gs[2, :2])
    param_names = ['Alpha', 'Zeta', 'Omega']
    colors = ['red', 'green', 'blue']
    for i, (param, color) in enumerate(zip(['alpha', 'zeta', 'omega'], colors)):
        ax6.plot(time_steps, llm_enhanced.parameter_history[param], 
                color=color, linewidth=2, label=f'{param_names[i]}')
    ax6.set_title('LLM动态参数调整 (基础参数)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('时间槽')
    ax6.set_ylabel('参数值')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 2:])
    param_names_ext = ['压缩比', '功率比', '最小φ']
    colors_ext = ['purple', 'orange', 'brown']
    for param, color, name in zip(['compression_ratio', 'power_ratio', 'min_phi'], colors_ext, param_names_ext):
        ax7.plot(time_steps, llm_enhanced.parameter_history[param], 
                color=color, linewidth=2, label=name)
    ax7.set_title('LLM动态参数调整 (SC-QOS参数)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('时间槽')
    ax7.set_ylabel('参数值')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 5. 用户体验和系统效率 (第四行)
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.plot(time_steps, baseline.qoe_history, 'b-', linewidth=2, label='固定参数')
    ax8.plot(time_steps, llm_enhanced.qoe_history, 'darkorange', linewidth=2, label='动态参数')
    ax8.set_title('用户体验质量(QoE)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('时间槽')
    ax8.set_ylabel('QoE评分')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[3, 1])
    ax9.plot(time_steps, baseline.resource_efficiency_history, 'b-', linewidth=2, label='固定参数')
    ax9.plot(time_steps, llm_enhanced.resource_efficiency_history, 'darkorange', linewidth=2, label='动态参数')
    ax9.set_title('资源效率', fontsize=12, fontweight='bold')
    ax9.set_xlabel('时间槽')
    ax9.set_ylabel('效率')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.plot(time_steps, baseline.energy_consumption_history, 'b-', linewidth=2, label='固定参数')
    ax10.plot(time_steps, llm_enhanced.energy_consumption_history, 'darkorange', linewidth=2, label='动态参数')
    ax10.set_title('能耗指标', fontsize=12, fontweight='bold')
    ax10.set_xlabel('时间槽')
    ax10.set_ylabel('能耗')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # 6. 性能改善率
    improvement_rates = calculate_improvement_rate(baseline.qos_history, llm_enhanced.qos_history)
    ax11 = fig.add_subplot(gs[3, 3])
    positive_rates = [max(0, rate) for rate in improvement_rates]
    negative_rates = [min(0, rate) for rate in improvement_rates]
    ax11.fill_between(time_steps, 0, positive_rates, alpha=0.7, color='lightgreen', label='性能提升')
    ax11.fill_between(time_steps, 0, negative_rates, alpha=0.7, color='lightcoral', label='性能下降')
    ax11.plot(time_steps, improvement_rates, 'k-', linewidth=1.5, alpha=0.8)
    ax11.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax11.set_title('性能改善率', fontsize=12, fontweight='bold')
    ax11.set_xlabel('时间槽')
    ax11.set_ylabel('改善率 (%)')
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)
    
    plt.suptitle('增强版MTUCB + Ollama LLM 全面性能分析', fontsize=16, fontweight='bold')
    plt.savefig('enhanced_mtucb_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_radar_chart_comparison(baseline, llm_enhanced):
    """创建雷达图对比多维度性能"""
    categories = [
        'QoS', '语义准确率', '语义速率', '压缩效率', 
        '功率效率', 'QoE', '资源效率', '系统稳定性'
    ]
    
    # 计算平均值并归一化
    baseline_values = [
        np.mean(baseline.qos_history),
        np.mean(baseline.semantic_accuracy_history),
        np.mean(baseline.semantic_rate_history),
        np.mean(baseline.compression_efficiency_history),
        np.mean(baseline.power_efficiency_history),
        np.mean(baseline.qoe_history),
        np.mean(baseline.resource_efficiency_history) / 2,  # 归一化
        1 - np.std(baseline.qos_history)  # 稳定性
    ]
    
    llm_values = [
        np.mean(llm_enhanced.qos_history),
        np.mean(llm_enhanced.semantic_accuracy_history),
        np.mean(llm_enhanced.semantic_rate_history),
        np.mean(llm_enhanced.compression_efficiency_history),
        np.mean(llm_enhanced.power_efficiency_history),
        np.mean(llm_enhanced.qoe_history),
        np.mean(llm_enhanced.resource_efficiency_history) / 2,  # 归一化
        1 - np.std(llm_enhanced.qos_history)  # 稳定性
    ]
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合圆形
    
    baseline_values += baseline_values[:1]
    llm_values += llm_values[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='固定参数MTUCB', color='blue')
    ax.fill(angles, baseline_values, alpha=0.25, color='blue')
    
    ax.plot(angles, llm_values, 'o-', linewidth=2, label='动态参数MTUCB+Ollama', color='orange')
    ax.fill(angles, llm_values, alpha=0.25, color='orange')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('多维度性能雷达图对比', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.savefig('enhanced_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_heatmap_analysis(baseline, llm_enhanced):
    """创建热力图分析"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 参数变化热力图
    params_data = np.array([
        llm_enhanced.parameter_history['alpha'],
        llm_enhanced.parameter_history['zeta'],
        llm_enhanced.parameter_history['omega'],
        llm_enhanced.parameter_history['compression_ratio'],
        llm_enhanced.parameter_history['power_ratio'],
        llm_enhanced.parameter_history['min_phi']
    ])
    
    param_labels = ['Alpha', 'Zeta', 'Omega', '压缩比', '功率比', '最小φ']
    
    # 重采样数据以便可视化
    sample_interval = max(1, len(llm_enhanced.parameter_history['alpha']) // 50)
    sampled_data = params_data[:, ::sample_interval]
    time_labels = list(range(0, len(llm_enhanced.parameter_history['alpha']), sample_interval))
    
    im1 = axes[0, 0].imshow(sampled_data, cmap='RdYlBu_r', aspect='auto')
    axes[0, 0].set_title('LLM参数动态调整热力图', fontweight='bold')
    axes[0, 0].set_xlabel('时间槽')
    axes[0, 0].set_ylabel('参数')
    axes[0, 0].set_yticks(range(len(param_labels)))
    axes[0, 0].set_yticklabels(param_labels)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 性能指标对比热力图
    metrics_baseline = np.array([
        baseline.qos_history[::sample_interval],
        baseline.semantic_accuracy_history[::sample_interval],
        baseline.compression_efficiency_history[::sample_interval],
        baseline.power_efficiency_history[::sample_interval],
        baseline.qoe_history[::sample_interval]
    ])
    
    metrics_llm = np.array([
        llm_enhanced.qos_history[::sample_interval],
        llm_enhanced.semantic_accuracy_history[::sample_interval],
        llm_enhanced.compression_efficiency_history[::sample_interval],
        llm_enhanced.power_efficiency_history[::sample_interval],
        llm_enhanced.qoe_history[::sample_interval]
    ])
    
    metric_labels = ['QoS', '语义准确率', '压缩效率', '功率效率', 'QoE']
    
    im2 = axes[0, 1].imshow(metrics_baseline, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('固定参数性能热力图', fontweight='bold')
    axes[0, 1].set_xlabel('时间槽')
    axes[0, 1].set_ylabel('性能指标')
    axes[0, 1].set_yticks(range(len(metric_labels)))
    axes[0, 1].set_yticklabels(metric_labels)
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].imshow(metrics_llm, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('动态参数性能热力图', fontweight='bold')
    axes[1, 0].set_xlabel('时间槽')
    axes[1, 0].set_ylabel('性能指标')
    axes[1, 0].set_yticks(range(len(metric_labels)))
    axes[1, 0].set_yticklabels(metric_labels)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 3. 改善矩阵
    improvement_matrix = metrics_llm - metrics_baseline
    im4 = axes[1, 1].imshow(improvement_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)
    axes[1, 1].set_title('性能改善热力图', fontweight='bold')
    axes[1, 1].set_xlabel('时间槽')
    axes[1, 1].set_ylabel('性能指标')
    axes[1, 1].set_yticks(range(len(metric_labels)))
    axes[1, 1].set_yticklabels(metric_labels)
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('enhanced_heatmap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_optimization_effectiveness_plot(llm_enhanced):
    """创建优化效果分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. LLM调用时间和置信度
    call_times = [t for t, _ in llm_enhanced.llm_suggestions]
    confidences = [s.confidence for _, s in llm_enhanced.llm_suggestions]
    
    axes[0, 0].scatter(call_times, confidences, c=confidences, cmap='RdYlGn', s=100)
    axes[0, 0].set_title('LLM建议置信度变化', fontweight='bold')
    axes[0, 0].set_xlabel('时间槽')
    axes[0, 0].set_ylabel('置信度')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 参数变化幅度
    param_changes = []
    for i in range(1, len(llm_enhanced.parameter_history['alpha'])):
        change = abs(llm_enhanced.parameter_history['alpha'][i] - llm_enhanced.parameter_history['alpha'][i-1])
        change += abs(llm_enhanced.parameter_history['zeta'][i] - llm_enhanced.parameter_history['zeta'][i-1])
        change += abs(llm_enhanced.parameter_history['omega'][i] - llm_enhanced.parameter_history['omega'][i-1])
        param_changes.append(change)
    
    axes[0, 1].plot(range(1, len(param_changes)+1), param_changes, 'r-', linewidth=2)
    axes[0, 1].set_title('参数变化幅度', fontweight='bold')
    axes[0, 1].set_xlabel('时间槽')
    axes[0, 1].set_ylabel('总变化幅度')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 性能改善趋势
    window_size = 20
    performance_trend = []
    for i in range(window_size, len(llm_enhanced.qos_history)):
        recent_avg = np.mean(llm_enhanced.qos_history[i-window_size:i])
        performance_trend.append(recent_avg)
    
    axes[1, 0].plot(range(window_size, len(llm_enhanced.qos_history)), performance_trend, 
                   'g-', linewidth=2, label='滑动平均QoS')
    axes[1, 0].set_title('性能改善趋势', fontweight='bold')
    axes[1, 0].set_xlabel('时间槽')
    axes[1, 0].set_ylabel('QoS')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 各指标相关性分析
    if len(llm_enhanced.metrics_history) > 50:
        recent_metrics = llm_enhanced.metrics_history[-50:]
        correlation_data = np.array([
            [m.avg_qos for m in recent_metrics],
            [m.semantic_accuracy for m in recent_metrics],
            [m.semantic_compression_efficiency for m in recent_metrics],
            [m.power_efficiency for m in recent_metrics],
            [m.qoe_score for m in recent_metrics]
        ])
        
        correlation_matrix = np.corrcoef(correlation_data)
        labels = ['QoS', '语义准确率', '压缩效率', '功率效率', 'QoE']
        
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('性能指标相关性', fontweight='bold')
        axes[1, 1].set_xticks(range(len(labels)))
        axes[1, 1].set_yticks(range(len(labels)))
        axes[1, 1].set_xticklabels(labels, rotation=45)
        axes[1, 1].set_yticklabels(labels)
        
        # 添加数值标注
        for i in range(len(labels)):
            for j in range(len(labels)):
                axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('optimization_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_improvement_rate(baseline_values, llm_values, window_size=10):
    """计算滑动窗口改善率"""
    improvement_rates = []
    
    for i in range(len(baseline_values)):
        if i < window_size:
            baseline_avg = np.mean(baseline_values[:i+1])
            llm_avg = np.mean(llm_values[:i+1])
        else:
            baseline_avg = np.mean(baseline_values[i-window_size:i+1])
            llm_avg = np.mean(llm_values[i-window_size:i+1])
        
        if baseline_avg > 0:
            improvement = ((llm_avg - baseline_avg) / baseline_avg) * 100
        else:
            improvement = 0
        
        improvement_rates.append(improvement)
    
    return improvement_rates


def print_enhanced_comparison_statistics(baseline, llm_enhanced):
    """打印增强的对比统计结果"""
    print("\n" + "="*80)
    print("📊 增强版MTUCB vs MTUCB+Ollama 全面性能分析")
    print("="*80)
    
    # 基础性能对比
    baseline_qos = np.mean(baseline.qos_history)
    llm_qos = np.mean(llm_enhanced.qos_history)
    qos_improvement = ((llm_qos - baseline_qos) / baseline_qos) * 100
    
    print(f"🎯 基础性能对比:")
    print(f"   平均QoS: {baseline_qos:.4f} → {llm_qos:.4f} ({qos_improvement:+.2f}%)")
    
    # 语义通信指标对比
    metrics_comparison = {
        '语义准确率': (baseline.semantic_accuracy_history, llm_enhanced.semantic_accuracy_history),
        '语义速率': (baseline.semantic_rate_history, llm_enhanced.semantic_rate_history),
        '压缩效率': (baseline.compression_efficiency_history, llm_enhanced.compression_efficiency_history),
        '功率效率': (baseline.power_efficiency_history, llm_enhanced.power_efficiency_history),
        '用户体验(QoE)': (baseline.qoe_history, llm_enhanced.qoe_history),
        '资源效率': (baseline.resource_efficiency_history, llm_enhanced.resource_efficiency_history),
        '能耗指标': (baseline.energy_consumption_history, llm_enhanced.energy_consumption_history),
    }
    
    print(f"\n🔬 语义通信与系统效率对比:")
    for metric_name, (baseline_values, llm_values) in metrics_comparison.items():
        baseline_avg = np.mean(baseline_values)
        llm_avg = np.mean(llm_values)
        improvement = ((llm_avg - baseline_avg) / baseline_avg) * 100
        print(f"   {metric_name}: {baseline_avg:.4f} → {llm_avg:.4f} ({improvement:+.2f}%)")
    
    # 网络适应性分析
    print(f"\n🌐 网络状态适应性:")
    
    # 拥塞期表现
    congestion_period_baseline = np.mean(baseline.qos_history[50:101])
    congestion_period_llm = np.mean(llm_enhanced.qos_history[50:101])
    congestion_improvement = ((congestion_period_llm - congestion_period_baseline) / congestion_period_baseline) * 100
    
    # 频繁切换期表现
    switch_period_baseline = np.mean(baseline.qos_history[100:151])
    switch_period_llm = np.mean(llm_enhanced.qos_history[100:151])
    switch_improvement = ((switch_period_llm - switch_period_baseline) / switch_period_baseline) * 100
    
    print(f"   拥塞期QoS改善: {congestion_improvement:+.2f}%")
    print(f"   频繁切换期QoS改善: {switch_improvement:+.2f}%")
    
    # LLM优化效果
    if llm_enhanced.llm_call_times:
        print(f"\n🤖 LLM优化效果:")
        print(f"   总调用次数: {len(llm_enhanced.llm_call_times)}")
        print(f"   平均响应时间: {np.mean(llm_enhanced.llm_call_times):.1f}秒")
        print(f"   平均置信度: {np.mean(llm_enhanced.confidence_history):.3f}")
        
        # 参数变化统计
        param_changes = {
            'alpha': np.std(llm_enhanced.parameter_history['alpha']),
            'zeta': np.std(llm_enhanced.parameter_history['zeta']),
            'omega': np.std(llm_enhanced.parameter_history['omega']),
            'compression_ratio': np.std(llm_enhanced.parameter_history['compression_ratio']),
            'power_ratio': np.std(llm_enhanced.parameter_history['power_ratio']),
            'min_phi': np.std(llm_enhanced.parameter_history['min_phi'])
        }
        
        print(f"\n📊 参数动态调整范围:")
        for param, std_val in param_changes.items():
            print(f"   {param}: 标准差 {std_val:.4f}")
    
    # 系统稳定性分析
    baseline_stability = 1 - np.std(baseline.qos_history)
    llm_stability = 1 - np.std(llm_enhanced.qos_history)
    
    print(f"\n⚖️ 系统稳定性:")
    print(f"   固定参数稳定性: {baseline_stability:.4f}")
    print(f"   动态参数稳定性: {llm_stability:.4f}")
    print(f"   稳定性变化: {(llm_stability - baseline_stability):+.4f}")
    
    print("="*80)


def main():
    """主函数 - 运行增强版对比实验"""
    print("🎯 增强版MTUCB + Ollama LLM 全面对比实验")
    print("="*60)
    
    # 实验参数
    num_users = 12
    num_workers = 6
    num_paths = 4
    T = 200
    llm_period = 25
    
    print(f"📋 实验配置:")
    print(f"   用户数: {num_users}, 工人数: {num_workers}, 路径数: {num_paths}")
    print(f"   时间槽: {T}, LLM调用周期: {llm_period}")
    print(f"   优化参数: 6个 (alpha, zeta, omega, compression_ratio, power_ratio, min_phi)")
    
    # 运行基础算法
    print("\n🔵 运行增强版基础MTUCB算法（固定参数）...")
    baseline = EnhancedMTUCBBaseline(num_users, num_workers, num_paths)
    baseline.run_simulation(T)
    print("✅ 基础算法完成")
    
    # 运行LLM增强算法
    print("\n🟠 运行增强版MTUCB + Ollama算法（6参数动态优化）...")
    llm_enhanced = EnhancedMTUCBWithOllama(
        num_users, num_workers, num_paths,
        llm_model="tinyllama", llm_period=llm_period
    )
    llm_enhanced.run_simulation(T)
    print("✅ LLM增强算法完成")
    
    # 生成全面的可视化分析
    print("\n📊 生成全面性能分析图表...")
    
    print("   📈 创建综合对比图...")
    create_comprehensive_comparison_plots(baseline, llm_enhanced)
    
    print("   🎯 创建雷达图对比...")
    create_radar_chart_comparison(baseline, llm_enhanced)
    
    print("   🔥 创建热力图分析...")
    create_heatmap_analysis(baseline, llm_enhanced)
    
    print("   ⚡ 创建优化效果分析...")
    create_optimization_effectiveness_plot(llm_enhanced)
    
    # 打印详细统计结果
    print_enhanced_comparison_statistics(baseline, llm_enhanced)
    
    print("\n🎉 增强版实验完成！")
    print("💡 生成的图表文件:")
    print("   • enhanced_mtucb_comprehensive_analysis.png - 综合性能分析")
    print("   • enhanced_radar_comparison.png - 多维度雷达图对比") 
    print("   • enhanced_heatmap_analysis.png - 热力图分析")
    print("   • optimization_effectiveness_analysis.png - 优化效果分析")
    
    # 保存数据用于进一步分析
    results_summary = {
        'baseline_avg_qos': np.mean(baseline.qos_history),
        'llm_avg_qos': np.mean(llm_enhanced.qos_history),
        'qos_improvement': ((np.mean(llm_enhanced.qos_history) - np.mean(baseline.qos_history)) / np.mean(baseline.qos_history)) * 100,
        'llm_calls': len(llm_enhanced.llm_suggestions),
        'avg_confidence': np.mean(llm_enhanced.confidence_history) if llm_enhanced.confidence_history else 0,
        'avg_response_time': np.mean(llm_enhanced.llm_call_times) if llm_enhanced.llm_call_times else 0
    }
    
    print(f"\n📋 实验总结:")
    print(f"   总体QoS改善: {results_summary['qos_improvement']:+.2f}%")
    print(f"   LLM调用次数: {results_summary['llm_calls']}")
    print(f"   平均置信度: {results_summary['avg_confidence']:.3f}")
    print(f"   平均响应时间: {results_summary['avg_response_time']:.1f}秒")


if __name__ == "__main__":
    main() 
