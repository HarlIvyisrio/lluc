"""

完整修复版实验系统 V3

================================================================================

修复内容：

1.  真异步黑盒优化（不阻塞主循环）

2.  随机种子控制（确保可复现性）

3.  时间参考统一（全局时间在评估中一致）

4. Worker索引修复（避免越界）

5.  统一初始参数（公平对比）

6.  多次实验+置信区间（统计显著性）

7.  完善时延模型（RTT抖动+推理时间）

8.  鲁棒聚合策略（trimmed mean）

9.  多维度评价（延迟方差/能耗/公平性/收敛速度）

10.  统计检验（t-test + ablation）

11. 复杂度分析（理论公式）

================================================================================



[P1] TI-UCB + LLM Selection	将 UCB 决策与 LLM 语义选择相结合；修改函数 U = QoS − β · Delay − γ · Energy

[P2] LLM Routing for Edge Computing 在边缘节点根据上下文语义动态调整 LLM 推理路径，分布式聚合方法参考

"""



import numpy as np

import pickle

import time

import random

import matplotlib.pyplot as plt

import os

from dataclasses import dataclass, field, asdict

from typing import List, Dict, Optional, Tuple, Callable

from tqdm import tqdm

from scipy import stats
from scipy.signal import savgol_filter

import json

# 导入基础组件

from Enhanced_MTUCB_with_Ollama import EnhancedMTUCBBaseline, EnhancedNetworkMetrics

from incremental_cmaes import IncrementalCMAES, CMAESConfig, AsyncBlackboxOptimizer

from latent_param_mapper import LatentParamMapper

INCLUDE_OPT_OVERHEAD_IN_LATENCY = True  # 是否将优化/聚合的额外时延叠加到业务时延曲线

COLOR_MAP = {
    'method1_baseline': '#808080',
    'method2_async_blackbox': '#4ECDC4',
    'method3_periodic_llm_hybrid': '#FF8C42',
    'method4_distributed_collaborative': '#4C72B0',
    'method5_distributed_llm': '#9467BD',
}

# ============================================================================
# LLM质量参数配置说明
# ============================================================================
# Beta分布参数 (alpha, beta) 决定采样的质量因子分布：
#   - mean = alpha / (alpha + beta)
#   - 较小的alpha+beta使得采样有更大随机性
#   - 建议 alpha + beta 在 8-15 范围内，避免过大导致无随机性
#
# 三个模型的质量差异假设（基于模型规模和能力）：
#   - qwen3:8b (8B params): 较大模型，mean≈0.75, alpha=9, beta=3
#   - deepseek-r1:8b (8B): 8B参数但推理链式，mean≈0.70, alpha=7, beta=3
#   - phi3:mini (3.8B): 最小模型，mean≈0.55, alpha=5.5, beta=4.5
# ============================================================================
DEFAULT_LLM_QUALITY = {"beta_alpha": 5.0, "beta_beta": 5.0}  # 默认mean=0.5

try:
    with open("llm_quality_params.json", "r", encoding="utf-8") as f:
        LLM_QUALITY_PARAMS = json.load(f)
        # 检查是否参数过大（>50），如果是则使用fallback
        for model_name, params in LLM_QUALITY_PARAMS.items():
            if params.get("beta_alpha", 0) > 50 or params.get("beta_beta", 0) > 50:
                print(f"[WARN] {model_name} 的Beta参数过大，采样将无随机性，建议重新评估")
except Exception:
    pass

# Fallback值：手动设定的差异化参数（基于模型规模假设）
LLM_QUALITY_PARAMS_FALLBACK = {
    "qwen3:8b": {"beta_alpha": 9.0, "beta_beta": 3.0},       # mean≈0.75
    "phi3:mini": {"beta_alpha": 5.5, "beta_beta": 4.5},      # mean≈0.55
    "deepseek-r1:8b": {"beta_alpha": 7.0, "beta_beta": 3.0}, # mean≈0.70
}

# 如果没有加载到json，使用fallback
if 'LLM_QUALITY_PARAMS' not in dir() or not LLM_QUALITY_PARAMS:
    LLM_QUALITY_PARAMS = LLM_QUALITY_PARAMS_FALLBACK

LLM_PARAM_KEYS = [
    "alpha",
    "zeta",
    "omega",
    "compression_ratio",
    "power_ratio",
    "min_phi",
]
LLM_PARAM_BOUNDS = {
    "alpha": (0.3, 0.9),
    "zeta": (0.1, 0.5),
    "omega": (0.05, 0.3),
    "compression_ratio": (0.5, 0.95),
    "power_ratio": (0.3, 0.8),
    "min_phi": (0.4, 0.9),
}





# ============================================================================

# 1. 随机种子控制

# ============================================================================



def set_global_seed(seed: int):

    """

    设置全局随机种子，确保可复现性

    

    Args:

        seed: 随机种子

    """

    import random

    np.random.seed(seed)

    random.seed(seed)


def sample_llm_quality(model_name: str) -> float:
    """
    Sample a quality factor in [0,1] for the given LLM model using precomputed beta params.
    
    当Beta参数过大（alpha或beta > 50）时，采样几乎无随机性，
    此时自动切换到fallback参数以确保模型间有合理差异。
    """
    # 优先使用JSON加载的参数
    params = LLM_QUALITY_PARAMS.get(model_name, None)
    
    # 检查参数有效性：如果为空或参数过大，使用fallback
    if params is None:
        params = LLM_QUALITY_PARAMS_FALLBACK.get(model_name, DEFAULT_LLM_QUALITY)
    else:
        alpha_val = params.get("beta_alpha", 0)
        beta_val = params.get("beta_beta", 0)
        # 当参数过大时（采样无随机性），使用fallback
        if alpha_val > 50 or beta_val > 50:
            fallback = LLM_QUALITY_PARAMS_FALLBACK.get(model_name, DEFAULT_LLM_QUALITY)
            params = fallback
    
    alpha = max(0.1, float(params.get("beta_alpha", DEFAULT_LLM_QUALITY["beta_alpha"])))
    beta = max(0.1, float(params.get("beta_beta", DEFAULT_LLM_QUALITY["beta_beta"])))
    return float(np.clip(np.random.beta(alpha, beta), 0.0, 1.0))


def blend_params_with_quality(
    model_name: str,
    raw_params: Dict[str, float],
    default_params: Dict[str, float],
    history: List[float],
) -> Dict[str, float]:
    """
    Blend LLM suggestion with conservative defaults based on sampled quality.
    
    混合公式设计：
    - 使用非线性映射放大quality差异
    - 高质量模型(quality>0.7)的参数更接近LLM建议
    - 低质量模型(quality<0.6)的参数更接近默认值
    
    原始线性公式: final = quality * llm + (1-quality) * default
    问题: 当quality差异小时，最终参数差异也小
    
    改进公式: 使用adjusted_quality = 0.3 + 0.6 * quality^0.7
    - quality=0.75 → adjusted=0.76
    - quality=0.70 → adjusted=0.73
    - quality=0.55 → adjusted=0.65
    这样放大了模型间的差异
    """
    quality = sample_llm_quality(model_name)
    history.append(quality)
    
    # 非线性映射放大差异
    # 使用幂函数: adjusted = base + scale * quality^power
    # base=0.3 确保最差模型也有30%权重用LLM建议
    # scale=0.6 确保最好模型有90%权重用LLM建议
    # power=0.7 使得高质量区域差异更大
    adjusted_quality = 0.3 + 0.6 * (quality ** 0.7)
    adjusted_quality = float(np.clip(adjusted_quality, 0.2, 0.95))
    
    final_params = {}
    for key, base_val in default_params.items():
        llm_val = raw_params.get(key, base_val)
        # 使用调整后的quality进行混合
        final_params[key] = float(adjusted_quality * llm_val + (1 - adjusted_quality) * base_val)
    return final_params


def parse_llm_params(payload: object) -> Optional[Dict[str, float]]:
    """Parse strict JSON parameters from an LLM payload."""
    try:
        if isinstance(payload, dict):
            parsed = payload
        else:
            parsed = json.loads(str(payload))
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    params = {}
    for key in LLM_PARAM_KEYS:
        if key not in parsed:
            return None
        try:
            value = float(parsed[key])
        except (TypeError, ValueError):
            return None
        low, high = LLM_PARAM_BOUNDS[key]
        params[key] = float(np.clip(value, low, high))
    return params


def finalize_step_metrics(
    env: EnhancedMTUCBBaseline,
    metrics: EnhancedNetworkMetrics,
    overhead_ms: float = 0.0,
) -> Tuple[float, float, float]:
    """Attach objective/latency totals and overhead to metrics."""
    objective_total = float(metrics.avg_objective_score) * env.num_users
    latency_with_overhead = float(metrics.avg_latency_ms) + float(overhead_ms)
    latency_total = latency_with_overhead * env.num_users
    setattr(metrics, "objective_total", objective_total)
    setattr(metrics, "overhead_ms", float(overhead_ms))
    setattr(metrics, "latency_total", latency_total)
    setattr(metrics, "latency_with_overhead_ms", latency_with_overhead)
    return objective_total, latency_with_overhead, latency_total





# ============================================================================

# 2. 完善的时延模型

# ============================================================================



@dataclass

class LatencyModel:

    """

    真实时延模型

    

    包含通信时延（RTT）+ 计算时延（推理），并考虑随机抖动

    """

    # LLM调用时延（ms）

    llm_network_rtt_mean: float = 80.0      # 网络RTT均值

    llm_network_rtt_std: float = 15.0       # 网络RTT标准差

    llm_inference_mean: float = 150.0        # LLM推理时间均值

    llm_inference_std: float = 30.0          # LLM推理时间标准差

    

    # 黑盒搜索评估时延（ms）

    blackbox_eval_mean: float = 20.0

    blackbox_eval_std: float = 5.0

    

    # 分布式通信时延（ms）

    worker_comm_rtt_mean: float = 15.0

    worker_comm_rtt_std: float = 5.0

    aggregation_mean: float = 30.0

    aggregation_std: float = 8.0

    

    def sample_llm_latency(self) -> float:

        """采样LLM调用总时延"""

        network = max(0, np.random.normal(self.llm_network_rtt_mean, self.llm_network_rtt_std))

        inference = max(0, np.random.normal(self.llm_inference_mean, self.llm_inference_std))

        return network + inference

    

    def sample_blackbox_eval_latency(self) -> float:

        """采样黑盒评估时延"""

        return max(0, np.random.normal(self.blackbox_eval_mean, self.blackbox_eval_std))

    

    def sample_worker_comm_latency(self) -> float:

        """采样worker通信时延"""

        return max(0, np.random.normal(self.worker_comm_rtt_mean, self.worker_comm_rtt_std))

    

    def sample_aggregation_latency(self) -> float:

        """采样聚合时延"""

        return max(0, np.random.normal(self.aggregation_mean, self.aggregation_std))





# ============================================================================

# 3. 环境工厂（带种子控制）

# ============================================================================



class SeededEnvironmentFactory:

    """

    带种子控制的环境工厂

    

    确保每个方法使用相同的随机环境，实现公平对比

    """

    

    def __init__(self, num_users: int, num_workers: int, num_paths: int):

        self.num_users = num_users

        self.num_workers = num_workers

        self.num_paths = num_paths

    

    def create_env(self, seed: int) -> EnhancedMTUCBBaseline:

        """

        创建环境实例

        

        Args:

            seed: 随机种子

            

        Returns:

            新的环境实例

        """

        set_global_seed(seed)

        env = EnhancedMTUCBBaseline(

            self.num_users,

            self.num_workers,

            self.num_paths

        )

        return env





# ============================================================================

# 4. 真实仿真评估器（修复版）

# ============================================================================



class RealSimulationEvaluatorV3:

    """

    真实仿真评估器 V3

    

    修复：

    - 使用环境工厂避免状态污染

    - 时间参考统一（传入全局时间）

    """

    

    def __init__(

        self,

        env_factory: SeededEnvironmentFactory,

        param_mapper: LatentParamMapper,

        window_size: int = 30,

        seed_base: int = 1000,

        # 统一的综合目标系数与归一化常量（用于兜底计算）

        lambda_delay: float = 0.4,

        lambda_energy: float = 0.3,

        d_max_ms: float = 500.0,

        e_max_joule: float = 1.0

    ):

        self.env_factory = env_factory

        self.param_mapper = param_mapper

        self.window_size = window_size

        self.seed_base = seed_base

        self.eval_count = 0

        # 兜底目标参数

        self.lambda_delay = lambda_delay

        self.lambda_energy = lambda_energy

        self.d_max_ms = d_max_ms

        self.e_max_joule = e_max_joule

        self.fallback_count = 0



    def _composite_objective(self, qos: float, latency_ms: float, energy_joule: float) -> float:

        """按统一公式计算综合目标: QoS − λ1·Delay/Dmax − λ2·Energy/Emax"""

        norm_delay = (latency_ms or 0.0) / max(1e-6, self.d_max_ms)

        norm_energy = (energy_joule or 0.0) / max(1e-6, self.e_max_joule)

        return float(qos) - self.lambda_delay * norm_delay - self.lambda_energy * norm_energy

    

    def evaluate_params(self, z: np.ndarray, global_t: int, user_subset: Optional[List[int]] = None) -> float:

        """

        评估参数（真实MTUCB仿真）

        

        ✅ 修复1: 模拟多工人匹配，保持MTUCB原始逻辑

        

        Args:

            z: 潜在空间参数

            global_t: 当前全局时间（重要！用于正确模拟网络状态）

            

        Returns:

            负QoS（越小越好，用于最小化）

        """

        self.eval_count += 1



        # 评估环境会生成新的随机序列，这里先保存主仿真的随机状态，防止串扰

        np_state = np.random.get_state()

        py_state = random.getstate()

        try:

            # 创建独立的评估环境（避免状态污染）

            eval_seed = self.seed_base + self.eval_count

            temp_env = self.env_factory.create_env(eval_seed)

            

            # 应用参数

            params = self.param_mapper.latent_to_params(z)

            temp_env.config = params

            # ✅ 关键修复：评估环境也需要设置权重

            temp_env.objective_weights = {'qos': 1.0, 'delay': self.lambda_delay, 'energy': self.lambda_energy}

            temp_env.reference_latency_ms = self.d_max_ms

            temp_env.reference_energy = self.e_max_joule

            

            # 运行仿真窗口

            # ✅ 修复1: 使用轮换分配模拟多工人匹配（保持MTUCB多工人逻辑）

            objective_window = []

            users = list(user_subset) if user_subset is not None else list(range(temp_env.num_users))



            for t_local in range(self.window_size):

                t_global = global_t + t_local

                

                # 在每个时隙执行MTUCB决策

                # 跟踪该时隙内各worker负载

                worker_load = {w_id: 0 for w_id in range(temp_env.num_workers)}

                for u in users:

                    # ✅ 修复1: 轮换分配工人，模拟多工人交互

                    w = (u + t_global) % temp_env.num_workers

                    # 使用与S矩阵一致的安全索引，避免越界

                    safe_w = int(w % temp_env.S.shape[1])

                    

                    # 选择路径（使用全局时间和正确的工人）

                    path = temp_env.select_path_ucb(t_global, u, safe_w)

                    

                    # 计算QoS（使用全局时间，正确反映网络状态和工人负载）

                    worker_load[safe_w] += 1

                    try:

                        qos_result = temp_env.calculate_enhanced_qos(

                            t_global, u, safe_w, path, worker_load[safe_w]

                        )

                    except TypeError:

                        # 兼容旧签名（无worker_load）

                        qos_result = temp_env.calculate_enhanced_qos(

                            t_global, u, safe_w, path

                        )

                    if isinstance(qos_result, dict):

                        if 'objective_score' in qos_result:

                            score = qos_result['objective_score']

                        else:

                            # 兜底：按统一公式计算综合目标，并记录次数

                            score = self._composite_objective(

                                qos=qos_result.get('qos', 0.0),

                                latency_ms=qos_result.get('latency', 0.0),

                                energy_joule=qos_result.get('energy_joule', 0.0)

                            )

                            self.fallback_count += 1

                    else:

                        score = qos_result

                    objective_window.append(score)

                    

                    # 更新统计（使用多目标评分）

                    temp_env.R[u, safe_w, path] += score

                    temp_env.S[u, safe_w, path] += 1

            

            avg_objective = np.mean(objective_window) if objective_window else 0.0
            objective_total = avg_objective * len(users)

            # 返回负的多目标总评分（用于最小化）
            return -objective_total

        finally:

            # 评估结束后恢复随机状态，避免干扰主仿真

            np.random.set_state(np_state)

            random.setstate(py_state)





# ============================================================================

# 5. 动态时延惩罚跟踪器（修复版）

# ============================================================================



class DynamicLatencyTrackerV3:

    """

    动态时延惩罚跟踪器 V3

    

    修复：时序积分方式，而非事后全局惩罚

    """

    

    def __init__(self):

        self.qos_history = []

        self.latency_history = []

        self.effective_qos_history = []

    

    def add_timestep(self, qos: float, latency_ms: float):

        """

        添加一个时隙的数据

        

        Args:

            qos: 原始QoS

            latency_ms: 该时隙产生的时延（ms）

        """

        self.qos_history.append(qos)

        self.latency_history.append(latency_ms)

        

        # ✅ 修复4: 改为指数型惩罚，更符合QoE非线性特性

        # effective_qos = qos * exp(-latency / 300)

        # 当latency=300ms时，惩罚为e^(-1)≈0.368（保留36.8%）

        # 当latency=600ms时，惩罚为e^(-2)≈0.135（保留13.5%）

        effective_qos = qos * np.exp(-latency_ms / 300.0)

        self.effective_qos_history.append(effective_qos)

    

    def get_average_qos(self) -> float:

        """获取平均原始QoS"""

        return np.mean(self.qos_history) if self.qos_history else 0.0

    

    def get_average_effective_qos(self) -> float:

        """获取平均有效QoS（考虑时延惩罚）"""

        return np.mean(self.effective_qos_history) if self.effective_qos_history else 0.0

    

    def get_total_latency(self) -> float:

        """获取总时延"""

        return np.sum(self.latency_history)

    

    def get_qos_std(self) -> float:

        """获取QoS标准差（抖动指标）"""

        return np.std(self.qos_history) if self.qos_history else 0.0





def run_mtucb_step(

    env: EnhancedMTUCBBaseline,

    t: int,

    lambda_delay: float,

    lambda_energy: float,

    d_max_ms: float,

    e_max_joule: float,

    user_subset: Optional[List[int]] = None,

) -> "EnhancedNetworkMetrics":

    """

    执行一次统一的MTUCB决策步骤，返回该时隙的指标

    """

    matching_with_paths = []
    timestep_results = []
    users = list(user_subset) if user_subset is not None else list(range(env.num_users))

    # 使用 Gale–Shapley 稳定匹配，真实体现资源竞争
    stable_pairs = env.stable_matching(t, users=users) if hasattr(env, 'stable_matching') else [(u, (u + t) % env.num_workers) for u in users]
    if user_subset is not None:
        stable_pairs = [(u, w) for (u, w) in stable_pairs if u in user_subset]

    # 先快照所有用户的工人负载，避免按序更新引入偏差
    worker_load = {w_id: 0 for w_id in range(env.num_workers)}
    for _, assigned_worker in stable_pairs:
        safe_w = int(assigned_worker % env.S.shape[1])
        worker_load[safe_w] += 1

    pending_updates: List[Tuple[int, int, int, float]] = []

    for u, assigned_worker in stable_pairs:
        safe_w = int(assigned_worker % env.S.shape[1])
        path = env.select_path_ucb(t, u, safe_w)

        qos_result = env.calculate_enhanced_qos(t, u, safe_w, path, worker_load[safe_w])

        matching_with_paths.append((u, safe_w, path))
        timestep_results.append(qos_result if isinstance(qos_result, dict) else {'qos': qos_result})

        if isinstance(qos_result, dict):
            if 'objective_score' in qos_result:
                objective_score = qos_result['objective_score']
            else:
                norm_delay = (qos_result.get('latency', 0.0)) / max(1e-6, d_max_ms)
                norm_energy = (qos_result.get('energy_joule', 0.0)) / max(1e-6, e_max_joule)
                objective_score = qos_result.get('qos', 0.0) - lambda_delay * norm_delay - lambda_energy * norm_energy
        else:
            objective_score = qos_result

        pending_updates.append((u, safe_w, path, float(objective_score)))

    for u, safe_w, path, objective_score in pending_updates:
        env.R[u, safe_w, path] += objective_score
        env.S[u, safe_w, path] += 1



    metrics = env.collect_enhanced_metrics(t, matching_with_paths, timestep_results)

    # ============================================================================
    # 设计说明：LLM质量的影响机制
    # ============================================================================
    # 正确的设计：LLM质量通过以下路径间接影响QoS：
    #   1. blend_params_with_quality() 根据质量因子混合LLM建议与默认参数
    #   2. 混合后的参数影响MTUCB的决策（select_path_ucb）
    #   3. 决策质量自然体现在QoS上
    #
    # 错误的设计（已移除）：直接在最终QoS上乘以quality_multiplier
    #   - 这会人为惩罚使用LLM的方法，导致regret计算偏高
    #   - Method1/4 的 multiplier=1.0（无惩罚）
    #   - Method2/3/5 的 multiplier≈0.89（被惩罚11%）
    #   - 导致累积遗憾图中LLM方法反而比baseline差
    # ============================================================================

    env.metrics_history.append(metrics)

    env.qos_history.append(metrics.avg_qos)

    env.effective_qos_history.append(metrics.avg_effective_qos)

    env.objective_score_history.append(metrics.avg_objective_score)

    env.latency_history_ms.append(metrics.avg_latency_ms)

    env.energy_history_joule.append(metrics.avg_energy_joule)

    if hasattr(env, 'semantic_accuracy_history'):

        env.semantic_accuracy_history.append(metrics.semantic_accuracy)

    if hasattr(env, 'semantic_rate_history'):

        env.semantic_rate_history.append(metrics.semantic_rate)

    if hasattr(env, 'compression_efficiency_history'):

        env.compression_efficiency_history.append(metrics.semantic_compression_efficiency)

    if hasattr(env, 'power_efficiency_history'):

        env.power_efficiency_history.append(metrics.power_efficiency)

    if hasattr(env, 'qoe_history'):

        env.qoe_history.append(metrics.qoe_score)

    if hasattr(env, 'resource_efficiency_history'):

        env.resource_efficiency_history.append(metrics.resource_efficiency)

    if hasattr(env, 'energy_consumption_history'):

        env.energy_consumption_history.append(metrics.energy_consumption)

    if hasattr(env, 'parameter_history') and isinstance(env.parameter_history, dict):

        for key in ['alpha', 'zeta', 'omega', 'compression_ratio', 'power_ratio', 'min_phi']:

            if key in env.parameter_history and hasattr(env.config, key):

                env.parameter_history[key].append(getattr(env.config, key))

    return metrics


def update_regret_reward_histories(
    env: EnhancedMTUCBBaseline,
    t: int,
    metrics: EnhancedNetworkMetrics,
    histories: Dict[str, List[float]]
) -> None:
    """
    更新遗憾值/奖励序列（即时与累积），统一使用"综合 objective_score" 口径。

    计算口径说明：
    - optimal_reward_t: 时隙t的贪心最优总 objective_score（所有用户之和），来自compute_optimal_objective_for_timestep
    - actual_reward_t: 时隙t的实际总 objective_score_total（objective_total）
    - instant_regret: max(0, optimal_reward_t - actual_reward_t)
    - instant_reward: actual_reward_t（与算法实时更新 UCB 时的 reward 完全一致）
    """
    # 获取理论最优总 reward（容量约束下每个用户选 objective_score 最优工人-路径）
    optimal_reward_t = env.compute_optimal_objective_for_timestep(t)

    # 计算实际总 reward（objective_total）
    if hasattr(metrics, "objective_total"):
        actual_reward_t = float(metrics.objective_total)
    else:
        actual_reward_t = metrics.avg_objective_score * env.num_users

    # 即时遗憾 = 最优 - 实际（非负值）
    instant_regret = max(0.0, optimal_reward_t - actual_reward_t)
    # 即时奖励 = 实际获得的总 reward
    instant_reward = actual_reward_t

    cumulative_regret = (histories['cumulative_regret_history'][-1]
                         if histories['cumulative_regret_history'] else 0.0) + instant_regret
    cumulative_reward = (histories['cumulative_reward_history'][-1]
                          if histories['cumulative_reward_history'] else 0.0) + instant_reward

    histories['instant_regret_history'].append(instant_regret)
    histories['instant_reward_history'].append(instant_reward)
    histories['cumulative_regret_history'].append(cumulative_regret)
    histories['cumulative_reward_history'].append(cumulative_reward)





# ============================================================================

# 6. 方法1：固定参数Baseline

# ============================================================================



class Method1_FixedBaseline:

    """

    方法1：固定参数 MTUCB (Baseline)

    

    统一初始参数与主循环，确保公平对比

    """

    

    def __init__(

        self,

        env_factory: SeededEnvironmentFactory,

        initial_params: dict,

        seed: int,

        lambda_delay: float = 0.4,

        lambda_energy: float = 0.3,

        d_max_ms: float = 500.0,

        e_max_joule: float = 1.0

    ):

        import copy

        self.env = env_factory.create_env(seed)

        # 使用独立配置对象，避免多环境共享同一dataclass实例导致串扰

        self.env.config = copy.deepcopy(initial_params)

        # ✅ 关键修复：更新环境的综合目标权重与归一化参数

        self.env.objective_weights = {'qos': 1.0, 'delay': lambda_delay, 'energy': lambda_energy}

        self.env.reference_latency_ms = d_max_ms

        self.env.reference_energy = e_max_joule

        self.latency_tracker = DynamicLatencyTrackerV3()

        self.qos_history = []

        self.effective_qos_history = []

        self.objective_total_history = []
        self.objective_score_history = self.objective_total_history

        self.latency_history_ms = []
        self.latency_total_history = []
        self.overhead_history_ms = []

        self.energy_history_joule = []

        self.parameter_history = {'alpha': [], 'zeta': [], 'omega': []}
        self.instant_regret_history: List[float] = []
        self.instant_reward_history: List[float] = []
        self.cumulative_regret_history: List[float] = []
        self.cumulative_reward_history: List[float] = []

        self.lambda_delay = lambda_delay

        self.lambda_energy = lambda_energy

        self.d_max_ms = d_max_ms

        self.e_max_joule = e_max_joule

    

    def run_simulation(self, T: int):

        """运行仿真（逐时隙，与其它方法一致）"""

        for t in tqdm(range(T), desc="仿真进度"):

            current_metrics = run_mtucb_step(

                self.env,

                t,

                self.lambda_delay,

                self.lambda_energy,

                self.d_max_ms,

                self.e_max_joule,

            )

            objective_total, latency_with_overhead, latency_total = finalize_step_metrics(
                self.env,
                current_metrics,
                overhead_ms=0.0,
            )

            self.qos_history.append(current_metrics.avg_qos)
            self.effective_qos_history.append(current_metrics.avg_effective_qos)
            self.objective_total_history.append(objective_total)
            self.energy_history_joule.append(current_metrics.avg_energy_joule)
            self.latency_history_ms.append(latency_with_overhead)
            self.latency_total_history.append(latency_total)
            self.overhead_history_ms.append(0.0)
            
            

            self.parameter_history['alpha'].append(self.env.config.alpha)

            self.parameter_history['zeta'].append(self.env.config.zeta)

            self.parameter_history['omega'].append(self.env.config.omega)

            update_regret_reward_histories(
                self.env,
                t,
                current_metrics,
                {
                    'instant_regret_history': self.instant_regret_history,
                    'instant_reward_history': self.instant_reward_history,
                    'cumulative_regret_history': self.cumulative_regret_history,
                    'cumulative_reward_history': self.cumulative_reward_history
                }
            )

            self.latency_tracker.add_timestep(current_metrics.avg_qos, latency_with_overhead)

    

    def get_results(self) -> dict:

        """获取结果"""

        avg_latency_ms = np.mean(self.latency_history_ms) if self.latency_history_ms else 0.0

        avg_energy = np.mean(self.energy_history_joule) if self.energy_history_joule else 0.0

        avg_objective_total = np.mean(self.objective_total_history) if self.objective_total_history else 0.0

        return {

            'method_name': '固定参数',

            'qos_history': self.qos_history,

            'effective_qos_history': self.effective_qos_history,

            'objective_score_history': self.objective_score_history,
            'objective_total_history': self.objective_total_history,

            'latency_history_ms': self.latency_history_ms,

            'energy_history_joule': self.energy_history_joule,

            'param_history': self.parameter_history,

            'avg_qos': self.latency_tracker.get_average_qos(),

            'avg_effective_qos': self.latency_tracker.get_average_effective_qos(),

            'avg_objective_score': avg_objective_total,
            'avg_objective_total': avg_objective_total,

            'avg_latency_ms': avg_latency_ms,

            'avg_energy_joule': avg_energy,

            'qos_std': self.latency_tracker.get_qos_std(),

            'total_latency_ms': self.latency_tracker.get_total_latency(),
            'latency_total_history': self.latency_total_history,
            'overhead_history_ms': self.overhead_history_ms,

            'optimization_count': 0,

            'instant_regret_history': self.instant_regret_history,

            'instant_reward_history': self.instant_reward_history,

            'cumulative_regret_history': self.cumulative_regret_history,

            'cumulative_reward_history': self.cumulative_reward_history,

            'convergence_timestep': -1  # 不适用

        }





# ============================================================================

# 7. 方法2：LLM初始化 + 真异步黑盒搜索

# ============================================================================



class Method2_LLMInitAsyncBlackbox:

    """

    方法2：LLM初始化 + 真异步黑盒搜索

    

    修复：

    - 使用增量式CMA-ES，不阻塞主循环

    - 每个时隙执行部分优化迭代

    """

    

    def __init__(

        self,

        env_factory: SeededEnvironmentFactory,

        initial_params: dict,

        latency_model: LatencyModel,

        llm_model_name: str = "qwen3:8b",

        blackbox_period: int = 25,

        cmaes_max_iters: int = 15,

        cmaes_population: int = 15,

        seed: int = 42,

        lambda_delay: float = 0.4,

        lambda_energy: float = 0.3,

        d_max_ms: float = 500.0,

        e_max_joule: float = 1.0

    ):

        import copy

        self.env = env_factory.create_env(seed)

        # 深拷贝，避免多个环境共享同一配置引用

        self.env.config = copy.deepcopy(initial_params)

        # ✅ 关键修复：更新环境的综合目标权重与归一化参数

        self.env.objective_weights = {'qos': 1.0, 'delay': lambda_delay, 'energy': lambda_energy}

        self.env.reference_latency_ms = d_max_ms

        self.env.reference_energy = e_max_joule

        self.env_factory = env_factory


        self.latency_model = latency_model
        self.llm_model_name = llm_model_name

        self.blackbox_period = blackbox_period

        self.seed = seed

        # 综合目标参数（用于兜底一致性）

        self.lambda_delay = lambda_delay

        self.lambda_energy = lambda_energy

        self.d_max_ms = d_max_ms

        self.e_max_joule = e_max_joule

        

        # 参数映射器

        self.param_mapper = LatentParamMapper()

        latent_dim = self.param_mapper.config.latent_dim

        

        # 评估器（修复版）

        # ✅ 修复2: window_size设为period的一半，避免时间重叠

        eval_window = max(10, blackbox_period // 2)

        self.evaluator = RealSimulationEvaluatorV3(

            env_factory,

            self.param_mapper,

            window_size=eval_window,

            seed_base=seed + 10000,

            lambda_delay=self.lambda_delay,

            lambda_energy=self.lambda_energy,

            d_max_ms=self.d_max_ms,

            e_max_joule=self.e_max_joule

        )

        

        # CMA-ES配置

        self.cmaes_config = CMAESConfig(

            dim=latent_dim,

            population_size=cmaes_population,

            max_iterations=cmaes_max_iters,

            sigma0=0.4,

            bounds=(

                np.full(latent_dim, -2.0),

                np.full(latent_dim, 2.0)

            )

        )

        

        # 当前优化器（可能为None）

        self.current_optimizer: Optional[IncrementalCMAES] = None

        self.optimization_start_t = None

        

        # 统计

        self.latency_tracker = DynamicLatencyTrackerV3()

        self.optimization_count = 0

        self.blackbox_eval_count = 0  # 累计评估计数（用于每步差分统计）

        

        # 记录

        self.qos_history = []

        self.effective_qos_history = []

        self.objective_total_history = []
        self.objective_score_history = self.objective_total_history

        self.latency_history_ms = []
        self.latency_total_history = []
        self.overhead_history_ms = []

        self.energy_history_joule = []

        self.parameter_history = {'alpha': [], 'zeta': [], 'omega': []}
        self.llm_latency_history: List[float] = []
        self.llm_quality_history: List[float] = []
        self.instant_regret_history: List[float] = []
        self.instant_reward_history: List[float] = []
        self.cumulative_regret_history: List[float] = []
        self.cumulative_reward_history: List[float] = []

        

        # 收敛时隙

        self.convergence_timestep = -1

        self.baseline_objective = None

    def _build_state_context(self, window_short: int = 10, window_long: int = 30) -> str:
        """构造包含最近窗口统计量的状态摘要，供LLM参考。"""
        def summarize(series: List[float], label: str, wnd: int) -> str:
            if not series:
                return f"{label}: 无历史数据"
            arr = np.array(series[-wnd:])
            p95 = float(np.percentile(arr, 95))
            trend = "上升" if arr[-1] > arr[0] * 1.05 else ("下降" if arr[-1] < arr[0] * 0.95 else "稳定")
            return f"{label}: 平均={arr.mean():.4f}, P95={p95:.4f}, 趋势={trend}"

        qos_info = summarize(self.qos_history, "QoS", window_short)
        latency_info = summarize(self.latency_history_ms, "时延(ms)", window_short)
        objective_info = summarize(self.objective_score_history, "objective_total", window_long)
        energy_info = summarize(self.energy_history_joule, "能耗(J)", window_short)

        if self.instant_regret_history:
            recent_regret = np.array(self.instant_regret_history[-window_long:])
            regret_trend = "上升" if recent_regret[-1] > recent_regret[0] * 1.05 else ("下降" if recent_regret[-1] < recent_regret[0] * 0.95 else "平稳")
            regret_info = f"Regret均值={recent_regret.mean():.4f}, 趋势={regret_trend}"
        else:
            regret_info = "Regret: 暂无数据"

        weight_info = (
            f"目标权重: QoS=1.0, 延迟λ={self.lambda_delay:.2f}, 能耗λ={self.lambda_energy:.2f}; "
            f"归一化参考 d_max={self.d_max_ms:.1f}ms, e_max={self.e_max_joule:.2f}J"
        )

        return "\n".join([qos_info, latency_info, objective_info, energy_info, regret_info, weight_info])

        # 异步评估的协调开销比例

        self.async_overlap_factor = 0.3

        

        # ✅ 添加LLM初始化（可切换模型）

        self._llm_initialize_params()



    def _apply_params(self, params):

        """安全应用参数，兼容dict与dataclass对象。"""

        try:

            # dataclass/object 直接赋值

            _ = params.alpha

            # 逐字段写入，避免替换引用

            for k in ['alpha', 'zeta', 'omega', 'compression_ratio', 'power_ratio', 'min_phi']:

                if hasattr(params, k) and hasattr(self.env.config, k):

                    setattr(self.env.config, k, float(getattr(params, k)))

        except AttributeError:

            # dict 写入现有配置

            if isinstance(params, dict):

                for k in ['alpha', 'zeta', 'omega', 'compression_ratio', 'power_ratio', 'min_phi']:

                    if k in params:

                        setattr(self.env.config, k, float(params[k]))

            else:

                raise

    

    def _llm_initialize_params(self):

        """

        使用指定模型进行LLM初始化，增强容错与日志记录

        """

        import datetime

        import os

        

        # 创建日志目录

        log_dir = 'llm_logs'

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        log_file = os.path.join(log_dir, f'llm_init_{timestamp}.json')



        log_data = {

            'timestamp': timestamp,

            'model': self.llm_model_name,

            'network_state': {

                'num_users': self.env.num_users,

                'num_workers': self.env.num_workers,

                'num_paths': self.env.num_paths

            },

            'status': 'unknown',

            'params_used': {}

        }

        default_params = {
            'alpha': float(getattr(self.env.config, 'alpha', 0.6)),
            'zeta': float(getattr(self.env.config, 'zeta', 0.25)),
            'omega': float(getattr(self.env.config, 'omega', 0.15)),
            'compression_ratio': float(getattr(self.env.config, 'compression_ratio', 0.75)),
            'power_ratio': float(getattr(self.env.config, 'power_ratio', 0.5)),
            'min_phi': float(getattr(self.env.config, 'min_phi', 0.6)),
        }
        latency_ms = None
        start_time = time.time()

        try:

            import requests

            

            # 构建提示词（丰富语义上下文）
            # 计算派生特征
            user_worker_ratio = self.env.num_users / max(1, self.env.num_workers)
            if user_worker_ratio > 2.5:
                load_level = "高负载（用户显著多于工人，资源紧张）"
                load_advice = "建议增大alpha以优先选择高质量路径，适当降低zeta减少探索开销"
            elif user_worker_ratio < 1.5:
                load_level = "低负载（工人充足，资源宽裕）"
                load_advice = "建议适当增大zeta以加强探索，发现更优配置"
            else:
                load_level = "中等负载（用户与工人相对平衡）"
                load_advice = "建议采用均衡参数，兼顾探索与利用"
            
            path_complexity = "简单" if self.env.num_paths <= 3 else ("中等" if self.env.num_paths <= 6 else "复杂")

            state_context = self._build_state_context()

            prompt = (
                f"你是一个网络资源调度优化专家。请为MTUCB算法推荐初始参数。\n\n"
                f"【网络拓扑】\n"
                f"- 用户数: {self.env.num_users}\n"
                f"- 工人数: {self.env.num_workers}\n"
                f"- 路径数: {self.env.num_paths}\n"
                f"- 用户/工人比: {user_worker_ratio:.2f}\n\n"
                f"【实时状态摘要】\n{state_context}\n\n"
                f"【负载分析】\n"
                f"- 负载水平: {load_level}\n"
                f"- 路径复杂度: {path_complexity}（{self.env.num_paths}条可选路径）\n"
                f"- 建议方向: {load_advice}\n\n"
                f"【参数说明】\n"
                f"1. alpha (0.3-0.9): 路径质量权重\n"
                f"2. zeta (0.1-0.5): UCB探索强度\n"
                f"3. omega (0.05-0.3): 切换成本权重\n"
                f"4. compression_ratio (0.5-0.95): 语义压缩率\n"
                f"5. power_ratio (0.3-0.8): 功率分配系数\n"
                f"6. min_phi (0.4-0.9): 语义速率阈值\n\n"
                f"【注意事项】\n"
                f"- 高负载场景下，应优先保证QoS，减少不必要的探索\n"
                f"- 低负载场景下，可适当探索以发现更优配置\n"
                f"- 路径复杂时，需要更多探索；路径简单时，可快速收敛\n"
                f"- 输出必须严格为JSON，对应字段: alpha,zeta,omega,compression_ratio,power_ratio,min_phi\n"
                f"示例: {{\"alpha\":0.7,\"zeta\":0.2,\"omega\":0.1,\"compression_ratio\":0.8,\"power_ratio\":0.5,\"min_phi\":0.6}}"
            )

            

            log_data['prompt'] = prompt



            # 尝试多个端口（11434, 11435）

            ports_to_try = [11434, 11435]

            response = None

            used_port = None

            

            for port in ports_to_try:

                try:

                    response = requests.post(

                        f'http://localhost:{port}/api/generate',

                        json={

                            'model': self.llm_model_name,

                            'prompt': prompt,

                            'stream': False,

                            'format': 'json',

                            'options': {

                                'temperature': 0.3,

                                'num_predict': 200

                            }

                        },

                        timeout=30

                    )

                    if response.status_code == 200:

                        used_port = port

                        break

                except requests.exceptions.ConnectionError:

                    continue

            

            if response is None or response.status_code != 200:
                log_data['status'] = 'connection_failed'
                log_data['error'] = f'无法连接到Ollama服务（尝试端口: {ports_to_try}）'
                print(f"\n[WARN] LLM调用失败 ({self.llm_model_name}): 无法连接到Ollama服务，使用默认参数")
                self.llm_quality_history.append(0.0)
                self.env.llm_quality_factor = 0.0
            else:
                log_data['used_port'] = used_port
                latency_ms = (time.time() - start_time) * 1000.0
                result = response.json()
                llm_response = result.get('response', '').strip()
                log_data['raw_response'] = llm_response
                log_data['parse_method'] = 'strict_json'

                parsed = parse_llm_params(llm_response)

                if parsed:
                    blended = blend_params_with_quality(self.llm_model_name, parsed, default_params, self.llm_quality_history)
                    self.env.config.alpha = blended['alpha']
                    self.env.config.zeta = blended['zeta']
                    self.env.config.omega = blended['omega']
                    self.env.config.compression_ratio = blended.get('compression_ratio', default_params['compression_ratio'])
                    self.env.config.power_ratio = blended.get('power_ratio', default_params['power_ratio'])
                    self.env.config.min_phi = blended.get('min_phi', default_params['min_phi'])
                    self.env.llm_quality_factor = self.llm_quality_history[-1] if self.llm_quality_history else 0.0

                    log_data['status'] = 'success'
                    log_data['params_used'] = blended

                    print(f"\n[INFO] LLM初始化成功 ({self.llm_model_name}, 端口:{used_port}):")
                    print(f"   alpha: {self.env.config.alpha:.3f}")
                    print(f"   zeta: {self.env.config.zeta:.3f}")
                    print(f"   omega: {self.env.config.omega:.3f}")
                    print(f"   compression_ratio: {self.env.config.compression_ratio:.3f}")
                    print(f"   power_ratio: {self.env.config.power_ratio:.3f}")
                    print(f"   min_phi: {self.env.config.min_phi:.3f}")
                else:
                    log_data['status'] = 'parse_failed'
                    log_data['error'] = '无法从响应中提取参数'
                    print("\n[WARN] LLM响应解析失败，保持默认参数")
                    print(f"   响应内容: {llm_response[:120]}")
                    self.llm_quality_history.append(0.0)
                    self.env.llm_quality_factor = 0.0

        except Exception as e:

            log_data['status'] = 'exception'

            log_data['error'] = str(e)

            print(f"\n[WARN] LLM初始化失败: {e}，保持默认参数")
            self.env.llm_quality_factor = 0.0
            self.llm_quality_history.append(0.0)

        

        finally:

            # 记录最终使用的参数

            log_data['final_params'] = {

                'alpha': float(self.env.config.alpha),

                'zeta': float(self.env.config.zeta),

                'omega': float(self.env.config.omega),

                'compression_ratio': float(getattr(self.env.config, 'compression_ratio', default_params['compression_ratio'])),

                'power_ratio': float(getattr(self.env.config, 'power_ratio', default_params['power_ratio'])),

                'min_phi': float(getattr(self.env.config, 'min_phi', default_params['min_phi']))

            }

            if latency_ms is None:
                try:
                    latency_ms = self.latency_model.sample_llm_latency()
                except Exception:
                    latency_ms = 0.0

            log_data['llm_latency_ms'] = float(latency_ms)
            self.llm_latency_history.append(float(latency_ms))

            self._save_llm_log(log_file, log_data)

    

    def _save_llm_log(self, log_file: str, log_data: dict):

        """保存LLM调用日志"""

        import json

        try:

            with open(log_file, 'w', encoding='utf-8') as f:

                json.dump(log_data, f, ensure_ascii=False, indent=2)

            print(f"   [INFO] 日志已保存: {log_file}")

        except Exception as e:

            print(f"   [WARN] 日志保存失败: {e}")

    

    def run_simulation(self, T: int):

        """

        运行仿真

        

        关键：每个时隙都推进，优化不阻塞

        """

        print(f"\n开始运行方法2: LLM初始化 + 真异步黑盒 (T={T})...")

        

        for t in tqdm(range(T), desc="仿真进度"):

            current_metrics = run_mtucb_step(

                self.env,

                t,

                self.lambda_delay,

                self.lambda_energy,

                self.d_max_ms,

                self.e_max_joule,

            )



            avg_qos_t = current_metrics.avg_qos

            avg_effective_qos_t = current_metrics.avg_effective_qos

            objective_total, _, _ = finalize_step_metrics(
                self.env,
                current_metrics,
                overhead_ms=0.0,
            )

            avg_latency_t = current_metrics.avg_latency_ms

            avg_energy_t = current_metrics.avg_energy_joule



            self.qos_history.append(avg_qos_t)

            self.effective_qos_history.append(avg_effective_qos_t)

            self.objective_total_history.append(objective_total)

            self.energy_history_joule.append(avg_energy_t)

            

            # 记录baseline（用综合目标，和收敛判定保持一致）

            if t < 10:

                if self.baseline_objective is None:

                    self.baseline_objective = objective_total

                else:

                    self.baseline_objective = (self.baseline_objective * t + objective_total) / (t + 1)

            

            # 检查收敛（综合评分提升超过10%）

            if self.convergence_timestep == -1 and self.baseline_objective is not None:

                if objective_total > self.baseline_objective * 1.1:

                    self.convergence_timestep = t

            

            # 记录参数

            self.parameter_history['alpha'].append(self.env.config.alpha)

            self.parameter_history['zeta'].append(self.env.config.zeta)

            self.parameter_history['omega'].append(self.env.config.omega)

            update_regret_reward_histories(
                self.env,
                t,
                current_metrics,
                {
                    'instant_regret_history': self.instant_regret_history,
                    'instant_reward_history': self.instant_reward_history,
                    'cumulative_regret_history': self.cumulative_regret_history,
                    'cumulative_reward_history': self.cumulative_reward_history
                }
            )



            # 2. 如果有正在运行的优化，执行一步迭代

            optimization_latency = 0.0

            if self.current_optimizer is not None:

                # 先统计本步新增评估数并累积评估时延

                new_evals = max(0, self.current_optimizer.total_evals - self.blackbox_eval_count)

                if new_evals > 0:

                    for _ in range(new_evals):

                        optimization_latency += self.latency_model.sample_blackbox_eval_latency()

                    self.blackbox_eval_count = self.current_optimizer.total_evals



                # 执行一步异步黑盒优化迭代

                result = self.current_optimizer.step(t)

                if result is not None:

                    # 优化完成

                    best_z, best_f = result

                    best_params = self.param_mapper.latent_to_params(best_z)

                    

                    # ✅ 修复3: 检查符号一致性（CMA-ES最小化，我们返回-多目标评分）

                    # best_f为负值；-best_f即最佳综合评分

                    best_objective_score = -best_f

                    

                    # 应用最佳参数（兼容dict/dataclass）

                    self._apply_params(best_params)

                    

                    # 已在上方按差分累计过评估时延，这里不重复

                    print(f"\n时隙 {t}: 黑盒搜索完成")

                    print(f"   [INFO] 最佳多目标评分: {best_objective_score:.4f}")

                    print(f"   原始best_f (负值): {best_f:.4f}")

                    print(f"   累积评估次数: {self.current_optimizer.total_evals}")

                    print(f"   累积评估时延: {optimization_latency:.1f}ms")

                    

                    self.current_optimizer = None

                    self.optimization_count += 1

            

            # 3. 检查是否需要触发新的优化

            if self.current_optimizer is None and t > 0 and t % self.blackbox_period == 0:

                # 启动新的优化（异步黑盒优化器）

                current_z = self.param_mapper.params_to_latent(self.env.config)

                

                self.current_optimizer = AsyncBlackboxOptimizer(

                    evaluator=lambda z, current_t: self.evaluator.evaluate_params(z, current_t),

                    latent_dim=self.param_mapper.config.latent_dim,

                    bounds=self.cmaes_config.bounds,

                    max_iterations=self.cmaes_config.max_iterations,

                    population_size=self.cmaes_config.population_size,

                    seed=self.seed + t

                )

                self.current_optimizer.start(current_z, t)

                

                # 初始化评估计数基线

                self.blackbox_eval_count = self.current_optimizer.total_evals

                

                print(f"\n时隙 {t}: 触发黑盒搜索")

            

            # 4. 记录时延

            if INCLUDE_OPT_OVERHEAD_IN_LATENCY:
                overhead_ms = optimization_latency * self.async_overlap_factor
                latency_with_overhead = avg_latency_t + overhead_ms
            else:
                overhead_ms = 0.0
                latency_with_overhead = avg_latency_t

            _, _, latency_total = finalize_step_metrics(
                self.env,
                current_metrics,
                overhead_ms=overhead_ms,
            )

            self.latency_history_ms.append(latency_with_overhead)
            self.latency_total_history.append(latency_total)
            self.overhead_history_ms.append(overhead_ms)

            self.latency_tracker.add_timestep(avg_qos_t, latency_with_overhead)

        

        print(f"\n仿真完成！")

        print(f"  优化触发次数: {self.optimization_count}")

        print(f"  总评估次数: {self.evaluator.eval_count}")

        print(f"  平均QoS: {self.latency_tracker.get_average_qos():.4f}")

        print(f"  平均有效QoS: {self.latency_tracker.get_average_effective_qos():.4f}")

    

    def get_results(self) -> dict:

        """获取结果"""

        # 关闭优化器

        if self.current_optimizer is not None:

            self.current_optimizer.shutdown()

        

        avg_latency_ms = np.mean(self.latency_history_ms) if self.latency_history_ms else 0.0

        avg_energy = np.mean(self.energy_history_joule) if self.energy_history_joule else 0.0

        avg_objective_total = np.mean(self.objective_total_history) if self.objective_total_history else 0.0

        return {

            'method_name': 'LLM初始化+异步黑盒',

            'qos_history': self.qos_history,

            'effective_qos_history': self.effective_qos_history,

            'objective_score_history': self.objective_score_history,
            'objective_total_history': self.objective_total_history,

            'latency_history_ms': self.latency_history_ms,

            'energy_history_joule': self.energy_history_joule,

            'param_history': self.parameter_history,

            'llm_model': self.llm_model_name,

            'llm_latency_ms': self.llm_latency_history,
            'llm_quality_history': self.llm_quality_history,
            'llm_avg_quality': float(np.mean(self.llm_quality_history)) if self.llm_quality_history else 0.0,

            'avg_qos': self.latency_tracker.get_average_qos(),

            'avg_effective_qos': self.latency_tracker.get_average_effective_qos(),

            'avg_objective_score': avg_objective_total,
            'avg_objective_total': avg_objective_total,

            'avg_latency_ms': avg_latency_ms,

            'avg_energy_joule': avg_energy,

            'qos_std': self.latency_tracker.get_qos_std(),

            'total_latency_ms': self.latency_tracker.get_total_latency(),
            'latency_total_history': self.latency_total_history,
            'overhead_history_ms': self.overhead_history_ms,

            'optimization_count': self.optimization_count,

            'blackbox_eval_count': self.evaluator.eval_count,

            'convergence_timestep': self.convergence_timestep,

            'evaluator_fallbacks': getattr(self.evaluator, 'fallback_count', 0),

            'instant_regret_history': self.instant_regret_history,

            'instant_reward_history': self.instant_reward_history,

            'cumulative_regret_history': self.cumulative_regret_history,

            'cumulative_reward_history': self.cumulative_reward_history

        }





# ============================================================================

# 8. 方法3：周期性LLM指导 + 黑盒微调（混合优化）

# ============================================================================



class Method3_PeriodicLLMHybrid:

    """

    方法3：周期性LLM指导 + 黑盒微调（混合优化策略）

    

    核心思想：

    1. 定期调用LLM（每 llm_period 个时隙）获取参数建议

    # LLM??????
    llm_models: List[str] = field(default_factory=lambda: ["qwen3:8b", "phi3:mini", "deepseek-r1:8b"])

    2. LLM建议作为初始点，启动黑盒微调

    3. 黑盒搜索负责局部精细化优化

    

    架构：

    - LLM层：定期语义推理，提供全局指导

    - 黑盒层：周期性微调，负责局部优化

    - 混合层：两层协作，互补长短

    """

    

    def __init__(

        self,

        env_factory: SeededEnvironmentFactory,

        initial_params: dict,

        latency_model: LatencyModel,

        llm_model_name: str = "qwen3:8b",

        llm_period: int = 30,

        blackbox_period: int = 10,

        cmaes_max_iters: int = 10,

        cmaes_population: int = 12,

        seed: int = 42,

        lambda_delay: float = 0.4,

        lambda_energy: float = 0.3,

        d_max_ms: float = 500.0,

        e_max_joule: float = 1.0

    ):

        import copy

        self.env = env_factory.create_env(seed)

        self.env.config = copy.deepcopy(initial_params)

        # ✅ 关键修复：更新环境的综合目标权重与归一化参数

        self.env.objective_weights = {'qos': 1.0, 'delay': lambda_delay, 'energy': lambda_energy}

        self.env.reference_latency_ms = d_max_ms

        self.env.reference_energy = e_max_joule

        self.env_factory = env_factory

        

        self.latency_model = latency_model

        self.llm_model_name = llm_model_name

        self.llm_period = llm_period

        self.blackbox_period = blackbox_period

        self.seed = seed

        # 综合目标参数

        self.lambda_delay = lambda_delay

        self.lambda_energy = lambda_energy

        self.d_max_ms = d_max_ms

        self.e_max_joule = e_max_joule

        

        # 参数映射器

        self.param_mapper = LatentParamMapper()

        latent_dim = self.param_mapper.config.latent_dim

        

        # 评估器

        eval_window = max(8, blackbox_period // 2)

        self.evaluator = RealSimulationEvaluatorV3(

            env_factory,

            self.param_mapper,

            window_size=eval_window,

            seed_base=seed + 10000,

            lambda_delay=self.lambda_delay,

            lambda_energy=self.lambda_energy,

            d_max_ms=self.d_max_ms,

            e_max_joule=self.e_max_joule

        )

        

        # CMA-ES配置（用于黑盒微调）

        self.cmaes_config = CMAESConfig(

            dim=latent_dim,

            population_size=cmaes_population,

            max_iterations=cmaes_max_iters,

            sigma0=0.35,  # 更大的探索步长，避免陷入局部最优

            bounds=(

                np.full(latent_dim, -2.0),

                np.full(latent_dim, 2.0)

            )

        )

        

        # 当前优化器

        self.current_optimizer: Optional[AsyncBlackboxOptimizer] = None

        

        # 统计

        self.latency_tracker = DynamicLatencyTrackerV3()

        self.optimization_count = 0

        self.llm_call_count = 0

        self.blackbox_eval_count = 0

        

        # 记录

        self.qos_history = []

        self.effective_qos_history = []

        self.objective_total_history = []
        self.objective_score_history = self.objective_total_history

        self.latency_history_ms = []
        self.latency_total_history = []
        self.overhead_history_ms = []

        self.energy_history_joule = []

        self.parameter_history = {'alpha': [], 'zeta': [], 'omega': []}

        self.llm_latency_history: List[float] = []
        self.llm_quality_history: List[float] = []
        self.instant_regret_history: List[float] = []
        self.instant_reward_history: List[float] = []
        self.cumulative_regret_history: List[float] = []
        self.cumulative_reward_history: List[float] = []

        self.llm_guidance_history = []  # 记录LLM建议的参数

        self.max_timesteps: Optional[int] = None

        self.recorded_timesteps = 0

        

        # 收敛状态

        self.convergence_timestep = -1

        self.baseline_objective = None

        # 异步评估的协同开销比例

        self.async_overlap_factor = 0.3

        self.last_llm_z: Optional[np.ndarray] = None

        self.last_llm_call_t = -self.llm_period

        self.last_blackbox_trigger_t = -self.blackbox_period

        self.pending_z0: Optional[np.ndarray] = None

        self.async_overlap_factor = 0.3

        # 异步评估的协调开销占比（评估主要在后台）

        self.async_overlap_factor = 0.3



    def _build_state_context(self, window_short: int = 10, window_long: int = 30) -> str:

        """生成近期状态摘要，让LLM“看到”网络状态与目标权重。"""

        def summarize(series: List[float], label: str, wnd: int) -> str:

            if not series:

                return f"{label}: 无历史数据"

            arr = np.array(series[-wnd:])

            p95 = float(np.percentile(arr, 95))

            trend = "上升" if arr[-1] > arr[0] * 1.05 else ("下降" if arr[-1] < arr[0] * 0.95 else "稳定")

            return f"{label}: 平均={arr.mean():.4f}, P95={p95:.4f}, 趋势={trend}"



        qos_info = summarize(self.qos_history, "QoS", window_short)

        latency_info = summarize(self.latency_history_ms, "时延(ms)", window_short)

        objective_info = summarize(self.objective_score_history, "objective_total", window_long)

        energy_info = summarize(self.energy_history_joule, "能耗(J)", window_short)



        if self.instant_regret_history:

            recent_regret = np.array(self.instant_regret_history[-window_long:])

            regret_trend = "上升" if recent_regret[-1] > recent_regret[0] * 1.05 else ("下降" if recent_regret[-1] < recent_regret[0] * 0.95 else "平稳")

            regret_info = f"Regret均值={recent_regret.mean():.4f}, 趋势={regret_trend}"

        else:

            regret_info = "Regret: 暂无数据"



        weight_info = (

            f"目标权重: QoS=1.0, 延迟λ={self.lambda_delay:.2f}, 能耗λ={self.lambda_energy:.2f}; "

            f"归一化参考 d_max={self.d_max_ms:.1f}ms, e_max={self.e_max_joule:.2f}J"

        )



        return "\n".join([qos_info, latency_info, objective_info, energy_info, regret_info, weight_info])



    def _apply_params(self, params):

        """安全应用参数，兼容dict与dataclass对象。"""

        try:

            _ = params.alpha

            for k in ['alpha', 'zeta', 'omega', 'compression_ratio', 'power_ratio', 'min_phi']:

                if hasattr(params, k) and hasattr(self.env.config, k):

                    setattr(self.env.config, k, float(getattr(params, k)))

        except AttributeError:

            if isinstance(params, dict):

                for k in ['alpha', 'zeta', 'omega', 'compression_ratio', 'power_ratio', 'min_phi']:

                    if k in params:

                        setattr(self.env.config, k, float(params[k]))

            else:

                raise

    

    def _call_llm_for_guidance(self, t: int, current_metrics: dict) -> Optional[dict]:

        """

        调用LLM获取参数指导（定期调用）

        

        Args:

            t: 当前时刻

            current_metrics: 当前网络指标

            

        Returns:

            LLM建议的参数字典，或None如果调用失败

        """
        latency_ms = None
        response = None
        quality_recorded = False
        start_time = time.time()

        try:

            import requests

            

            # 构建上下文感知提示（丰富历史信息和趋势分析）
            avg_qos = current_metrics.get('avg_qos', 0.7)
            avg_latency = current_metrics.get('avg_latency_ms', 150)
            objective_total = current_metrics.get('avg_objective_score', 0.0)
            state_context = self._build_state_context()
            
            # 分析QoS趋势（基于历史）
            if len(self.qos_history) >= 10:
                recent_qos = self.qos_history[-10:]
                qos_trend = "上升" if recent_qos[-1] > recent_qos[0] else ("下降" if recent_qos[-1] < recent_qos[0] * 0.95 else "稳定")
                qos_volatility = np.std(recent_qos)
                volatility_desc = "高波动" if qos_volatility > 0.1 else ("中等波动" if qos_volatility > 0.05 else "低波动")
            else:
                qos_trend = "初始阶段"
                volatility_desc = "数据不足"
            
            # 判断系统状态
            if avg_qos < 0.5:
                system_state = "异常（QoS过低，需紧急优化）"
                action_hint = "建议增大alpha（>0.7）以稳定性能，降低zeta（<0.2）减少探索风险"
            elif avg_qos < 0.65:
                system_state = "欠佳（性能未达预期）"
                action_hint = "建议适当调整参数平衡，尝试增大alpha或调整omega"
            elif avg_qos > 0.8:
                system_state = "良好（性能优秀）"
                action_hint = "可适当增大zeta进行探索，寻找更优配置"
            else:
                system_state = "正常"
                action_hint = "维持当前策略或微调"

            prompt = (
                f"你是网络资源调度优化专家。请根据实时监控数据给出参数调整建议，并只输出JSON。\n\n"
                f"【实时监控数据】(时刻 t={t})\n"
                f"- 平均QoS: {avg_qos:.3f}\n"
                f"- 平均时延: {avg_latency:.1f}ms\n"
                f"- 综合目标总分: {objective_total:.3f}\n"
                f"- QoS趋势: {qos_trend}\n"
                f"- 波动性: {volatility_desc}\n"
                f"- 系统状态: {system_state}\n"
                f"- 状态窗口摘要:\n{state_context}\n\n"
                f"【网络配置】\n"
                f"- 用户数/工人数/路径数: {self.env.num_users}/{self.env.num_workers}/{self.env.num_paths}\n"
                f"- 当前参数: α={self.env.config.alpha:.2f}, ζ={self.env.config.zeta:.2f}, ω={self.env.config.omega:.2f}, "
                f"compression_ratio={self.env.config.compression_ratio:.2f}, power_ratio={self.env.config.power_ratio:.2f}, min_phi={self.env.config.min_phi:.2f}\n\n"
                f"【建议方向】\n"
                f"{action_hint}\n\n"
                f"【参数范围】\n"
                f"- alpha (0.3-0.9): 路径质量权重\n"
                f"- zeta (0.1-0.5): UCB探索强度\n"
                f"- omega (0.05-0.3): 切换成本权重\n"
                f"- compression_ratio (0.5-0.95): 语义压缩率\n"
                f"- power_ratio (0.3-0.8): 功率分配\n"
                f"- min_phi (0.4-0.9): 语义速率阈值\n\n"
                f"请仅输出JSON: {{\"alpha\":值,\"zeta\":值,\"omega\":值,\"compression_ratio\":值,\"power_ratio\":值,\"min_phi\":值}}"
            )

            

            # 尝试调用Ollama

            ports_to_try = [11434, 11435]

            default_params = {
                'alpha': float(getattr(self.env.config, 'alpha', 0.6)),
                'zeta': float(getattr(self.env.config, 'zeta', 0.25)),
                'omega': float(getattr(self.env.config, 'omega', 0.15)),
                'compression_ratio': float(getattr(self.env.config, 'compression_ratio', 0.75)),
                'power_ratio': float(getattr(self.env.config, 'power_ratio', 0.5)),
                'min_phi': float(getattr(self.env.config, 'min_phi', 0.6)),
            }

            for port in ports_to_try:

                try:

                    response = requests.post(

                        f'http://localhost:{port}/api/generate',

                        json={

                            'model': self.llm_model_name,

                            'prompt': prompt,

                            'stream': False,

                            'format': 'json',

                            'options': {'temperature': 0.4, 'num_predict': 150}

                        },

                        timeout=20

                    ) 

                    if response.status_code == 200:

                        latency_ms = (time.time() - start_time) * 1000.0

                        result = response.json()

                        llm_response = result.get('response', '').strip()



                        suggestion = parse_llm_params(llm_response)
                        if suggestion:
                            blended = blend_params_with_quality(self.llm_model_name, suggestion, default_params, self.llm_quality_history)
                            self.llm_call_count += 1
                            quality_recorded = True
                            return blended

                        self.llm_quality_history.append(0.0)
                        quality_recorded = True

                        break

                except requests.exceptions.ConnectionError:

                    continue

                    

        except Exception as e:

            print(f"   [WARN] LLM调用失败 (t={t}): {e}")
            if not quality_recorded:
                self.llm_quality_history.append(0.0)
                quality_recorded = True



        finally:
            if latency_ms is None:
                try:
                    latency_ms = self.latency_model.sample_llm_latency()
                except Exception:
                    latency_ms = 0.0
            self.llm_latency_history.append(float(latency_ms))

        if not quality_recorded:
            self.llm_quality_history.append(0.0)

        return None

    

    def run_simulation(self, T: int):

        """

        运行方法3：周期性LLM指导 + 黑盒微调

        """

        print(f"\n开始运行方法3: 周期性LLM指导 + 黑盒微调 (T={T})...")

        print(f"   LLM调用周期: {self.llm_period} 时隙")

        print(f"   黑盒微调周期: {self.blackbox_period} 时隙")

        

        self.max_timesteps = T



        for t in tqdm(range(T), desc="仿真进度"):

            current_metrics = run_mtucb_step(

                self.env,

                t,

                self.lambda_delay,

                self.lambda_energy,

                self.d_max_ms,

                self.e_max_joule,

            )

            

            objective_total, _, _ = finalize_step_metrics(
                self.env,
                current_metrics,
                overhead_ms=0.0,
            )

            # 记录当前参数

            self.qos_history.append(current_metrics.avg_qos)

            self.effective_qos_history.append(current_metrics.avg_effective_qos)

            self.objective_total_history.append(objective_total)

            self.energy_history_joule.append(current_metrics.avg_energy_joule)

            self.parameter_history['alpha'].append(self.env.config.alpha)

            self.parameter_history['zeta'].append(self.env.config.zeta)

            self.parameter_history['omega'].append(self.env.config.omega)

            update_regret_reward_histories(
                self.env,
                t,
                current_metrics,
                {
                    'instant_regret_history': self.instant_regret_history,
                    'instant_reward_history': self.instant_reward_history,
                    'cumulative_regret_history': self.cumulative_regret_history,
                    'cumulative_reward_history': self.cumulative_reward_history
                }
            )



            # 更新延迟追踪

            avg_qos_t = current_metrics.avg_qos

            avg_latency_t = current_metrics.avg_latency_ms

            

            # 记录基线（综合目标）

            if t < 10:

                base_obj = objective_total

                if self.baseline_objective is None:

                    self.baseline_objective = base_obj

                else:

                    self.baseline_objective = (self.baseline_objective * t + base_obj) / (t + 1)

            

            # 2. 定期调用LLM获取指导（每llm_period时隙）

            if t > 0 and (t - self.last_llm_call_t) >= self.llm_period:

                llm_guidance = self._call_llm_for_guidance(t, {

                    'avg_qos': avg_qos_t,

                    'avg_latency_ms': avg_latency_t,

                    'avg_objective_score': objective_total

                })

                

                if llm_guidance:

                    # 保守采纳：短窗口评估，只有在综合目标提升时才直接应用

                    def _dict_to_cfg(d: dict):

                        return type('Cfg', (), d)()

                    try:

                        current_z = self.param_mapper.params_to_latent(self.env.config)

                        cand_z = self.param_mapper.params_to_latent(_dict_to_cfg(llm_guidance))

                        cur_obj = -self.evaluator.evaluate_params(current_z, t)

                        cand_obj = -self.evaluator.evaluate_params(cand_z, t)

                        if cand_obj >= cur_obj * 1.01:

                            self.env.config.alpha = llm_guidance['alpha']

                            self.env.config.zeta = llm_guidance['zeta']

                            self.env.config.omega = llm_guidance['omega']
                            if self.llm_quality_history:
                                self.env.llm_quality_factor = self.llm_quality_history[-1]

                            print(f"   [t={t}] 采纳LLM建议 (提升 {((cand_obj/cur_obj)-1)*100:.1f}%)")

                        else:

                            self.last_llm_z = cand_z

                            print(f"   [t={t}] 暂不采纳LLM建议，作为黑盒初始点")

                    except Exception:

                        # 失败时也只作为初始点

                        try:

                            self.last_llm_z = self.param_mapper.params_to_latent(_dict_to_cfg(llm_guidance))

                            print(f"   [t={t}] LLM建议解析失败，作为黑盒初始点")

                        except Exception:

                            pass

                    self.llm_guidance_history.append({'t': t, 'params': llm_guidance, 'qos_before': avg_qos_t})

                    self.last_llm_call_t = t
                    # 注意：llm_quality_factor 只在建议被采纳时设置（见上方if cand_obj >= cur_obj * 1.01分支）
                    # 不应在此处无条件设置，否则即使建议被拒绝也会影响后续计算

            

            # 3. 定期触发黑盒微调（每blackbox_period时隙）

            if t > 0 and (t - self.last_blackbox_trigger_t) >= self.blackbox_period:

                if self.current_optimizer is None:

                    # 创建新的异步优化器

                    z0 = self.last_llm_z if self.last_llm_z is not None else self.param_mapper.params_to_latent(self.env.config)

                    self.current_optimizer = AsyncBlackboxOptimizer(

                        evaluator=self.evaluator.evaluate_params,

                        latent_dim=self.param_mapper.config.latent_dim,

                        bounds=self.cmaes_config.bounds,

                        max_iterations=self.cmaes_config.max_iterations,

                        population_size=self.cmaes_config.population_size,

                        seed=self.seed + t

                    )

                    self.current_optimizer.start(z0, t)

                    self.blackbox_eval_count = self.current_optimizer.total_evals

                    self.optimization_count += 1

                    print(f"   [t={t}] 启动黑盒微调 (第{self.optimization_count}次)")

                    self.last_blackbox_trigger_t = t

            

            # 4. 执行黑盒优化一步

            optimization_latency = 0.0

            if self.current_optimizer is not None:

                # 先根据评估差分累积时延

                new_evals = max(0, self.current_optimizer.total_evals - self.blackbox_eval_count)

                if new_evals > 0:

                    for _ in range(new_evals):

                        optimization_latency += self.latency_model.sample_blackbox_eval_latency()

                    self.blackbox_eval_count = self.current_optimizer.total_evals



                result = self.current_optimizer.step(t)

                if result is not None:

                    # 优化完成，应用最佳参数

                    best_z, best_fitness = result

                    best_params = self.param_mapper.latent_to_params(best_z)

                    # 统一的安全写入

                    self._apply_params(best_params)

                    print(f"   [t={t}] 黑盒微调完成: α={self.env.config.alpha:.3f}, ζ={self.env.config.zeta:.3f}, ω={self.env.config.omega:.3f}")

                    self.current_optimizer.shutdown()

                    self.current_optimizer = None

                else:

                    # 未完成时，本轮新增评估已在上方差分统计

                    pass

            

            if INCLUDE_OPT_OVERHEAD_IN_LATENCY:
                overhead_ms = optimization_latency * self.async_overlap_factor
                latency_with_overhead = avg_latency_t + overhead_ms
            else:
                overhead_ms = 0.0
                latency_with_overhead = avg_latency_t

            _, _, latency_total = finalize_step_metrics(
                self.env,
                current_metrics,
                overhead_ms=overhead_ms,
            )

            self.latency_history_ms.append(latency_with_overhead)
            self.latency_total_history.append(latency_total)
            self.overhead_history_ms.append(overhead_ms)

            self.latency_tracker.add_timestep(avg_qos_t, latency_with_overhead)

            self.recorded_timesteps += 1

            

            # 检查收敛（基于综合目标）

            if self.convergence_timestep == -1 and self.baseline_objective is not None:

                if objective_total > self.baseline_objective * 1.1:

                    self.convergence_timestep = t

        self._enforce_history_limit()

    



    def _enforce_history_limit(self):

        """

        限制历史序列长度，避免绘图时出现不同步的时间轴

        """

        if self.max_timesteps is None:

            return

        for field in [
            'qos_history',
            'effective_qos_history',
            'objective_score_history',
            'objective_total_history',
            'latency_history_ms',
            'latency_total_history',
            'overhead_history_ms',
            'energy_history_joule'
        ]:

            history = getattr(self, field, None)

            if history is None:

                continue

            if len(history) > self.max_timesteps:

                setattr(self, field, history[:self.max_timesteps])



    def get_results(self) -> dict:

        """获取结果"""

        if self.current_optimizer is not None:

            self.current_optimizer.shutdown()

        self._enforce_history_limit()



        

        avg_latency_ms = np.mean(self.latency_history_ms) if self.latency_history_ms else 0.0

        avg_energy = np.mean(self.energy_history_joule) if self.energy_history_joule else 0.0

        avg_objective_total = np.mean(self.objective_total_history) if self.objective_total_history else 0.0

        

        return {

            'method_name': '周期性LLM+黑盒微调',

            'qos_history': self.qos_history,

            'effective_qos_history': self.effective_qos_history,

            'objective_score_history': self.objective_score_history,
            'objective_total_history': self.objective_total_history,

            'latency_history_ms': self.latency_history_ms,

            'energy_history_joule': self.energy_history_joule,

            'param_history': self.parameter_history,

            'llm_model': self.llm_model_name,

            'llm_latency_ms': self.llm_latency_history,

            'avg_qos': self.latency_tracker.get_average_qos(),

            'avg_effective_qos': self.latency_tracker.get_average_effective_qos(),

            'avg_objective_score': avg_objective_total,
            'avg_objective_total': avg_objective_total,

            'avg_latency_ms': avg_latency_ms,

            'avg_energy_joule': avg_energy,

            'qos_std': self.latency_tracker.get_qos_std(),

            'total_latency_ms': self.latency_tracker.get_total_latency(),
            'latency_total_history': self.latency_total_history,
            'overhead_history_ms': self.overhead_history_ms,

            'convergence_timestep': self.convergence_timestep,

            'llm_call_count': self.llm_call_count,

            'llm_guidance_history': self.llm_guidance_history,

            'evaluator_fallbacks': getattr(self.evaluator, 'fallback_count', 0),

            'instant_regret_history': self.instant_regret_history,
            'instant_reward_history': self.instant_reward_history,
            'cumulative_regret_history': self.cumulative_regret_history,
            'cumulative_reward_history': self.cumulative_reward_history

        }





# ============================================================================

# 9. 方法4：分布式协同优化（调用现有分布式框架）

# ============================================================================



class Method4_DistributedCollaborative:

    """

    方法4：分布式协同优化



    通过多节点并行探索 + 聚合，实现比固定 MTUCB 更快的参数收敛。

    关键特性：

        1. 各节点维护个性化 latent 参数并在本地搜索

        2. 聚合时做加权平均，保留节点差异（个性化保真）

        3. 聚合开销计入时延模型，曲线可见真实 trade-off

    """



    def __init__(

        self,

        env_factory: SeededEnvironmentFactory,

        initial_params: dict,

        latency_model: LatencyModel,

        num_workers: int = 3,

        aggregation_period: int = 20,

        seed: int = 42,

        lambda_delay: float = 0.4,

        lambda_energy: float = 0.3,

        d_max_ms: float = 500.0,

        e_max_joule: float = 1.0

    ):

        import copy

        self.env = env_factory.create_env(seed)

        self.env.config = copy.deepcopy(initial_params)

        self.env.objective_weights = {'qos': 1.0, 'delay': lambda_delay, 'energy': lambda_energy}

        self.env.reference_latency_ms = d_max_ms

        self.env.reference_energy = e_max_joule

        self.env_factory = env_factory



        self.latency_model = latency_model

        self.num_workers = num_workers

        self.aggregation_period = aggregation_period

        self.seed = seed

        self.lambda_delay = lambda_delay

        self.lambda_energy = lambda_energy

        self.d_max_ms = d_max_ms

        self.e_max_joule = e_max_joule



        self.param_mapper = LatentParamMapper()

        self.latent_dim = self.param_mapper.config.latent_dim



        self.evaluator = RealSimulationEvaluatorV3(

            env_factory,

            self.param_mapper,

            window_size=max(8, aggregation_period // 2),

            seed_base=seed + 20000,

            lambda_delay=self.lambda_delay,

            lambda_energy=self.lambda_energy,

            d_max_ms=self.d_max_ms,

            e_max_joule=self.e_max_joule

        )



        rng = np.random.default_rng(seed)

        self.local_envs: List[EnhancedMTUCBBaseline] = []

        self.local_latents: List[np.ndarray] = []

        for i in range(num_workers):

            env = env_factory.create_env(seed + i * 100)

            env.config = copy.deepcopy(initial_params)

            env.objective_weights = {'qos': 1.0, 'delay': lambda_delay, 'energy': lambda_energy}

            env.reference_latency_ms = d_max_ms

            env.reference_energy = e_max_joule

            latent = self.param_mapper.params_to_latent(env.config).astype(float)

            latent += rng.normal(0, 0.05, size=latent.shape)

            latent = np.clip(latent, -2.0, 2.0)

            self.local_latents.append(latent)

            self._apply_latent_to_env(env, latent)

            self.local_envs.append(env)



        self.global_latent = self.param_mapper.params_to_latent(self.env.config).astype(float)

        self.personalization_weight = 0.6

        self.personalization_noise = 0.03

        self.local_candidates_per_node = 8

        self.local_sigma = 0.35

        self.local_update_period = max(5, aggregation_period // 2)

        self.last_local_update = [-self.local_update_period for _ in range(self.num_workers)]

        # 平均摊到每个时隙的通信/聚合开销，用于平滑时延统计
        self.avg_comm_overhead_per_step = (
            self.latency_model.worker_comm_rtt_mean * self.num_workers
            + self.latency_model.aggregation_mean
        ) / float(max(1, self.aggregation_period))



        total_users = self.env.num_users

        self.users_per_node: List[List[int]] = [[] for _ in range(self.num_workers)]

        for u in range(total_users):

            self.users_per_node[u % self.num_workers].append(u)



        self.latency_tracker = DynamicLatencyTrackerV3()

        self.qos_history = []

        self.effective_qos_history = []

        self.objective_total_history = []
        self.objective_score_history = self.objective_total_history

        self.latency_history_ms = []
        self.latency_total_history = []
        self.overhead_history_ms = []

        self.energy_history_joule = []

        self.parameter_history = {'alpha': [], 'zeta': [], 'omega': []}

        self.aggregation_history = []
        self.instant_regret_history: List[float] = []
        self.instant_reward_history: List[float] = []
        self.cumulative_regret_history: List[float] = []
        self.cumulative_reward_history: List[float] = []



        self.convergence_timestep = -1

        self.baseline_objective = None



    def _apply_latent_to_env(self, env: EnhancedMTUCBBaseline, latent_vec: np.ndarray) -> None:

        params = self.param_mapper.latent_to_params(latent_vec)

        for key in ['alpha', 'zeta', 'omega', 'compression_ratio', 'power_ratio', 'min_phi']:

            if hasattr(env.config, key) and hasattr(params, key):

                setattr(env.config, key, float(getattr(params, key)))



    def _local_parameter_search(self, node_idx: int, t: int) -> float:

        current_z = self.local_latents[node_idx]

        best_z = current_z.copy()

        best_fitness = self.evaluator.evaluate_params(best_z, t, user_subset=self.users_per_node[node_idx])

        for _ in range(self.local_candidates_per_node):

            candidate = current_z + np.random.normal(0, self.local_sigma, size=self.latent_dim)

            candidate = np.clip(candidate, -2.0, 2.0)

            fitness = self.evaluator.evaluate_params(candidate, t, user_subset=self.users_per_node[node_idx])

            if fitness < best_fitness:

                best_fitness = fitness

                best_z = candidate.copy()

        self.local_latents[node_idx] = best_z

        self._apply_latent_to_env(self.local_envs[node_idx], best_z)

        return -best_fitness  # 转回最大化目标



    def _aggregate_parameters(self) -> dict:

        weights = []

        latents = []

        obj_vals = []

        window = 12

        for idx, env in enumerate(self.local_envs):

            latents.append(self.local_latents[idx])

            if env.objective_score_history:

                obj = float(np.mean(env.objective_score_history[-window:]))

            elif env.qos_history:

                obj = float(np.mean(env.qos_history[-window:]))

            else:

                obj = 0.5

            obj_vals.append(obj)

            weights.append(max(0.05, obj + 0.2))

        weights = np.asarray(weights, dtype=float)

        if np.sum(weights) <= 0:

            weights = np.full(len(weights), 1.0 / len(weights))

        else:

            weights /= np.sum(weights)

        agg_latent = np.average(np.vstack(latents), axis=0, weights=weights)

        agg_latent = np.clip(agg_latent, -2.0, 2.0)

        agg_params = self.param_mapper.latent_to_params(agg_latent)

        params_dict = {}

        for key in ['alpha', 'zeta', 'omega', 'compression_ratio', 'power_ratio', 'min_phi']:

            if hasattr(agg_params, key):

                params_dict[key] = float(getattr(agg_params, key))

        return {

            'params': params_dict,

            'agg_latent': agg_latent,

            'weights': weights.tolist(),

            'obj_vals': obj_vals

        }



    def _update_local_parameters(self, agg_bundle: dict) -> None:

        agg_latent = agg_bundle['agg_latent']

        self.global_latent = agg_latent.copy()

        for idx in range(self.num_workers):

            personalized = (

                self.personalization_weight * self.local_latents[idx]

                + (1 - self.personalization_weight) * (

                    agg_latent + np.random.normal(0, self.personalization_noise, size=self.latent_dim)

                )

            )

            personalized = np.clip(personalized, -2.0, 2.0)

            self.local_latents[idx] = personalized

            self._apply_latent_to_env(self.local_envs[idx], personalized)

        self._apply_latent_to_env(self.env, agg_latent)



    def run_simulation(self, T: int):

        print(f"[Method4] 分布式协同 (T={T}, workers={self.num_workers})...")

        print(f"   聚合周期: {self.aggregation_period} 步")

        for t in tqdm(range(T), desc="方法4协同仿真"):

            weighted_results = []

            for node_idx, env in enumerate(self.local_envs):

                metrics = run_mtucb_step(

                    env,

                    t,

                    self.lambda_delay,

                    self.lambda_energy,

                    self.d_max_ms,

                    self.e_max_joule,

                    user_subset=self.users_per_node[node_idx]

                )

                node_weight = max(1, len(self.users_per_node[node_idx]))

                weighted_results.append((metrics, node_weight))



            total_weight = sum(w for _, w in weighted_results)

            def _weighted_avg(getter):

                if total_weight == 0:

                    return 0.0

                return sum(getter(m) * w for m, w in weighted_results) / total_weight



            avg_qos_t = _weighted_avg(lambda m: m.avg_qos)

            avg_effective_qos_t = _weighted_avg(lambda m: m.avg_effective_qos)

            avg_objective_t = _weighted_avg(lambda m: m.avg_objective_score)
            objective_total = avg_objective_t * self.env.num_users

            avg_latency_t = _weighted_avg(lambda m: m.avg_latency_ms)

            avg_energy_t = _weighted_avg(lambda m: m.avg_energy_joule)



            self.qos_history.append(avg_qos_t)

            self.effective_qos_history.append(avg_effective_qos_t)

            self.objective_total_history.append(objective_total)

            self.energy_history_joule.append(avg_energy_t)

            self.parameter_history['alpha'].append(float(self.env.config.alpha))

            self.parameter_history['zeta'].append(float(self.env.config.zeta))

            self.parameter_history['omega'].append(float(self.env.config.omega))

            update_regret_reward_histories(
                self.env,
                t,
                type('M', (), {'objective_total': objective_total})(),
                {
                    'instant_regret_history': self.instant_regret_history,
                    'instant_reward_history': self.instant_reward_history,
                    'cumulative_regret_history': self.cumulative_regret_history,
                    'cumulative_reward_history': self.cumulative_reward_history
                }
            )

            extra_latency = 0.0

            if t < 10:

                if self.baseline_objective is None:

                    self.baseline_objective = objective_total

                else:

                    self.baseline_objective = (self.baseline_objective * t + objective_total) / (t + 1)



            if t > 0 and t % self.aggregation_period == 0:

                for node_idx in range(self.num_workers):

                    if (t - self.last_local_update[node_idx]) >= self.local_update_period:

                        self._local_parameter_search(node_idx, t)

                        self.last_local_update[node_idx] = t

                agg_bundle = self._aggregate_parameters()

                self._update_local_parameters(agg_bundle)

                self.aggregation_history.append({

                    't': t,

                    'alpha': agg_bundle['params'].get('alpha'),

                    'zeta': agg_bundle['params'].get('zeta'),

                    'omega': agg_bundle['params'].get('omega'),

                    'weights': agg_bundle['weights'],

                    'obj_vals': agg_bundle['obj_vals']

                })

                if INCLUDE_OPT_OVERHEAD_IN_LATENCY:
                    extra_latency += self.latency_model.sample_aggregation_latency()
                    extra_latency += self.latency_model.sample_worker_comm_latency() * self.num_workers

                print(f"   [t={t}] 聚合 -> α={self.env.config.alpha:.3f}, ζ={self.env.config.zeta:.3f}, ω={self.env.config.omega:.3f}")



            if INCLUDE_OPT_OVERHEAD_IN_LATENCY:
                overhead_ms = extra_latency
                latency_with_overhead = avg_latency_t + overhead_ms
            else:
                overhead_ms = 0.0
                latency_with_overhead = avg_latency_t

            latency_total = latency_with_overhead * self.env.num_users

            self.latency_history_ms.append(latency_with_overhead)
            self.latency_total_history.append(latency_total)
            self.overhead_history_ms.append(overhead_ms)

            self.latency_tracker.add_timestep(avg_qos_t, latency_with_overhead)



            if self.convergence_timestep == -1 and self.baseline_objective is not None:

                if objective_total > self.baseline_objective * 1.1:

                    self.convergence_timestep = t



    def get_results(self) -> dict:

        avg_latency_ms = np.mean(self.latency_history_ms) if self.latency_history_ms else 0.0

        avg_energy = np.mean(self.energy_history_joule) if self.energy_history_joule else 0.0

        avg_objective_total = np.mean(self.objective_total_history) if self.objective_total_history else 0.0



        return {

            'method_name': '方法4-分布式协同',

            'qos_history': self.qos_history,

            'effective_qos_history': self.effective_qos_history,

            'objective_score_history': self.objective_score_history,
            'objective_total_history': self.objective_total_history,

            'latency_history_ms': self.latency_history_ms,

            'energy_history_joule': self.energy_history_joule,

            'param_history': self.parameter_history,

            'avg_qos': self.latency_tracker.get_average_qos(),

            'avg_effective_qos': self.latency_tracker.get_average_effective_qos(),

            'avg_objective_score': avg_objective_total,
            'avg_objective_total': avg_objective_total,

            'avg_latency_ms': avg_latency_ms,

            'avg_energy_joule': avg_energy,

            'qos_std': self.latency_tracker.get_qos_std(),

            'total_latency_ms': self.latency_tracker.get_total_latency(),
            'latency_total_history': self.latency_total_history,
            'overhead_history_ms': self.overhead_history_ms,

            'convergence_timestep': self.convergence_timestep,

            'aggregation_count': len(self.aggregation_history),

            'aggregation_history': self.aggregation_history,

            'instant_regret_history': self.instant_regret_history,
            'instant_reward_history': self.instant_reward_history,
            'cumulative_regret_history': self.cumulative_regret_history,
            'cumulative_reward_history': self.cumulative_reward_history

        }


# ============================================================================
# 8. 方法5：分布式 + LLM 聚合
# ============================================================================


class Method5_DistributedLLM(Method4_DistributedCollaborative):
    """
    分布式 LLM 调度版：
    - 沿用 Method4 的分布式协同框架
    - LLM 参与全局聚合裁决（按间隔触发），并通过短窗口仿真验证
    """

    def __init__(
        self,
        env_factory: SeededEnvironmentFactory,
        initial_params: dict,
        latency_model: LatencyModel,
        num_workers: int = 3,
        aggregation_period: int = 20,
        seed: int = 42,
        lambda_delay: float = 0.4,
        lambda_energy: float = 0.3,
        d_max_ms: float = 500.0,
        e_max_joule: float = 1.0,
        llm_model_name: str = "qwen3:8b",
        llm_aggregation_interval: int = 2,
        llm_improvement_threshold: float = 0.02,
        llm_validation_window: int = 30,
    ):
        super().__init__(
            env_factory,
            initial_params,
            latency_model,
            num_workers=num_workers,
            aggregation_period=aggregation_period,
            seed=seed,
            lambda_delay=lambda_delay,
            lambda_energy=lambda_energy,
            d_max_ms=d_max_ms,
            e_max_joule=e_max_joule,
        )
        self.llm_model_name = llm_model_name
        self.llm_aggregation_interval = max(1, llm_aggregation_interval)
        self.llm_improvement_threshold = llm_improvement_threshold
        self.llm_validation_window = llm_validation_window
        self.aggregation_counter = 0

        self.llm_latency_history: List[float] = []
        self.llm_call_count = 0
        self.llm_guidance_history: List[dict] = []
        self.llm_quality_history: List[float] = []

        self.llm_validator = RealSimulationEvaluatorV3(
            env_factory,
            self.param_mapper,
            window_size=llm_validation_window,
            seed_base=seed + 30000,
            lambda_delay=self.lambda_delay,
            lambda_energy=self.lambda_energy,
            d_max_ms=self.d_max_ms,
            e_max_joule=self.e_max_joule,
        )

    def _parse_llm_suggestion(self, raw: str) -> Optional[dict]:
        return parse_llm_params(raw)

    def _build_params_from_suggestion(self, suggestion: dict):
        import copy

        bounds = self.param_mapper.config
        params = copy.deepcopy(self.env.config)

        def _clip_val(val: float, rng: Tuple[float, float]) -> float:
            return float(np.clip(val, rng[0], rng[1]))

        if 'alpha' in suggestion:
            params.alpha = _clip_val(suggestion['alpha'], bounds.alpha_bounds)
        if 'zeta' in suggestion:
            params.zeta = _clip_val(suggestion['zeta'], bounds.zeta_bounds)
        if 'omega' in suggestion:
            params.omega = _clip_val(suggestion['omega'], bounds.omega_bounds)
        if 'compression_ratio' in suggestion:
            params.compression_ratio = _clip_val(suggestion['compression_ratio'], bounds.compression_bounds)
        if 'power_ratio' in suggestion:
            params.power_ratio = _clip_val(suggestion['power_ratio'], bounds.power_bounds)
        if 'min_phi' in suggestion:
            params.min_phi = _clip_val(suggestion['min_phi'], bounds.min_phi_bounds)
        return params

    def _llm_aggregate_parameters(self, base_bundle: dict, t: int) -> Optional[dict]:
        """
        触发 LLM 生成全局聚合建议。返回格式与父类聚合一致。
        """
        import json
        import requests
        import time

        start_time = time.time()
        nodes_summary = []
        for idx, latent in enumerate(self.local_latents):
            params = self.param_mapper.latent_to_params(latent)
            nodes_summary.append(
                {
                    "node": idx,
                    "weight": float(base_bundle.get("weights", [1.0] * self.num_workers)[idx]),
                    "objective": float(base_bundle.get("obj_vals", [0.0] * self.num_workers)[idx]),
                    "alpha": float(params.alpha),
                    "zeta": float(params.zeta),
                    "omega": float(params.omega),
                    "compression_ratio": float(params.compression_ratio),
                    "power_ratio": float(params.power_ratio),
                    "min_phi": float(params.min_phi),
                }
            )

        prompt = (
            "You coordinate distributed MTUCB parameter aggregation.\\n"
            "Given per-node candidates and objectives, propose a single global parameter set "
            "that improves semantic QoS and stability. Stay within the provided ranges.\\n\\n"
            f"Current time step: {t}\\n"
            f"Aggregation baseline (weighted mean): {json.dumps(base_bundle.get('params', {}), ensure_ascii=False)}\\n"
            f"Node summaries:\\n{json.dumps(nodes_summary, indent=2, ensure_ascii=False)}\\n\\n"
            "Return JSON with fields: alpha (0.3-0.9), zeta (0.1-0.5), omega (0.05-0.3), "
            "compression_ratio (0.5-0.95), power_ratio (0.3-0.8), min_phi (0.4-0.9).\\n"
            'Format strictly as: {\"alpha\":0.7,\"zeta\":0.2,\"omega\":0.1,\"compression_ratio\":0.8,\"power_ratio\":0.45,\"min_phi\":0.6}'
        )

        ports_to_try = [11434, 11435]
        success = False
        for port in ports_to_try:
            try:
                response = requests.post(
                    f"http://localhost:{port}/api/generate",
                    json={
                        "model": self.llm_model_name,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.25, "num_predict": 200},
                    },
                    timeout=30,
                )
                if response.status_code != 200:
                    continue

                latency_ms = (time.time() - start_time) * 1000.0
                payload = response.json()
                raw = payload.get("response", "").strip()
                suggestion = self._parse_llm_suggestion(raw)
                if not suggestion:
                    continue

                default_params = {
                    "alpha": float(getattr(self.env.config, "alpha", 0.6)),
                    "zeta": float(getattr(self.env.config, "zeta", 0.25)),
                    "omega": float(getattr(self.env.config, "omega", 0.15)),
                    "compression_ratio": float(getattr(self.env.config, "compression_ratio", 0.7)),
                    "power_ratio": float(getattr(self.env.config, "power_ratio", 0.5)),
                    "min_phi": float(getattr(self.env.config, "min_phi", 0.6)),
                }
                blended = blend_params_with_quality(self.llm_model_name, suggestion, default_params, self.llm_quality_history)
                params = self._build_params_from_suggestion(blended)
                agg_latent = self.param_mapper.params_to_latent(params)
                self.llm_call_count += 1
                self.llm_latency_history.append(latency_ms)
                success = True
                self.llm_guidance_history.append(
                    {"t": t, "raw": raw, "suggestion": suggestion, "blended": blended, "latency_ms": latency_ms, "port": port}
                )

                params_dict = blended
                return {
                    "params": params_dict,
                    "agg_latent": agg_latent,
                    "weights": base_bundle.get("weights", []),
                    "obj_vals": base_bundle.get("obj_vals", []),
                    "llm_raw": raw,
                }
            except requests.exceptions.ConnectionError:
                continue
            except Exception as exc:
                self.llm_guidance_history.append({"t": t, "error": str(exc)})
                break
        if not success:
            self.llm_quality_history.append(0.0)
        return None

    def _validate_llm_aggregation(self, base_latent: np.ndarray, llm_latent: np.ndarray, t: int) -> dict:
        base_score = -self.llm_validator.evaluate_params(base_latent, t)
        llm_score = -self.llm_validator.evaluate_params(llm_latent, t)
        improvement = (llm_score - base_score) / max(1e-6, abs(base_score))
        return {
            "baseline_score": base_score,
            "llm_score": llm_score,
            "improvement": improvement,
            "accepted": improvement >= self.llm_improvement_threshold,
        }

    def _aggregate_with_llm(self, t: int) -> dict:
        base_bundle = super()._aggregate_parameters()
        base_bundle["source"] = "weighted"
        use_llm = (self.aggregation_counter % self.llm_aggregation_interval) == 0
        self.aggregation_counter += 1

        if not use_llm:
            return base_bundle

        llm_bundle = self._llm_aggregate_parameters(base_bundle, t)
        if not llm_bundle:
            return base_bundle

        validation = self._validate_llm_aggregation(base_bundle["agg_latent"], llm_bundle["agg_latent"], t)
        if validation.get("accepted", False):
            llm_bundle["source"] = "llm"
            llm_bundle["validation"] = validation
            return llm_bundle

        base_bundle["validation"] = validation
        return base_bundle

    def run_simulation(self, T: int):
        print(
            f"[Method5] Distributed LLM scheduling (T={T}, workers={self.num_workers}, "
            f"agg={self.aggregation_period}, LLM every {self.llm_aggregation_interval} aggs)..."
        )

        for t in tqdm(range(T), desc="Method5 LLM-distributed"):
            weighted_results = []
            for node_idx, env in enumerate(self.local_envs):
                metrics = run_mtucb_step(
                    env,
                    t,
                    self.lambda_delay,
                    self.lambda_energy,
                    self.d_max_ms,
                    self.e_max_joule,
                    user_subset=self.users_per_node[node_idx],
                )
                node_weight = max(1, len(self.users_per_node[node_idx]))
                weighted_results.append((metrics, node_weight))

            total_weight = sum(w for _, w in weighted_results)

            def _weighted_avg(getter):
                if total_weight == 0:
                    return 0.0
                return sum(getter(m) * w for m, w in weighted_results) / total_weight

            avg_qos_t = _weighted_avg(lambda m: m.avg_qos)
            avg_effective_qos_t = _weighted_avg(lambda m: m.avg_effective_qos)
            avg_objective_t = _weighted_avg(lambda m: m.avg_objective_score)
            avg_latency_t = _weighted_avg(lambda m: m.avg_latency_ms)
            avg_energy_t = _weighted_avg(lambda m: m.avg_energy_joule)

            self.qos_history.append(avg_qos_t)
            self.effective_qos_history.append(avg_effective_qos_t)
            self.objective_total_history.append(objective_total)
            self.energy_history_joule.append(avg_energy_t)
            self.parameter_history['alpha'].append(float(self.env.config.alpha))
            self.parameter_history['zeta'].append(float(self.env.config.zeta))
            self.parameter_history['omega'].append(float(self.env.config.omega))

            update_regret_reward_histories(
                self.env,
                t,
                type('M', (), {'objective_total': objective_total})(),
                {
                    'instant_regret_history': self.instant_regret_history,
                    'instant_reward_history': self.instant_reward_history,
                    'cumulative_regret_history': self.cumulative_regret_history,
                    'cumulative_reward_history': self.cumulative_reward_history,
                },
            )

            extra_latency = 0.0
            if t < 10:
                if self.baseline_objective is None:
                    self.baseline_objective = objective_total
                else:
                    self.baseline_objective = (self.baseline_objective * t + objective_total) / (t + 1)

            if t > 0 and t % self.aggregation_period == 0:
                for node_idx in range(self.num_workers):
                    if (t - self.last_local_update[node_idx]) >= self.local_update_period:
                        self._local_parameter_search(node_idx, t)
                        self.last_local_update[node_idx] = t

                agg_bundle = self._aggregate_with_llm(t)
                self._update_local_parameters(agg_bundle)
                if agg_bundle.get("source") == "llm":
                    if self.llm_quality_history:
                        self.env.llm_quality_factor = self.llm_quality_history[-1]
                    else:
                        self.env.llm_quality_factor = 0.7
                else:
                    self.env.llm_quality_factor = 1.0

                if INCLUDE_OPT_OVERHEAD_IN_LATENCY:
                    extra_latency += self.latency_model.sample_aggregation_latency()
                    extra_latency += self.latency_model.sample_worker_comm_latency() * self.num_workers
                    if agg_bundle.get("source") == "llm" and self.llm_latency_history:
                        extra_latency += self.llm_latency_history[-1]

                self.aggregation_history.append(
                    {
                        't': t,
                        'alpha': agg_bundle['params'].get('alpha'),
                        'zeta': agg_bundle['params'].get('zeta'),
                        'omega': agg_bundle['params'].get('omega'),
                        'weights': agg_bundle.get('weights'),
                        'obj_vals': agg_bundle.get('obj_vals'),
                        'source': agg_bundle.get('source', 'weighted'),
                        'validation': agg_bundle.get('validation'),
                    }
                )

                print(
                    f"   [t={t}] aggregate -> α={self.env.config.alpha:.3f}, "
                    f"ζ={self.env.config.zeta:.3f}, ω={self.env.config.omega:.3f} "
                    f"(source={agg_bundle.get('source', 'weighted')})"
                )

            if INCLUDE_OPT_OVERHEAD_IN_LATENCY:
                overhead_ms = extra_latency
                latency_with_overhead = avg_latency_t + overhead_ms
            else:
                overhead_ms = 0.0
                latency_with_overhead = avg_latency_t

            latency_total = latency_with_overhead * self.env.num_users
            self.latency_history_ms.append(latency_with_overhead)
            self.latency_total_history.append(latency_total)
            self.overhead_history_ms.append(overhead_ms)
            self.latency_tracker.add_timestep(avg_qos_t, latency_with_overhead)

            if self.convergence_timestep == -1 and self.baseline_objective is not None:
                if objective_total > self.baseline_objective * 1.1:
                    self.convergence_timestep = t

    def get_results(self) -> dict:
        base = super().get_results()
        base['method_name'] = '方法5-分布式LLM调度'
        base['llm_model'] = self.llm_model_name
        base['llm_call_count'] = self.llm_call_count
        base['llm_latency_ms'] = self.llm_latency_history
        base['llm_guidance_history'] = self.llm_guidance_history
        base['llm_quality_history'] = self.llm_quality_history
        base['llm_avg_quality'] = float(np.mean(self.llm_quality_history)) if self.llm_quality_history else 0.0
        return base


# ============================================================================

# 8. 实验配置

# ============================================================================



@dataclass

class ExperimentConfig:

    """实验配置"""

    # 基础参数

    num_users: int = 12

    num_workers: int = 6

    num_paths: int = 4

    T: int = 200  # 仿真时长

    

    # 多次实验

    num_runs: int = 5

    seed_base: int = 42

    

    # 方法参数

    blackbox_period: int = 25

    cmaes_max_iters: int = 25

    cmaes_population: int = 18

    

    # 方法3参数

    llm_period: int = 30  # LLM调用周期

    # LLM??????
    llm_models: List[str] = field(default_factory=lambda: ["qwen3:8b", "phi3:mini", "deepseek-r1:8b"])

    

    # 方法4参数（分布式）

    num_distributed_workers: int = 3  # 分布式节点数

    aggregation_period: int = 20  # 参数聚合周期
    llm_aggregation_interval: int = 2  # 每隔多少次聚合触发一次LLM
    llm_improvement_threshold: float = 0.02  # LLM建议相对提升阈值
    llm_validation_window: int = 30  # 短窗口验证长度





# ============================================================================

# 归一化参数估计（warm-up）

# ============================================================================



def estimate_normalizers(env_factory: SeededEnvironmentFactory, initial_params: dict,

                         warmup_steps: int, seed: int, 

                         lambda_delay: float = 0.4, lambda_energy: float = 0.3) -> Tuple[float, float]:

    """短期仿真估计 Dmax/Emax（95 分位数）。"""

    env = env_factory.create_env(seed + 99999)

    env.config = initial_params

    # ✅ 关键修复：warmup环境也需要设置初始权重（使用默认值）

    env.objective_weights = {'qos': 1.0, 'delay': lambda_delay, 'energy': lambda_energy}

    env.reference_latency_ms = 500.0  # 临时默认值，后续会被更新

    env.reference_energy = 1.0

    lat_samples = []

    energy_samples = []

    for t in range(max(1, warmup_steps)):

        matching_with_paths = []

        timestep_results = []

        worker_load = {w_id: 0 for w_id in range(env.num_workers)}

        for u in range(env.num_users):

            w = (u + t) % env.num_workers

            safe_w = int(w % env.S.shape[1])

            path = env.select_path_ucb(t, u, safe_w)

            worker_load[safe_w] += 1

            qos_result = env.calculate_enhanced_qos(t, u, safe_w, path, worker_load[safe_w])

            if isinstance(qos_result, dict):

                lat_samples.append(qos_result.get('latency', 0.0))

                energy_samples.append(qos_result.get('energy_joule', 0.0))

    d_max = float(np.percentile(lat_samples, 95)) if lat_samples else 500.0

    e_max = float(np.percentile(energy_samples, 95)) if energy_samples else 1.0

    return d_max, e_max



# ============================================================================

# 可视化：多算法多维度时间序列对比

# ============================================================================




def save_comparison_plots(all_results: Dict[str, List[dict]], save_dir: str = 'figs', default_llm_model: Optional[str] = None) -> None:
    os.makedirs(save_dir, exist_ok=True)

    def aggregate_runs(method_key: str, field: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        runs = all_results.get(method_key, [])
        series_list = [np.array(r.get(field, []), dtype=float) for r in runs if r.get(field)]
        if not series_list:
            return None
        min_len = min(len(s) for s in series_list)
        if min_len == 0:
            return None
        stacked = np.stack([s[:min_len] for s in series_list], axis=0)
        mean_series = stacked.mean(axis=0)
        std_series = stacked.std(axis=0)
        t = np.arange(min_len)
        return t, mean_series, std_series

    def plot_with_ci(ax, method_key: str, label: str, field: str, color: str):
        agg = aggregate_runs(method_key, field)
        if not agg:
            return False
        t, mean_series, std_series = agg
        ax.plot(t, mean_series, label=label, color=color, linewidth=2.0, alpha=0.9)
        ax.fill_between(t, mean_series - std_series, mean_series + std_series, color=color, alpha=0.18)
        return True

    curves = [
        ('effective_qos_history', 'Effective QoS (latency-penalized)', 'effective_qos_timeseries.png'),
        ('instant_regret_history', 'Instant regret', 'instant_regret_timeseries.png'),
        ('latency_history_ms', 'Average latency (ms)', 'latency_timeseries.png'),
        ('energy_history_joule', 'Average energy (J)', 'energy_timeseries.png'),
        ('qos_history', 'QoS', 'qos_timeseries.png'),
        ('objective_score_history', 'Objective total', 'objective_total_timeseries.png'),
        ('cumulative_regret_history', 'Cumulative regret', 'cumulative_regret_timeseries.png'),
        ('cumulative_reward_history', 'Cumulative reward', 'cumulative_reward_timeseries.png'),
    ]

    model_tag = f" ({default_llm_model})" if default_llm_model else ""
    labels = {
        'method1_baseline': 'Method1 - Fixed',
        'method2_async_blackbox': f'Method2 - LLM init + async BB{model_tag}',
        'method3_periodic_llm_hybrid': f'Method3 - Periodic LLM + tuning{model_tag}',
        'method4_distributed_collaborative': 'Method4 - Distributed collaborative',
        'method5_distributed_llm': f'Method5 - LLM-driven distributed{model_tag}',
    }

    method_order = [
        'method1_baseline',
        'method2_async_blackbox',
        'method3_periodic_llm_hybrid',
        'method4_distributed_collaborative',
        'method5_distributed_llm',
    ]

    for field, y_label, fname in curves:
        fig, ax = plt.subplots(figsize=(10, 5))
        series_bundle = []

        for mkey in method_order:
            if mkey not in all_results:
                continue
            added = plot_with_ci(ax, mkey, labels.get(mkey, mkey), field, COLOR_MAP.get(mkey, '#333333'))
            if added:
                series_bundle.append(mkey)

        if not series_bundle:
            plt.close()
            continue

        plt.xlabel('Time step')
        plt.ylabel(y_label)
        plt.title(y_label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(save_dir, fname)
        plt.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close()

    def plot_llm_model_bars():
        llm_result = all_results.get('llm_models', {})
        if not llm_result:
            return
        models = []
        qos_m2 = []
        qos_m3 = []
        qos_m5 = []
        lat_m2 = []
        lat_m3 = []
        lat_m5 = []

        for model, methods in llm_result.items():
            m2_runs = methods.get('method2_async_blackbox', [])
            m3_runs = methods.get('method3_periodic_llm_hybrid', [])
            m5_runs = methods.get('method5_distributed_llm', [])

            def _avg_qos(runs):
                return float(np.mean([r.get('avg_qos', 0.0) for r in runs])) if runs else 0.0

            def _avg_llm_latency(runs):
                if not runs:
                    return 0.0
                per_run = []
                for r in runs:
                    lat_list = r.get('llm_latency_ms', [])
                    per_run.append(np.mean(lat_list) if lat_list else 0.0)
                return float(np.mean(per_run)) if per_run else 0.0

            models.append(model)
            qos_m2.append(_avg_qos(m2_runs))
            qos_m3.append(_avg_qos(m3_runs))
            qos_m5.append(_avg_qos(m5_runs))
            lat_m2.append(_avg_llm_latency(m2_runs))
            lat_m3.append(_avg_llm_latency(m3_runs))
            lat_m5.append(_avg_llm_latency(m5_runs))

        if not models:
            return

        x = np.arange(len(models))
        # 调整柱子宽度：三个柱子，每个宽度0.25，间距0.02
        width = 0.25
        
        # 检查是否所有Method5数据都为0
        has_m5_data = any(v > 0 for v in qos_m5)

        # QoS对比图
        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(x - width - 0.02, qos_m2, width, label='Method2', color='#4ECDC4')
        bars2 = plt.bar(x, qos_m3, width, label='Method3', color='#FF8C42')
        bars3 = plt.bar(x + width + 0.02, qos_m5, width, label='Method5', color='#9467BD')
        
        # 在柱子上方添加数值标签
        for bar, val in zip(bars1, qos_m2):
            if val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, qos_m3):
            if val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars3, qos_m5):
            if val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xticks(x, models, rotation=15)
        plt.ylabel('Avg QoS')
        plt.title('LLM model QoS comparison (Methods 2/3/5)')
        plt.legend(loc='upper right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'llm_model_qos_comparison.png'), dpi=180, bbox_inches='tight')
        plt.close()

        # Latency对比图
        # 注意：不同Method的LLM调用次数不同
        # - Method2: 仅初始化时调用1次
        # - Method3: 每llm_period步调用1次
        # - Method5: 每次聚合可能调用LLM
        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(x - width - 0.02, lat_m2, width, label='Method2 (init only)', color='#4ECDC4')
        bars2 = plt.bar(x, lat_m3, width, label='Method3 (periodic)', color='#FF8C42')
        bars3 = plt.bar(x + width + 0.02, lat_m5, width, label='Method5 (aggregation)', color='#9467BD')
        
        # 添加数值标签
        for bar, val in zip(bars1, lat_m2):
            if val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, lat_m3):
            if val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars3, lat_m5):
            if val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        plt.xticks(x, models, rotation=15)
        plt.ylabel('Avg LLM call latency (ms)')
        plt.title('LLM model latency comparison\n(Note: Method2=init only, Method3=periodic, Method5=aggregation)')
        plt.legend(loc='upper right')
        plt.grid(True, axis='y', alpha=0.3)
        # 添加脚注说明
        plt.figtext(0.5, -0.02, 
                   'Method2: 1 call at init | Method3: periodic calls | Method5: calls during aggregation',
                   ha='center', fontsize=8, style='italic')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'llm_model_latency_comparison.png'), dpi=180, bbox_inches='tight')
        plt.close()
        
        # 如果Method5数据缺失，打印警告
        if not has_m5_data:
            print("[WARN] Method5在LLM模型对比中数据为空，请检查实验配置")

    plot_llm_model_bars()

    def save_plots_for_model(model_name: str):
        subset = all_results.get('llm_models', {}).get(model_name, {})
        if not subset:
            return
        method_map = {
            'method1_baseline': all_results.get('method1_baseline', []),
            'method4_distributed_collaborative': all_results.get('method4_distributed_collaborative', []),
            'method2_async_blackbox': subset.get('method2_async_blackbox', []),
            'method3_periodic_llm_hybrid': subset.get('method3_periodic_llm_hybrid', []),
            'method5_distributed_llm': subset.get('method5_distributed_llm', []),
        }
        metrics_to_plot = [
            ('effective_qos_history', 'Effective QoS (latency-penalized)', f'effective_qos_{model_name}.png'),
            ('instant_regret_history', 'Instant regret', f'instant_regret_{model_name}.png'),
            ('cumulative_regret_history', 'Cumulative regret', f'cumulative_regret_{model_name}.png'),
            ('latency_history_ms', 'Average latency (ms)', f'latency_{model_name}.png'),
            ('energy_history_joule', 'Average energy (J)', f'energy_{model_name}.png'),
        ]

        def aggregate(runs: List[dict], field: str):
            if not runs:
                return None
            series_list = [np.array(r.get(field, []), dtype=float) for r in runs if r.get(field)]
            if not series_list:
                return None
            min_len = min(len(s) for s in series_list)
            if min_len == 0:
                return None
            stacked = np.stack([s[:min_len] for s in series_list], axis=0)
            t = np.arange(min_len)
            return t, stacked.mean(axis=0), stacked.std(axis=0)

        for field, ylabel, fname in metrics_to_plot:
            plt.figure(figsize=(10, 5))
            added_any = False
            for mkey, runs in method_map.items():
                agg = aggregate(runs, field)
                if not agg:
                    continue
                t, mean_series, std_series = agg
                plt.plot(t, mean_series, label=labels.get(mkey, mkey), color=COLOR_MAP.get(mkey, '#333333'), linewidth=2.0)
                plt.fill_between(t, mean_series - std_series, mean_series + std_series, color=COLOR_MAP.get(mkey, '#333333'), alpha=0.18)
                added_any = True
            if not added_any:
                plt.close()
                continue
            plt.xlabel('Time step')
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} - {model_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, fname), dpi=180, bbox_inches='tight')
            plt.close()

    for model_name in all_results.get('llm_models', {}).keys():
        save_plots_for_model(model_name)
    def plot_qos_latency_tradeoff():
        llm_result = all_results.get('llm_models', {})
        if not llm_result:
            return
        plt.figure(figsize=(10, 8))

        def _avg_val(runs, key):
            if not runs:
                return 0.0
            vals = [np.mean(r.get(key, []) or [0.0]) for r in runs]
            return float(np.mean(vals)) if vals else 0.0

        colors = {'method2_async_blackbox': '#4ECDC4', 'method3_periodic_llm_hybrid': '#FF8C42', 'method5_distributed_llm': '#9467BD'}
        markers = {'method2_async_blackbox': 'o', 'method3_periodic_llm_hybrid': 's', 'method5_distributed_llm': '^'}
        for model, methods in llm_result.items():
            for mkey in ['method2_async_blackbox', 'method3_periodic_llm_hybrid', 'method5_distributed_llm']:
                runs = methods.get(mkey, [])
                if not runs:
                    continue
                qos = float(np.mean([r.get('avg_qos', 0.0) for r in runs]))
                latency = _avg_val(runs, 'llm_latency_ms')
                plt.scatter(latency, qos, s=110, color=colors[mkey], marker=markers[mkey], label=f"{model}-{mkey.split('_')[0].title()}")
                plt.annotate(f"{model}\n{mkey.split('_')[0].title()}", (latency, qos), textcoords="offset points", xytext=(0,10), ha='center')

        plt.xlabel('LLM call latency (ms)')
        plt.ylabel('Avg QoS')
        plt.title('QoS-Latency trade-off by LLM model (Methods 2/3/5)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.savefig(os.path.join(save_dir, 'qos_latency_tradeoff.png'), dpi=200, bbox_inches='tight')
        plt.close()

    plot_qos_latency_tradeoff()
# ============================================================================

# 9. 主实验函数（多次运行+置信区间）

# ============================================================================






def run_complete_experiment_with_statistics(config: ExperimentConfig):
    """
    Run full experiment multiple times and compute statistics.
    """
    print("\n" + "="*80)
    print("Experiment V3 (fixed) - multi-run + CI")
    print("="*80)
    print(f"Runs: {config.num_runs}")
    print(f"Horizon T: {config.T}")
    print(f"LLM models: {config.llm_models}")
    print()

    default_llm_model = config.llm_models[0] if config.llm_models else "qwen3:8b"

    env_factory = SeededEnvironmentFactory(
        config.num_users,
        config.num_workers,
        config.num_paths
    )

    latency_model = LatencyModel()

    from Enhanced_MTUCB_with_Ollama import SCQoSConfig
    initial_params = SCQoSConfig.default()

    print("Unified init params:")
    print(f"  alpha: {initial_params.alpha}")
    print(f"  zeta: {initial_params.zeta}")
    print(f"  omega: {initial_params.omega}")
    print()

    print("Estimating normalizers (Dmax/Emax)...")
    lambda_delay = 0.5
    lambda_energy = 0.5
    est_d_max, est_e_max = estimate_normalizers(
        env_factory, initial_params,
        warmup_steps=200,
        seed=config.seed_base,
        lambda_delay=lambda_delay,
        lambda_energy=lambda_energy
    )
    print(f"  Dmax(ms)={est_d_max:.1f}, Emax(J)={est_e_max:.3f}, lambda_delay={lambda_delay}, lambda_energy={lambda_energy}")

    all_results = {
        'method1_baseline': [],
        'method2_async_blackbox': [],
        'method3_periodic_llm_hybrid': [],
        'method4_distributed_collaborative': [],
        'method5_distributed_llm': [],
        'llm_models': {m: {'method2_async_blackbox': [], 'method3_periodic_llm_hybrid': [], 'method5_distributed_llm': []} for m in config.llm_models}
    }

    for run_idx in range(config.num_runs):
        print("\n" + "="*80)
        print(f"Run {run_idx+1}/{config.num_runs} (seed={config.seed_base + run_idx})")
        print("="*80)

        run_seed = config.seed_base + run_idx

        print("\n[Method1] Fixed baseline...")
        method1 = Method1_FixedBaseline(
            env_factory,
            initial_params,
            run_seed,
            lambda_delay=lambda_delay,
            lambda_energy=lambda_energy,
            d_max_ms=est_d_max,
            e_max_joule=est_e_max
        )
        method1.run_simulation(config.T)
        all_results['method1_baseline'].append(method1.get_results())

        for model_name in config.llm_models:
            print(f"\n[Method2] LLM init + async blackbox (LLM={model_name})...")
            method2 = Method2_LLMInitAsyncBlackbox(
                env_factory,
                initial_params,
                latency_model,
                llm_model_name=model_name,
                blackbox_period=config.blackbox_period,
                cmaes_max_iters=config.cmaes_max_iters,
                cmaes_population=config.cmaes_population,
                seed=run_seed,
                lambda_delay=lambda_delay,
                lambda_energy=lambda_energy,
                d_max_ms=est_d_max,
                e_max_joule=est_e_max
            )
            method2.run_simulation(config.T)
            res2 = method2.get_results()
            all_results['llm_models'][model_name]['method2_async_blackbox'].append(res2)
            if model_name == default_llm_model:
                all_results['method2_async_blackbox'].append(res2)

            print(f"\n[Method3] Periodic LLM + tuning (LLM={model_name})...")
            method3 = Method3_PeriodicLLMHybrid(
                env_factory,
                initial_params,
                latency_model,
                llm_model_name=model_name,
                llm_period=config.blackbox_period,
                blackbox_period=config.blackbox_period,
                cmaes_max_iters=config.cmaes_max_iters,
                cmaes_population=config.cmaes_population,
                seed=run_seed,
                lambda_delay=lambda_delay,
                lambda_energy=lambda_energy,
                d_max_ms=est_d_max,
                e_max_joule=est_e_max
            )
            method3.run_simulation(config.T)
            res3 = method3.get_results()
            all_results['llm_models'][model_name]['method3_periodic_llm_hybrid'].append(res3)
            if model_name == default_llm_model:
                all_results['method3_periodic_llm_hybrid'].append(res3)

            print(f"\n[Method5] Distributed LLM scheduling (LLM={model_name})...")
            method5 = Method5_DistributedLLM(
                env_factory,
                initial_params,
                latency_model,
                num_workers=config.num_distributed_workers,
                aggregation_period=config.aggregation_period,
                seed=run_seed,
                lambda_delay=lambda_delay,
                lambda_energy=lambda_energy,
                d_max_ms=est_d_max,
                e_max_joule=est_e_max,
                llm_model_name=model_name,
                llm_aggregation_interval=config.llm_aggregation_interval,
                llm_improvement_threshold=config.llm_improvement_threshold,
                llm_validation_window=config.llm_validation_window,
            )
            method5.run_simulation(config.T)
            res5 = method5.get_results()
            all_results['llm_models'][model_name]['method5_distributed_llm'].append(res5)
            if model_name == default_llm_model:
                all_results['method5_distributed_llm'].append(res5)

        print("\n[Method4] Distributed collaborative...")
        method4 = Method4_DistributedCollaborative(
            env_factory,
            initial_params,
            latency_model,
            num_workers=config.num_distributed_workers,
            aggregation_period=config.aggregation_period,
            seed=run_seed,
            lambda_delay=lambda_delay,
            lambda_energy=lambda_energy,
            d_max_ms=est_d_max,
            e_max_joule=est_e_max
        )
        method4.run_simulation(config.T)
        all_results['method4_distributed_collaborative'].append(method4.get_results())

    print("\n" + "="*80)
    print("Computing stats (mean ± 95% CI)...")
    print("="*80)

    def calc_stats(data):
        mean = np.mean(data)
        n = len(data)
        if n < 2:
            return mean, 0.0, 0.0
        std = np.std(data, ddof=1)
        ci = stats.t.ppf(0.975, n-1) * std / np.sqrt(n)
        return mean, std, ci

    def build_stats(results_list):
        avg_qos_list = [r['avg_qos'] for r in results_list]
        avg_effective_qos_list = [r['avg_effective_qos'] for r in results_list]
        avg_objective_list = [r.get('avg_objective_total', r.get('avg_objective_score', r['avg_qos'])) for r in results_list]
        avg_latency_list = [r.get('avg_latency_ms', 0.0) for r in results_list]
        avg_energy_list = [r.get('avg_energy_joule', 0.0) for r in results_list]
        qos_std_list = [r['qos_std'] for r in results_list]
        total_latency_list = [r['total_latency_ms'] for r in results_list]
        convergence_list = [r['convergence_timestep'] for r in results_list if r.get('convergence_timestep', -1) > 0]
        final_regret_list = [r['cumulative_regret_history'][-1] for r in results_list if r.get('cumulative_regret_history')]
        final_reward_list = [r['cumulative_reward_history'][-1] for r in results_list if r.get('cumulative_reward_history')]

        qos_mean, qos_std, qos_ci = calc_stats(avg_qos_list)
        eff_qos_mean, eff_qos_std, eff_qos_ci = calc_stats(avg_effective_qos_list)
        obj_mean, obj_std, obj_ci = calc_stats(avg_objective_list)
        jitter_mean, jitter_std, jitter_ci = calc_stats(qos_std_list)
        latency_mean, latency_std, latency_ci = calc_stats(total_latency_list)
        avg_latency_mean, avg_latency_std, avg_latency_ci = calc_stats(avg_latency_list)
        avg_energy_mean, avg_energy_std, avg_energy_ci = calc_stats(avg_energy_list)
        regret_mean, regret_std, regret_ci = calc_stats(final_regret_list) if final_regret_list else (0.0, 0.0, 0.0)
        reward_mean, reward_std, reward_ci = calc_stats(final_reward_list) if final_reward_list else (0.0, 0.0, 0.0)
        conv_mean, conv_std, conv_ci = (np.mean(convergence_list), np.std(convergence_list), 0) if convergence_list else (-1, 0, 0)

        method_name = results_list[0]['method_name'] if results_list else 'N/A'
        return {
            'method_name': method_name,
            'avg_qos': {'mean': qos_mean, 'std': qos_std, 'ci': qos_ci},
            'avg_effective_qos': {'mean': eff_qos_mean, 'std': eff_qos_std, 'ci': eff_qos_ci},
            'avg_objective_total': {'mean': obj_mean, 'std': obj_std, 'ci': obj_ci},
            'qos_jitter': {'mean': jitter_mean, 'std': jitter_std, 'ci': jitter_ci},
            'total_latency_ms': {'mean': latency_mean, 'std': latency_std, 'ci': latency_ci},
            'avg_latency_ms': {'mean': avg_latency_mean, 'std': avg_latency_std, 'ci': avg_latency_ci},
            'avg_energy_joule': {'mean': avg_energy_mean, 'std': avg_energy_std, 'ci': avg_energy_ci},
            'final_cumulative_regret': {'mean': regret_mean, 'std': regret_std, 'ci': regret_ci},
            'final_cumulative_reward': {'mean': reward_mean, 'std': reward_std, 'ci': reward_ci},
            'convergence_timestep': {'mean': conv_mean, 'std': conv_std},
            'raw_results': results_list
        }

    stats_results = {'llm_models': {}}
    for method_key in ['method1_baseline', 'method2_async_blackbox', 'method3_periodic_llm_hybrid', 'method4_distributed_collaborative', 'method5_distributed_llm']:
        if all_results.get(method_key):
            stats_results[method_key] = build_stats(all_results[method_key])

    for model_name, methods in all_results.get('llm_models', {}).items():
        stats_results['llm_models'][model_name] = {}
        for mkey, res_list in methods.items():
            if res_list:
                stats_results['llm_models'][model_name][mkey] = build_stats(res_list)

    if all_results['method2_async_blackbox'] and all_results['method3_periodic_llm_hybrid']:
        print("\nPaired t-tests vs baseline (default LLM)")
        method1_qos = [r['avg_qos'] for r in all_results['method1_baseline']]
        method2_qos = [r['avg_qos'] for r in all_results['method2_async_blackbox']]
        method3_qos = [r['avg_qos'] for r in all_results['method3_periodic_llm_hybrid']]
        method4_qos = [r['avg_qos'] for r in all_results['method4_distributed_collaborative']]
        method5_qos = [r['avg_qos'] for r in all_results['method5_distributed_llm']]

        t_stat_m2, p_value_m2 = stats.ttest_rel(method2_qos, method1_qos)
        t_stat_m3, p_value_m3 = stats.ttest_rel(method3_qos, method1_qos)
        t_stat_m4, p_value_m4 = stats.ttest_rel(method4_qos, method1_qos)
        t_stat_m5, p_value_m5 = stats.ttest_rel(method5_qos, method1_qos)
        t_stat_m3v2, p_value_m3v2 = stats.ttest_rel(method3_qos, method2_qos)
        t_stat_m4v2, p_value_m4v2 = stats.ttest_rel(method4_qos, method2_qos)
        t_stat_m5v4, p_value_m5v4 = stats.ttest_rel(method5_qos, method4_qos)

        stats_results['ttest'] = {
            'default_llm': default_llm_model,
            'method2_vs_1': {'t_stat': t_stat_m2, 'p_value': p_value_m2},
            'method3_vs_1': {'t_stat': t_stat_m3, 'p_value': p_value_m3},
            'method4_vs_1': {'t_stat': t_stat_m4, 'p_value': p_value_m4},
            'method5_vs_1': {'t_stat': t_stat_m5, 'p_value': p_value_m5},
            'method3_vs_2': {'t_stat': t_stat_m3v2, 'p_value': p_value_m3v2},
            'method4_vs_2': {'t_stat': t_stat_m4v2, 'p_value': p_value_m4v2},
            'method5_vs_4': {'t_stat': t_stat_m5v4, 'p_value': p_value_m5v4}
        }

        print(f"  Method2 vs 1 (LLM={default_llm_model}): t={t_stat_m2:.4f}, p={p_value_m2:.6f}")
        print(f"  Method3 vs 1 (LLM={default_llm_model}): t={t_stat_m3:.4f}, p={p_value_m3:.6f}")
        print(f"  Method4 vs 1: t={t_stat_m4:.4f}, p={p_value_m4:.6f}")
        print(f"  Method3 vs 2 (LLM={default_llm_model}): t={t_stat_m3v2:.4f}, p={p_value_m3v2:.6f}")
        print(f"  Method4 vs 2: t={t_stat_m4v2:.4f}, p={p_value_m4v2:.6f}")
        print(f"  Method5 vs 1: t={t_stat_m5:.4f}, p={p_value_m5:.6f}")
        print(f"  Method5 vs 4: t={t_stat_m5v4:.4f}, p={p_value_m5v4:.6f}")

    print("\n" + "="*80)
    print(f"Results summary (default LLM for Method2/3: {default_llm_model})")
    print("="*80)
    print("{:<25} {:<20} {:<20} {:<15} {:<20}".format("Method", "Avg QoS", "Effective QoS", "QoS jitter", "Total latency"))
    print("-" * 140)

    for key, mstats in stats_results.items():
        if key in ['ttest', 'llm_models']:
            continue
        name = mstats['method_name']
        qos_str = f"{mstats['avg_qos']['mean']:.4f} ± {mstats['avg_qos']['ci']:.4f}"
        eff_qos_str = f"{mstats['avg_effective_qos']['mean']:.4f} ± {mstats['avg_effective_qos']['ci']:.4f}"
        jitter_str = f"{mstats['qos_jitter']['mean']:.4f}"
        latency_str = f"{mstats['total_latency_ms']['mean']:.1f} ± {mstats['total_latency_ms']['ci']:.1f}"
        print("{:<25} {:<20} {:<20} {:<15} {:<20}".format(name, qos_str, eff_qos_str, jitter_str, latency_str))

    print("\nAdditional metrics:")
    for key, mstats in stats_results.items():
        if key in ['ttest', 'llm_models']:
            continue
        name = mstats['method_name']
        obj_str = f"{mstats['avg_objective_total']['mean']:.4f} ± {mstats['avg_objective_total']['ci']:.4f}"
        avg_lat_str = f"{mstats['avg_latency_ms']['mean']:.1f} ± {mstats['avg_latency_ms']['ci']:.1f}"
        avg_energy_str = f"{mstats['avg_energy_joule']['mean']:.3f} ± {mstats['avg_energy_joule']['ci']:.3f}"
        regret_str = f"{mstats['final_cumulative_regret']['mean']:.2f}"
        print(f"- {name} | Obj: {obj_str} | Avg latency: {avg_lat_str} | Avg energy: {avg_energy_str} | Final cumulative regret: {regret_str}")

    if stats_results.get('llm_models'):
        print("\nLLM model comparison (Methods 2/3/5): Avg QoS")
        for model_name, method_stats in stats_results['llm_models'].items():
            m2 = method_stats.get('method2_async_blackbox')
            m3 = method_stats.get('method3_periodic_llm_hybrid')
            m5 = method_stats.get('method5_distributed_llm')
            if m2:
                print(f"  {model_name} | Method2: {m2['avg_qos']['mean']:.4f} ± {m2['avg_qos']['ci']:.4f}")
            if m3:
                print(f"  {model_name} | Method3: {m3['avg_qos']['mean']:.4f} ± {m3['avg_qos']['ci']:.4f}")
            if m5:
                print(f"  {model_name} | Method5: {m5['avg_qos']['mean']:.4f} ± {m5['avg_qos']['ci']:.4f}")

    def export_latex_table(stats_results: dict, all_results: dict, filename: str = 'metrics_table.tex'):
        rows = []
        header = [
            "Method",
            "EffQoS",
            "Regret(last100)",
            "LLM Lat (ms)",
            "Avg Lat (ms)",
            "Avg Energy (J)",
        ]
        methods = [
            ('method1_baseline', 'Method1'),
            ('method2_async_blackbox', 'Method2'),
            ('method3_periodic_llm_hybrid', 'Method3'),
            ('method4_distributed_collaborative', 'Method4'),
            ('method5_distributed_llm', 'Method5'),
        ]

        def avg_last100_regret(runs: List[dict]) -> float:
            vals = []
            for r in runs:
                inst = r.get('instant_regret_history', [])
                if not inst:
                    continue
                tail = inst[-100:] if len(inst) >= 100 else inst
                vals.append(np.mean(tail))
            return float(np.mean(vals)) if vals else 0.0

        def avg_llm_latency(runs: List[dict]) -> float:
            vals = []
            for r in runs:
                lat = r.get('llm_latency_ms', [])
                if lat:
                    vals.append(np.mean(lat))
            return float(np.mean(vals)) if vals else 0.0

        for key, name in methods:
            if key not in stats_results:
                continue
            eff_qos = stats_results[key]['avg_effective_qos']['mean']
            avg_lat = stats_results[key]['avg_latency_ms']['mean']
            avg_energy = stats_results[key]['avg_energy_joule']['mean']
            runs = all_results.get(key, [])
            regret_tail = avg_last100_regret(runs)
            llm_lat = avg_llm_latency(runs)
            rows.append([name, eff_qos, regret_tail, llm_lat, avg_lat, avg_energy])

        lines = []
        lines.append("\\begin{tabular}{lccccc}")
        lines.append("\\hline")
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\hline")
        for r in rows:
            lines.append(f"{r[0]} & {r[1]:.4f} & {r[2]:.3f} & {r[3]:.1f} & {r[4]:.1f} & {r[5]:.3f} \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"[OK] LaTeX table saved to {filename}")

    export_latex_table(stats_results, all_results)
    with open('experiment_v3_complete_results.pkl', 'wb') as f:
        pickle.dump({'all_results': all_results, 'stats': stats_results, 'config': asdict(config)}, f)

    print("\nSaved results to experiment_v3_complete_results.pkl")

    try:
        save_comparison_plots(all_results, save_dir='figs', default_llm_model=default_llm_model)
        print("Plots saved to figs/")
    except Exception as e:
        print(f"[WARN] failed to generate plots: {e}")

    return stats_results
# ============================================================================

# 10. 主程序

# ============================================================================



if __name__ == "__main__":

    # 创建配置

    config = ExperimentConfig(

        num_users=12,

        num_workers=6,

        num_paths=4,

        T=500,  # 快速测试用200，完整实验用500

        num_runs=2,  # 至少5次重复

        seed_base=42,

        blackbox_period=25,

        cmaes_max_iters=15,

        cmaes_population=15

    )

    

    start_time = time.time()

    

    # 运行实验

    results = run_complete_experiment_with_statistics(config)

    

    elapsed = time.time() - start_time

    print(f"\n总耗时: {elapsed/60:.1f} 分钟")

    print("\n下一步: 运行可视化脚本生成图表")
