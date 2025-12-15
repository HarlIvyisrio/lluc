"""
LLM模型质量离线评估脚本 (改进版)

改进内容：
1. 增加多种复杂场景（正常/高负载/低负载/突发/故障）
2. 丰富prompt上下文（与主实验一致）
3. 根据场景难度给予不同权重
4. 使用更小的Beta参数以增加采样随机性
"""

import json
import numpy as np
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

from tqdm import tqdm

from Enhanced_MTUCB_with_Ollama import SCQoSConfig
from run_fixed_experiment_v3_complete import SeededEnvironmentFactory, LatencyModel, run_mtucb_step

MODELS = ['qwen3:8b', 'phi3:mini', 'deepseek-r1:8b']
SAMPLES_PER_SCENARIO = 30  # 每种场景30个样本
WINDOW = 30
OUTPUT_PATH = 'llm_quality_params.json'


@dataclass
class EvaluationScenario:
    """评估场景定义"""
    name: str
    description: str
    num_users: int
    num_workers: int
    num_paths: int
    weight: float  # 场景权重
    difficulty: str  # 简单/中等/困难
    context_hint: str  # 给LLM的上下文提示


# 定义多种评估场景
SCENARIOS = [
    # 简单场景 - 所有模型都能做好
    EvaluationScenario(
        name="normal_balanced",
        description="正常平衡负载",
        num_users=12, num_workers=6, num_paths=4,
        weight=0.15, difficulty="简单",
        context_hint="系统运行正常，用户与工人数量平衡，无异常"
    ),
    EvaluationScenario(
        name="low_load",
        description="低负载场景",
        num_users=6, num_workers=8, num_paths=3,
        weight=0.10, difficulty="简单",
        context_hint="系统负载较低，工人资源充足，可适当探索"
    ),
    
    # 中等场景 - 需要一定推理能力
    EvaluationScenario(
        name="high_load",
        description="高负载场景",
        num_users=20, num_workers=6, num_paths=4,
        weight=0.20, difficulty="中等",
        context_hint="系统负载较高，用户显著多于工人，需要优化资源分配"
    ),
    EvaluationScenario(
        name="complex_paths",
        description="复杂路径场景",
        num_users=15, num_workers=8, num_paths=8,
        weight=0.15, difficulty="中等",
        context_hint="路径选择复杂，有8条可选路径，需要平衡探索与利用"
    ),
    
    # 困难场景 - 需要深度理解
    EvaluationScenario(
        name="overloaded",
        description="过载场景",
        num_users=30, num_workers=5, num_paths=3,
        weight=0.20, difficulty="困难",
        context_hint="系统严重过载（用户/工人比=6），必须优先保证QoS稳定性，减少探索"
    ),
    EvaluationScenario(
        name="sparse_workers",
        description="工人稀缺场景",
        num_users=18, num_workers=3, num_paths=5,
        weight=0.20, difficulty="困难",
        context_hint="工人数量严重不足，资源极度紧张，需要精确优化参数"
    ),
]


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def fit_beta_from_stats(mean: float, std: float, max_sum: float = 15.0) -> Tuple[float, float]:
    """
    根据均值和标准差拟合Beta分布参数
    限制alpha+beta的最大值，确保采样有足够随机性
    """
    mean = np.clip(mean, 0.05, 0.95)
    var = max(std ** 2, 0.001)
    
    # 计算理论alpha/beta
    alpha = ((1 - mean) / var - 1 / mean) * mean ** 2
    beta = alpha * (1 / mean - 1)
    
    # 检查有效性
    if alpha <= 0 or beta <= 0 or not np.isfinite(alpha) or not np.isfinite(beta):
        # 使用默认值
        alpha = mean * 10
        beta = (1 - mean) * 10
    
    # 限制alpha+beta的最大值，确保有足够随机性
    total = alpha + beta
    if total > max_sum:
        scale = max_sum / total
        alpha *= scale
        beta *= scale
    
    return float(max(0.5, alpha)), float(max(0.5, beta))


def call_llm_params(model_name: str, scenario: EvaluationScenario, env) -> Dict[str, float]:
    """调用LLM获取参数建议（使用丰富的语义上下文）"""
    import requests
    import re

    # 计算派生特征
    user_worker_ratio = scenario.num_users / max(1, scenario.num_workers)
    if user_worker_ratio > 2.5:
        load_level = "高负载（用户显著多于工人）"
    elif user_worker_ratio < 1.5:
        load_level = "低负载（工人充足）"
    else:
        load_level = "中等负载"

    # 构建丰富的prompt
    prompt = (
        f"你是网络资源调度优化专家。请根据以下场景给出MTUCB参数建议。\n\n"
        f"【场景描述】\n"
        f"- 场景类型: {scenario.description}\n"
        f"- 难度级别: {scenario.difficulty}\n"
        f"- 场景背景: {scenario.context_hint}\n\n"
        f"【网络配置】\n"
        f"- 用户数: {scenario.num_users}\n"
        f"- 工人数: {scenario.num_workers}\n"
        f"- 路径数: {scenario.num_paths}\n"
        f"- 用户/工人比: {user_worker_ratio:.2f}\n"
        f"- 负载水平: {load_level}\n\n"
        f"【参数说明】\n"
        f"- alpha (0.3-0.9): 路径质量权重，越大越偏向历史表现好的路径\n"
        f"- zeta (0.1-0.5): UCB探索强度，越大越倾向尝试未知路径\n"
        f"- omega (0.05-0.3): 切换成本权重，越大越倾向保持当前路径\n\n"
        f"请仅输出JSON: {{\"alpha\":值, \"zeta\":值, \"omega\":值}}"
    )

    ports = [11434, 11435]
    for port in ports:
        try:
            resp = requests.post(
                f"http://localhost:{port}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.3, "num_predict": 200}
                },
                timeout=30,
            )
            if resp.status_code != 200:
                continue
            raw = resp.json().get('response', '').strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and all(k in parsed for k in ['alpha', 'zeta', 'omega']):
                    return {
                        'alpha': float(np.clip(parsed['alpha'], 0.3, 0.9)),
                        'zeta': float(np.clip(parsed['zeta'], 0.1, 0.5)),
                        'omega': float(np.clip(parsed['omega'], 0.05, 0.3)),
                    }
            except Exception:
                pass
            # fallback: 正则提取数字
            nums = [float(x) for x in re.findall(r"([0-1](?:\.[0-9]+)?)", raw)]
            if len(nums) >= 3:
                return {
                    'alpha': float(np.clip(nums[0], 0.3, 0.9)),
                    'zeta': float(np.clip(nums[1], 0.1, 0.5)),
                    'omega': float(np.clip(nums[2], 0.05, 0.3)),
                }
        except requests.exceptions.ConnectionError:
            continue
        except Exception:
            break

    # 完全失败时使用默认参数
    cfg = SCQoSConfig.default()
    return {'alpha': cfg.alpha, 'zeta': cfg.zeta, 'omega': cfg.omega}


def simulate_short(env, T: int, lambda_delay: float, lambda_energy: float, d_max: float, e_max: float) -> float:
    """短窗口仿真，返回平均QoS"""
    qos_list = []
    for t in range(T):
        metrics = run_mtucb_step(env, t, lambda_delay, lambda_energy, d_max, e_max)
        qos_list.append(metrics.avg_qos)
    return float(np.mean(qos_list)) if qos_list else 0.0


def main():
    set_seed(42)
    latency_model = LatencyModel()
    cfg = SCQoSConfig.default()

    lambda_delay = 0.5
    lambda_energy = 0.5
    d_max = 500.0
    e_max = 1.0

    # 按场景和模型收集分数
    scores_by_scenario = {s.name: {m: [] for m in MODELS} for s in SCENARIOS}
    
    print(f"开始评估 {len(MODELS)} 个LLM模型，共 {len(SCENARIOS)} 种场景")
    print("=" * 60)

    for scenario in SCENARIOS:
        print(f"\n场景: {scenario.name} ({scenario.difficulty})")
        print(f"  配置: {scenario.num_users}用户/{scenario.num_workers}工人/{scenario.num_paths}路径")
        
        factory = SeededEnvironmentFactory(
            num_users=scenario.num_users,
            num_workers=scenario.num_workers,
            num_paths=scenario.num_paths
        )
        
        for i in tqdm(range(SAMPLES_PER_SCENARIO), desc=f"  评估 {scenario.name}"):
            seed = 10000 + hash(scenario.name) % 10000 + i
            
            for model in MODELS:
                # 创建环境并调用LLM
                env_state = factory.create_env(seed)
                env_state.config = cfg
                params = call_llm_params(model, scenario, env_state)
                
                # 使用LLM建议的参数进行仿真
                env = factory.create_env(seed + 999)
                env.config.alpha = params.get('alpha', cfg.alpha)
                env.config.zeta = params.get('zeta', cfg.zeta)
                env.config.omega = params.get('omega', cfg.omega)
                
                avg_qos = simulate_short(env, WINDOW, lambda_delay, lambda_energy, d_max, e_max)
                scores_by_scenario[scenario.name][model].append(avg_qos)

    # 计算加权分数
    print("\n" + "=" * 60)
    print("计算加权质量分数...")
    
    weighted_scores = {m: [] for m in MODELS}
    
    for scenario in SCENARIOS:
        print(f"\n场景 {scenario.name} (权重={scenario.weight:.2f}):")
        for model in MODELS:
            arr = np.array(scores_by_scenario[scenario.name][model])
            mean_qos = float(np.mean(arr)) if arr.size else 0.0
            print(f"  {model}: {mean_qos:.4f}")
            # 加权添加到总分
            weighted_scores[model].extend([mean_qos * scenario.weight] * len(arr))

    # 生成最终质量参数
    print("\n" + "=" * 60)
    print("生成Beta分布参数...")
    
    quality_summary = {}
    for model in MODELS:
        arr = np.array(weighted_scores[model], dtype=float)
        mean = float(np.mean(arr)) if arr.size else 0.5
        std = float(np.std(arr)) if arr.size else 0.1
        
        # 使用限制的Beta参数（max_sum=12，确保有随机性）
        alpha, beta = fit_beta_from_stats(mean, std, max_sum=12.0)
        
        quality_summary[model] = {
            'mean': mean,
            'std': std,
            'beta_alpha': alpha,
            'beta_beta': beta,
        }
        print(f"  {model}: mean={mean:.4f}, std={std:.4f}, alpha={alpha:.2f}, beta={beta:.2f}")

    # 保存结果
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(quality_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 保存质量参数到 {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
