"""
低维 Latent Space 参数映射模块
基于 FedBPT 思想：将 LLM 作为初始化/方向引导器，通过低维 z 映射到 6 个 MTUCB 参数

核心思想：
- 使用低维 latent z (d=5) 代替直接操作 6 个高维参数
- z 空间更容易被黑盒优化器（如 CMA-ES）搜索
- LLM 只在初始化或关键决策点提供方向性引导
- 局部搜索在 z 空间进行，降低搜索复杂度

参数映射关系：
z = [z1, z2, z3, z4, z5]  (每个维度归一化到 [-1, 1])
->
params = [alpha, zeta, omega, compression_ratio, power_ratio, min_phi]
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

try:
    from sc_qos_optimizer import SCQoSConfig
except ImportError:
    @dataclass
    class SCQoSConfig:  # type: ignore
        alpha: float = 0.6
        zeta: float = 0.25
        omega: float = 0.15
        compression_ratio: float = 0.7
        power_ratio: float = 0.5
        min_phi: float = 0.6

        alpha_bounds: Tuple[float, float] = (0.3, 0.9)
        zeta_bounds: Tuple[float, float] = (0.1, 0.5)
        omega_bounds: Tuple[float, float] = (0.05, 0.3)
        compression_bounds: Tuple[float, float] = (0.5, 0.95)
        power_bounds: Tuple[float, float] = (0.3, 0.8)
        min_phi_bounds: Tuple[float, float] = (0.4, 0.9)
        z_bounds: Tuple[float, float] = (-2.0, 2.0)
        latent_dim: int = 5

        @classmethod
        def default(cls):
            return cls()


@dataclass
class LatentConfig:
    """Latent 空间配置"""
    latent_dim: int = 5  # latent 维度
    param_dim: int = 6   # 参数维度
    
    # 参数边界约束
    alpha_bounds: Tuple[float, float] = (0.3, 0.9)
    zeta_bounds: Tuple[float, float] = (0.1, 0.5)
    omega_bounds: Tuple[float, float] = (0.05, 0.3)
    compression_bounds: Tuple[float, float] = (0.5, 0.95)
    power_bounds: Tuple[float, float] = (0.3, 0.8)
    min_phi_bounds: Tuple[float, float] = (0.4, 0.9)
    
    # z 空间边界
    z_bounds: Tuple[float, float] = (-2.0, 2.0)


class LatentParamMapper:
    """
    低维 Latent 到高维参数的映射器
    
    映射策略（简单仿射 + 非线性变换）：
    - z1 -> alpha: 控制路径质量权重（QoS 核心参数）
    - z2 -> zeta: 控制探索强度（exploration-exploitation tradeoff）
    - z3 -> omega: 控制切换成本（稳定性）
    - z4 -> compression_ratio, power_ratio: 通过一个 latent 同时影响压缩和功率（资源效率）
    - z5 -> min_phi: 控制语义速率阈值（服务质量门槛）
    """
    
    def __init__(self, config: Optional[LatentConfig] = None):
        self.config = config or LatentConfig()
        self.mapping_history: List[Tuple[np.ndarray, SCQoSConfig]] = []
        
    def latent_to_params(self, z: np.ndarray) -> SCQoSConfig:
        """将 latent vector z 映射到 MTUCB 参数"""
        assert len(z) == self.config.latent_dim
        
        z = np.clip(z, self.config.z_bounds[0], self.config.z_bounds[1])
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def tanh(x):
            return np.tanh(x)
        
        # z1 -> alpha
        alpha_normalized = sigmoid(z[0])
        alpha = self.config.alpha_bounds[0] + alpha_normalized * (
            self.config.alpha_bounds[1] - self.config.alpha_bounds[0]
        )
        
        # z2 -> zeta
        zeta_normalized = sigmoid(z[1])
        zeta = self.config.zeta_bounds[0] + zeta_normalized * (
            self.config.zeta_bounds[1] - self.config.zeta_bounds[0]
        )
        
        # z3 -> omega
        omega_normalized = sigmoid(z[2])
        omega = self.config.omega_bounds[0] + omega_normalized * (
            self.config.omega_bounds[1] - self.config.omega_bounds[0]
        )
        
        # z4 -> compression_ratio 和 power_ratio
        resource_mode = tanh(z[3])
        
        if resource_mode > 0:
            compression_normalized = 0.5 + 0.5 * resource_mode
            power_normalized = 0.5 - 0.3 * resource_mode
        else:
            compression_normalized = 0.5 + 0.3 * resource_mode
            power_normalized = 0.5 - 0.5 * resource_mode
        
        compression_ratio = self.config.compression_bounds[0] + compression_normalized * (
            self.config.compression_bounds[1] - self.config.compression_bounds[0]
        )
        power_ratio = self.config.power_bounds[0] + power_normalized * (
            self.config.power_bounds[1] - self.config.power_bounds[0]
        )
        
        # z5 -> min_phi
        min_phi_normalized = sigmoid(z[4])
        min_phi = self.config.min_phi_bounds[0] + min_phi_normalized * (
            self.config.min_phi_bounds[1] - self.config.min_phi_bounds[0]
        )
        
        params = SCQoSConfig(
            alpha=float(alpha),
            zeta=float(zeta),
            omega=float(omega),
            compression_ratio=float(compression_ratio),
            power_ratio=float(power_ratio),
            min_phi=float(min_phi)
        )
        
        self.mapping_history.append((z.copy(), params))
        return params
    
    def params_to_latent(self, params) -> np.ndarray:
        """
        将 MTUCB 参数反向映射到 latent vector

        Args:
            params: SCQoSConfig对象或包含参数的dict/动态对象
        """
        def get_param(name: str, default: float) -> float:
            if isinstance(params, dict):
                return float(params.get(name, default))
            return float(getattr(params, name, default))

        def inverse_sigmoid(y):
            y = np.clip(y, 1e-7, 1 - 1e-7)
            return np.log(y / (1 - y))
        
        def inverse_tanh(y):
            y = np.clip(y, -1 + 1e-7, 1 - 1e-7)
            return 0.5 * np.log((1 + y) / (1 - y))

        alpha = get_param('alpha', 0.6)
        zeta = get_param('zeta', 0.25)
        omega = get_param('omega', 0.15)
        compression_ratio = get_param('compression_ratio', 0.7)
        power_ratio = get_param('power_ratio', 0.5)
        min_phi = get_param('min_phi', 0.6)
        
        alpha_normalized = (alpha - self.config.alpha_bounds[0]) / (
            self.config.alpha_bounds[1] - self.config.alpha_bounds[0]
        )
        z1 = inverse_sigmoid(alpha_normalized)
        
        zeta_normalized = (zeta - self.config.zeta_bounds[0]) / (
            self.config.zeta_bounds[1] - self.config.zeta_bounds[0]
        )
        z2 = inverse_sigmoid(zeta_normalized)
        
        omega_normalized = (omega - self.config.omega_bounds[0]) / (
            self.config.omega_bounds[1] - self.config.omega_bounds[0]
        )
        z3 = inverse_sigmoid(omega_normalized)
        
        compression_normalized = (compression_ratio - self.config.compression_bounds[0]) / (
            self.config.compression_bounds[1] - self.config.compression_bounds[0]
        )
        power_normalized = (power_ratio - self.config.power_bounds[0]) / (
            self.config.power_bounds[1] - self.config.power_bounds[0]
        )
        
        resource_mode = compression_normalized - power_normalized
        resource_mode = np.clip(resource_mode, -1, 1)
        z4 = inverse_tanh(resource_mode)
        
        min_phi_normalized = (min_phi - self.config.min_phi_bounds[0]) / (
            self.config.min_phi_bounds[1] - self.config.min_phi_bounds[0]
        )
        z5 = inverse_sigmoid(min_phi_normalized)
        
        z = np.array([z1, z2, z3, z4, z5])
        z = np.clip(z, self.config.z_bounds[0], self.config.z_bounds[1])
        
        return z
    
    def get_default_z(self) -> np.ndarray:
        """获取默认的 latent vector"""
        default_params = SCQoSConfig.default()
        return self.params_to_latent(default_params)
    
    def sample_random_z(self, n: int = 1, std: float = 1.0) -> np.ndarray:
        """从 latent 空间随机采样"""
        z_samples = np.random.randn(n, self.config.latent_dim) * std
        z_samples = np.clip(z_samples, self.config.z_bounds[0], self.config.z_bounds[1])
        return z_samples
    
    def get_mapping_statistics(self) -> Dict[str, any]:
        """获取映射统计信息"""
        if not self.mapping_history:
            return {}
        
        z_history = np.array([z for z, _ in self.mapping_history])
        
        return {
            'num_mappings': len(self.mapping_history),
            'z_mean': np.mean(z_history, axis=0),
            'z_std': np.std(z_history, axis=0),
            'z_min': np.min(z_history, axis=0),
            'z_max': np.max(z_history, axis=0),
        }


if __name__ == "__main__":
    print("🧪 测试 Latent Parameter Mapper")
    print("="*60)
    
    mapper = LatentParamMapper()
    
    print("\n📋 测试1: 默认 latent vector")
    default_z = mapper.get_default_z()
    print(f"默认 z: {default_z}")
    default_params = mapper.latent_to_params(default_z)
    print(f"映射参数: alpha={default_params.alpha:.3f}, zeta={default_params.zeta:.3f}")
    
    print("\n✅ Latent Mapper 测试完成！")


