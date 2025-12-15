"""
增量式CMA-ES优化器 - 支持ask/tell接口
允许在主仿真循环中逐步执行优化，不阻塞主流程
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class CMAESConfig:
    """CMA-ES配置"""
    dim: int  # 搜索空间维度
    population_size: Optional[int] = None  # 种群大小（默认4+3*ln(dim)）
    sigma0: float = 0.3  # 初始步长
    max_iterations: int = 50  # 最大迭代数
    tol: float = 1e-8  # 收敛容差
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # 搜索边界


class IncrementalCMAES:
    """
    增量式CMA-ES优化器
    
    使用ask/tell接口，允许在主循环的每个时隙执行部分优化迭代
    避免阻塞整个仿真流程
    """
    
    def __init__(self, config: CMAESConfig, seed: Optional[int] = None):
        """
        初始化CMA-ES优化器
        
        Args:
            config: CMA-ES配置
            seed: 随机种子（确保可复现性）
        """
        self.config = config
        self.dim = config.dim
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        
        # 种群大小
        if config.population_size is None:
            self.lambda_ = int(4 + 3 * np.log(self.dim))
        else:
            self.lambda_ = config.population_size
        
        self.mu = self.lambda_ // 2  # 选择父代数量
        
        # 权重向量（用于重组）
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # 学习率
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # 期望范数
        self.chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        # 初始化状态变量
        self.reset()
    
    def reset(self, x0: Optional[np.ndarray] = None):
        """
        重置优化器状态
        
        Args:
            x0: 初始搜索点（如果为None，则使用随机初始化）
        """
        # 均值向量
        if x0 is not None:
            self.xmean = x0.copy()
        else:
            self.xmean = self.rng.randn(self.dim)
        
        # 步长
        self.sigma = self.config.sigma0
        
        # 协方差矩阵
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)  # 演化路径（协方差）
        self.ps = np.zeros(self.dim)  # 演化路径（步长）
        
        # 特征分解缓存
        self.B = np.eye(self.dim)  # 特征向量
        self.D = np.ones(self.dim)  # 特征值的平方根
        self.invsqrtC = np.eye(self.dim)
        
        # 迭代计数
        self.iteration = 0
        self.eval_count = 0
        
        # 最佳解
        self.best_x = self.xmean.copy()
        self.best_fitness = np.inf
        
        # 当前代候选解（ask阶段生成）
        self.current_population = None
        self.current_fitnesses = None
        
        # 收敛标志
        self.converged = False
    
    def ask(self) -> List[np.ndarray]:
        """
        生成一批候选解（不阻塞）
        
        Returns:
            候选解列表
        """
        if self.converged:
            return []
        
        # 生成候选解
        self.current_population = []
        for _ in range(self.lambda_):
            # 采样标准正态分布
            z = self.rng.randn(self.dim)
            # 转换到搜索空间
            y = self.B @ (self.D * z)
            x = self.xmean + self.sigma * y
            
            # 边界约束
            if self.config.bounds is not None:
                lower, upper = self.config.bounds
                x = np.clip(x, lower, upper)
            
            self.current_population.append(x)
        
        return self.current_population
    
    def tell(self, population: List[np.ndarray], fitnesses: List[float]):
        """
        根据评估结果更新CMA-ES状态（不阻塞）
        
        Args:
            population: 候选解列表
            fitnesses: 对应的适应度值（越小越好）
        """
        if self.converged:
            return
        
        assert len(population) == len(fitnesses) == self.lambda_
        
        self.current_fitnesses = np.array(fitnesses)
        self.eval_count += self.lambda_
        
        # 排序（升序：小适应度更好）
        idx_sorted = np.argsort(self.current_fitnesses)
        
        # 更新最佳解
        if self.current_fitnesses[idx_sorted[0]] < self.best_fitness:
            self.best_fitness = self.current_fitnesses[idx_sorted[0]]
            self.best_x = population[idx_sorted[0]].copy()
        
        # 选择精英
        selected_x = [population[i] for i in idx_sorted[:self.mu]]
        
        # 计算加权平均（新的均值）
        xold = self.xmean.copy()
        self.xmean = np.sum([self.weights[i] * selected_x[i] for i in range(self.mu)], axis=0)
        
        # 更新演化路径（步长控制）
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  self.invsqrtC @ (self.xmean - xold) / self.sigma
        
        # 步长自适应
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        
        # 更新演化路径（协方差矩阵）
        hsig = (np.linalg.norm(self.ps) / 
                np.sqrt(1 - (1 - self.cs) ** (2 * self.iteration + 2)) / self.chiN) < \
               (1.4 + 2 / (self.dim + 1))
        
        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  (self.xmean - xold) / self.sigma
        
        # 更新协方差矩阵
        artmp = [(selected_x[i] - xold) / self.sigma for i in range(self.mu)]
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) + 
                           (1 - hsig) * self.cc * (2 - self.cc) * self.C)
        
        for i in range(self.mu):
            self.C += self.cmu * self.weights[i] * np.outer(artmp[i], artmp[i])
        
        # 对称化
        self.C = (self.C + self.C.T) / 2
        
        # 特征分解（每次迭代都更新）
        self._update_eigensystem()
        
        self.iteration += 1
        
        # 检查收敛
        self._check_convergence()
    
    def _update_eigensystem(self):
        """更新协方差矩阵的特征分解"""
        # 特征分解
        eigvals, eigvecs = np.linalg.eigh(self.C)
        
        # 确保特征值为正
        eigvals = np.maximum(eigvals, 1e-12)
        
        self.D = np.sqrt(eigvals)
        self.B = eigvecs
        
        # 计算C^{-1/2}
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
    
    def _check_convergence(self):
        """检查是否收敛"""
        # 迭代数达到上限
        if self.iteration >= self.config.max_iterations:
            self.converged = True
            return
        
        # 步长过小
        if self.sigma * np.max(self.D) < self.config.tol:
            self.converged = True
            return
        
        # 适应度变化过小
        if self.current_fitnesses is not None:
            fitness_range = np.max(self.current_fitnesses) - np.min(self.current_fitnesses)
            if fitness_range < self.config.tol:
                self.converged = True
                return
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """
        获取当前最佳解
        
        Returns:
            (最佳解, 最佳适应度)
        """
        return self.best_x.copy(), self.best_fitness
    
    def is_converged(self) -> bool:
        """检查是否已收敛"""
        return self.converged
    
    def get_progress(self) -> dict:
        """
        获取优化进度信息
        
        Returns:
            包含迭代数、评估次数、最佳适应度等信息的字典
        """
        return {
            'iteration': self.iteration,
            'eval_count': self.eval_count,
            'best_fitness': self.best_fitness,
            'sigma': self.sigma,
            'converged': self.converged
        }


class AsyncBlackboxOptimizer:
    """
    真异步黑盒优化器
    
    在主仿真循环中逐步执行优化，不阻塞时序推进
    """
    
    def __init__(
        self,
        evaluator: Callable,  # 评估函数
        latent_dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        max_iterations: int = 50,
        population_size: int = 20,
        seed: Optional[int] = None
    ):
        """
        初始化异步优化器
        
        Args:
            evaluator: 评估函数 (latent_vector, current_t) -> fitness
            latent_dim: 潜在空间维度
            bounds: 搜索边界
            max_iterations: 最大迭代数
            population_size: 种群大小
            seed: 随机种子
        """
        self.evaluator = evaluator
        self.latent_dim = latent_dim
        self.bounds = bounds
        self.seed = seed
        
        # CMA-ES配置
        config = CMAESConfig(
            dim=latent_dim,
            population_size=population_size,
            sigma0=0.3,
            max_iterations=max_iterations,
            bounds=bounds
        )
        
        # 创建增量式CMA-ES
        self.cmaes = IncrementalCMAES(config, seed=seed)
        
        # 状态
        self.is_running = False
        self.current_population = None
        self.current_t = None
        self.start_time = None
        
        # 统计
        self.total_evals = 0
    
    def start(self, z0: np.ndarray, current_t: int):
        """
        开始一次优化（初始化）
        
        Args:
            z0: 初始搜索点
            current_t: 当前全局时间
        """
        self.cmaes.reset(z0)
        self.is_running = True
        # 使用启动时刻作为评估基准，避免优化期间目标函数随时间漂移
        self.current_t = current_t
        self.start_time = time.time()
        
        # 立即生成第一批候选解
        self.current_population = self.cmaes.ask()
    
    def step(self, current_t: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        执行一步优化（在主循环的一个时隙中调用）
        
        Args:
            current_t: 当前全局时间
            
        Returns:
            如果优化完成，返回(最佳解, 最佳适应度)；否则返回None
        """
        if not self.is_running:
            return None
        
        # 如果已经生成了候选解，进行评估
        if self.current_population is not None and len(self.current_population) > 0:
            # 评估当前批次的候选解
            fitnesses = []
            for x in self.current_population:
                # 始终使用启动时刻的环境状态进行评估，避免目标函数随时间漂移
                eval_t = self.current_t if self.current_t is not None else current_t
                fitness = self.evaluator(x, eval_t)
                fitnesses.append(fitness)
            
            self.total_evals += len(self.current_population)
            
            # 告诉CMA-ES评估结果
            self.cmaes.tell(self.current_population, fitnesses)
            
            # 检查是否收敛
            if self.cmaes.is_converged():
                best_x, best_f = self.cmaes.get_best()
                self.is_running = False
                return best_x, best_f
            
            # 生成下一批候选解
            self.current_population = self.cmaes.ask()
        
        return None
    
    def get_progress(self) -> dict:
        """获取优化进度"""
        progress = self.cmaes.get_progress()
        progress['total_evals'] = self.total_evals
        progress['is_running'] = self.is_running
        return progress
    
    def shutdown(self):
        """关闭优化器，释放资源"""
        self.is_running = False
        self.current_population = None
        self.current_t = None
        print(f"   🛑 AsyncBlackboxOptimizer已关闭 (总评估数: {self.total_evals})")


# ============ 简单测试 ============
if __name__ == "__main__":
    print("测试增量式CMA-ES...")
    
    # 测试函数：Sphere function
    def sphere(x):
        return np.sum(x ** 2)
    
    # 配置
    dim = 5
    config = CMAESConfig(
        dim=dim,
        population_size=10,
        max_iterations=20,
        bounds=(np.full(dim, -5.0), np.full(dim, 5.0))
    )
    
    # 创建优化器
    cmaes = IncrementalCMAES(config, seed=42)
    cmaes.reset(np.ones(dim) * 2.0)
    
    # 模拟主循环
    print(f"\n开始优化 (目标: 最小化 sum(x^2))...")
    t = 0
    while not cmaes.is_converged():
        # Ask: 生成候选解
        population = cmaes.ask()
        
        # 评估（模拟在主循环中评估）
        fitnesses = [sphere(x) for x in population]
        
        # Tell: 更新状态
        cmaes.tell(population, fitnesses)
        
        # 打印进度
        progress = cmaes.get_progress()
        print(f"时隙 {t}: 迭代={progress['iteration']}, "
              f"最佳适应度={progress['best_fitness']:.6f}, "
              f"sigma={progress['sigma']:.6f}")
        
        t += 1
    
    best_x, best_f = cmaes.get_best()
    print(f"\n优化完成！")
    print(f"最佳解: {best_x}")
    print(f"最佳适应度: {best_f:.8f}")
    print(f"总迭代数: {cmaes.iteration}")


