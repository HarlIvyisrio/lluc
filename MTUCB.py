import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import seaborn as sns
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import ListedColormap

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Environment:
    def __init__(self, num_users: int, num_workers: int, num_paths: int, T: int, alpha: float = 0.3, zeta: float = 0.2,
                 omega: float = 0.1):
        self.num_users = num_users
        self.num_workers = num_workers
        self.num_paths = num_paths
        self.T = T

        # 初始化知识库适配度矩阵 A(u_i, w_j)
        self.compatibility_matrix = np.random.uniform(0.5, 1.0, size=(num_users, num_workers))

        # 初始化用户支付意愿 W(u_i)
        self.willingness_to_pay = np.random.uniform(0.3, 1.0, size=num_users)

        # 初始化路径质量矩阵,随时间变化
        self.path_qualities = np.random.uniform(0.4, 0.9, size=(T, num_workers, num_paths))

        # 设置算法参数
        self.alpha = alpha  # QoS计算中路径质量的权重
        self.beta = 0.8  # 权重平衡参数
        self.zeta = zeta  # UCB探索参数
        self.omega = omega  # 切换成本权重
        self.worker_capacity = 2  # 每个工人的最大服务容量

        # 记录最优QoS和最优匹配用于计算遗憾和最优选择率，每个时间槽一个值
        self.optimal_qos, self.optimal_assignments = self._calculate_optimal_qos()

    def _calculate_optimal_qos(self) -> Tuple[List[float], List[List[Tuple[int, int, int]]]]:
        """计算每个时间槽的理论最优QoS总和和对应的最优匹配组合，使用改进的匈牙利算法思想"""
        optimal_qos_per_t = []
        optimal_assignments = []
        
        for t in range(self.T):
            # 构建用户-工人-路径的完整收益矩阵
            benefit_matrix = np.zeros((self.num_users, self.num_workers, self.num_paths))
            for u in range(self.num_users):
                for w in range(self.num_workers):
                    for p in range(self.num_paths):
                        # 初始时假设每个工人仅服务一个用户
                        benefit_matrix[u, w, p] = self.calculate_qos(t, u, w, p, 0)
            
            # 多次迭代直到收敛，考虑负载影响
            final_assignments = []
            final_total_qos = 0
            
            # 用迭代方法找到更好的匹配
            for attempt in range(3):  # 尝试几次不同的初始分配
                # 当前迭代的变量
                current_assignments = []
                worker_loads = {w: 0 for w in range(self.num_workers)}
                used_users = set()
                total_qos = 0
                
                # 创建初始分配，基于当前benefit_matrix
                # 将benefit_matrix展平为用户 x (工人-路径)
                user_count = self.num_users
                option_count = self.num_workers * self.num_paths
                flattened_benefits = benefit_matrix.reshape(user_count, option_count)
                
                # 按照benefit降序为每个用户分配
                for u in range(self.num_users):
                    if u in used_users:
                        continue
                    
                    # 按收益降序获取所有(工人,路径)组合
                    options = [(flattened_benefits[u, i], i // self.num_paths, i % self.num_paths) 
                               for i in range(option_count)]
                    options.sort(reverse=True)
                    
                    for _, w, p in options:
                        if worker_loads[w] < self.worker_capacity:
                            # 计算实际QoS，考虑当前负载
                            actual_qos = self.calculate_qos(t, u, w, p, worker_loads[w])
                            current_assignments.append((u, w, p))
                            total_qos += actual_qos
                            used_users.add(u)
                            worker_loads[w] += 1
                            break
                
                # 优化当前分配：尝试交换或替换以提高总QoS
                for _ in range(5):  # 执行几轮优化
                    improved = False
                    
                    # 对于每个已分配的用户，尝试更改其分配
                    for i, (u, w, p) in enumerate(current_assignments):
                        current_qos = self.calculate_qos(t, u, w, p, worker_loads[w] - 1)
                        
                        # 尝试其他(工人,路径)组合
                        for new_w in range(self.num_workers):
                            if new_w == w or worker_loads[new_w] >= self.worker_capacity:
                                continue
                                
                            for new_p in range(self.num_paths):
                                # 计算新分配的QoS
                                new_qos = self.calculate_qos(t, u, new_w, new_p, worker_loads[new_w])
                                
                                if new_qos > current_qos:
                                    # 更新分配
                                    worker_loads[w] -= 1
                                    worker_loads[new_w] += 1
                                    current_assignments[i] = (u, new_w, new_p)
                                    total_qos = total_qos - current_qos + new_qos
                                    improved = True
                                    break
                            if improved:
                                break
                    
                    if not improved:
                        break
                
                # 更新最终分配结果
                if total_qos > final_total_qos:
                    final_total_qos = total_qos
                    final_assignments = current_assignments.copy()
            
            optimal_qos_per_t.append(final_total_qos)
            optimal_assignments.append(final_assignments)
        
        return optimal_qos_per_t, optimal_assignments

    def get_path_quality(self, t: int, worker: int, path: int) -> float:
        """获取特定时间点的路径质量"""
        return self.path_qualities[t, worker, path]

    def get_compatibility(self, user: int, worker: int) -> float:
        """获取用户和工人的知识库适配度"""
        return self.compatibility_matrix[user, worker]

    def get_willingness_to_pay(self, user: int) -> float:
        """获取用户的支付意愿"""
        return self.willingness_to_pay[user]

    def calculate_qos(self, t: int, user: int, worker: int, path: int, current_load: int) -> float:
        """计算服务质量(QoS)，考虑负载衰减"""
        path_quality = self.get_path_quality(t, worker, path)
        compatibility = self.get_compatibility(user, worker)
        # load_factor = 1 - 0.05 * current_load  # 示例：每增加一个负载，QoS下降5%
        # 修改后负载因子计算
        load_factor = 0.85 ** current_load  # 指数衰减更符合实际网络特性
        base_qos = (self.alpha * path_quality + (1 - self.alpha) * compatibility) * load_factor
        return base_qos

    def get_worker_preference(self, worker: int, user: int) -> float:
        """获取工人对用户的偏好分数，分数越高，偏好越高"""
        preference_score = self.beta * self.get_compatibility(user, worker) + (
                    1 - self.beta) * self.get_willingness_to_pay(user)
        return preference_score


class MTUCB:
    def __init__(self, env: Environment):
        self.env = env
        self.num_users = env.num_users
        self.num_workers = env.num_workers
        self.num_paths = env.num_paths

        # 初始化累积奖励和选择次数矩阵（用户 × 工人 × 路径）
        self.R = np.zeros((self.num_users, self.num_workers, self.num_paths))
        self.S = np.zeros((self.num_users, self.num_workers, self.num_paths))

        # 历史匹配记录
        self.historical_matches = {u: [] for u in range(self.num_users)}

        # 记录性能指标
        self.average_qos_history = []
        self.cumulative_regret = []
        self.optimal_choice_rate = []
        self.system_utility = []
        self.average_reward_history = []  # 新增

        # 记录匹配结果用于绘制热力图
        self.matching_history = []

    def _get_env_volatility(self, worker: int) -> float:
        """获取工人最近10个时间片的路径质量波动率"""
        if len(self.env.path_qualities) < 10:
            return 1.0
        recent_data = self.env.path_qualities[-10:, worker, :]
        return np.std(recent_data) + 1e-6  # 防止除零

    def calculate_preference(self, t: int, user: int, worker: int, gamma: float = 0.5) -> float:
        """计算用户对工人的偏好分数，结合路径选择"""
        # 计算所有路径的预期奖励（加权平均）
        total_expected_reward = 0
        total_selections = 0
        for p in range(self.num_paths):
            total_expected_reward += self.R[user, worker, p]
            total_selections += self.S[user, worker, p]

        if total_selections > 0:
            avg_reward = total_expected_reward / total_selections
        else:
            avg_reward = self.env.get_compatibility(user, worker)

        # 获取上一次匹配的工人
        last_match = self.historical_matches[user][-1][1] if self.historical_matches[user] else None
        switched = (last_match != worker) if last_match is not None else False

        # 计算切换成本
        switching_cost = gamma * self.env.omega if switched else 0

        # 计算UCB项 based on total selections over all paths，动态调整zeta
        dynamic_zeta = self.env.zeta / np.sqrt(t + 1)
        ucb_term = np.sqrt(dynamic_zeta * np.log(t + 1) / (total_selections + 1))

        # 在calculate_preference方法中添加：
        # 修正后的代码
        stability = 0.2 * (1 - 1 / (1 + len([w for _, w in self.historical_matches[user] if w == worker])))
        
        # 增加对环境中最佳路径质量的考虑
        path_quality_bonus = 0.15 * max([self.env.get_path_quality(t, worker, p) for p in range(self.num_paths)])
        
        preference = avg_reward + ucb_term + stability + path_quality_bonus - switching_cost
        return preference

    def capacity_constrained_gale_shapley(self, t: int) -> List[Tuple[int, int]]:
        """执行改进的容量受限Gale-Shapley算法，更好地近似最优匹配策略"""
        # 初始化
        free_users = list(range(self.num_users))
        proposals = {u: set() for u in range(self.num_users)}
        matches = {w: [] for w in range(self.num_workers)}

        # 计算所有用户-工人对的QoS估计值
        estimated_qos = {}
        for u in range(self.num_users):
            for w in range(self.num_workers):
                # 寻找历史上表现最好的路径
                best_path_qos = 0
                for p in range(self.num_paths):
                    if self.S[u, w, p] > 0:
                        path_qos = self.R[u, w, p] / self.S[u, w, p]
                        if path_qos > best_path_qos:
                            best_path_qos = path_qos
                
                if best_path_qos == 0:  # 如果没有历史数据
                    best_path_qos = self.env.get_compatibility(u, w)
                
                # 计算偏好分数    
                pref = self.calculate_preference(t, u, w)
                estimated_qos[(u, w)] = pref
        
        # 改进的匹配过程：先按QoS降序排序尝试贪心匹配，然后再用Gale-Shapley细化
        # 按照估计的QoS降序排序所有用户-工人对
        sorted_pairs = sorted(estimated_qos.items(), key=lambda x: x[1], reverse=True)
        
        # 先尝试贪心匹配
        used_users = set()
        worker_capacity = {w: self.env.worker_capacity for w in range(self.num_workers)}
        
        for (u, w), _ in sorted_pairs:
            if u not in used_users and worker_capacity[w] > 0:
                matches[w].append(u)
                used_users.add(u)
                worker_capacity[w] -= 1
                if u in free_users:
                    free_users.remove(u)
        
        # 对剩余未匹配的用户使用传统Gale-Shapley算法
        while free_users:
            u = free_users.pop(0)
            # 用户根据偏好向工人发起提议，按偏好顺序
            preferences = sorted(range(self.num_workers), key=lambda w: self.calculate_preference(t, u, w), reverse=True)
            for w in preferences:
                if w not in proposals[u]:
                    proposals[u].add(w)
                    if len(matches[w]) < self.env.worker_capacity:
                        matches[w].append(u)
                        break
                    else:
                        # 找到当前匹配中工人偏好最低的用户
                        worst_u = min(matches[w], key=lambda u_prime: self.env.get_worker_preference(w, u_prime))
                        if self.env.get_worker_preference(w, u) > self.env.get_worker_preference(w, worst_u):
                            matches[w].remove(worst_u)
                            matches[w].append(u)
                            free_users.append(worst_u)  # 追加最差用户
                            break
            else:
                # 用户无法提出更多提议，保持自由状态
                continue

        # 构建匹配结果
        matching = []
        for w in range(self.num_workers):
            for u in matches[w]:
                matching.append((u, w))

        return matching

    def select_optimal_path(self, t: int, user: int, worker: int, preferred_path: int = 0) -> int:
        """使用改进的UCB策略选择最优路径，更好地考虑路径相关性和当前路径质量"""
        best_score = float('-inf')
        best_path = 0

        # 获取当前时间槽的路径质量
        current_qualities = [self.env.get_path_quality(t, worker, p) for p in range(self.num_paths)]
        
        # 计算路径质量排序
        quality_ranks = np.argsort(current_qualities)[::-1]  # 从高到低排序

        for p in range(self.num_paths):
            if self.S[user, worker, p] > 0:
                avg_reward = self.R[user, worker, p] / self.S[user, worker, p]
            else:
                avg_reward = self.env.get_compatibility(user, worker) * current_qualities[p]

            # 路径质量排名奖励，使算法更倾向于选择当前质量高的路径
            quality_rank_bonus = 0.1 * (self.num_paths - np.where(quality_ranks == p)[0][0]) / self.num_paths

            # 考虑路径相关性，例如某些路径可能更优
            path_bonus = 0.1 if p == preferred_path else 0.0  # preferred_path可以根据经验或其他策略定义

            # 动态调整zeta
            dynamic_zeta = self.env.zeta / np.sqrt(t + 1)
            ucb_term = np.sqrt(dynamic_zeta * np.log(t + 1) / (self.S[user, worker, p] + 1))
            
            # 预测路径质量
            predicted_quality = 0.7 * self.env.path_qualities[t, worker, p] + 0.3 * np.mean(
                self.env.path_qualities[max(0, t - 3):t, worker, p])
            
            # 综合评分
            score = avg_reward + ucb_term + path_bonus + quality_rank_bonus
            score += 0.15 * predicted_quality  # 考虑预测的路径质量

            if score > best_score:
                best_score = score
                best_path = p

        return best_path

    def calculate_total_reward(self, qos: float, switched: bool) -> float:
        """计算总奖励，考虑切换成本"""
        if switched:
            return max(qos - self.env.omega, 0)  # 确保奖励非负
        return qos

    def run(self) -> None:
        """运行MTUCB算法"""
        for t in tqdm(range(self.env.T)):
            # 执行匹配
            matching = self.capacity_constrained_gale_shapley(t)
            # 扩展为 (u, w, p)
            matching_with_p = []
            total_qos = 0
            worker_load = {w: 0 for w in range(self.env.num_workers)}
            for (u, w) in matching:
                worker_load[w] += 1
                path = self.select_optimal_path(t, u, w)
                qos = self.env.calculate_qos(t, u, w, path, worker_load[w])
                switched = (self.historical_matches[u][-1][1] != w) if self.historical_matches[u] else False
                total_reward = self.calculate_total_reward(qos, switched)
                total_qos += qos
                matching_with_p.append((u, w, p := path))  # 修正变量名

                # 更新累积奖励和选择次数，仅使用qos
                self.R[u, w, p] += qos  # 累加QoS奖励，不包括切换成本
                self.S[u, w, p] += 1

                # 更新历史匹配，增加历史记录长度至10
                self.historical_matches[u].append((t, w))
                max_history = 10  # 增加历史记录长度
                if len(self.historical_matches[u]) > max_history:
                    self.historical_matches[u].pop(0)

            self.matching_history.append(matching_with_p)

            # 记录性能指标
            avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
            self.average_qos_history.append(avg_qos)

            # 记录平均奖励
            if matching_with_p:
                rewards = [self.R[u, w, p] / self.S[u, w, p] for (u, w, p) in matching_with_p if self.S[u, w, p] > 0]
                avg_reward = np.mean(rewards) if rewards else 0
            else:
                avg_reward = 0
            self.average_reward_history.append(avg_reward)

            # 计算遗憾
            regret = self.env.optimal_qos[t] - total_qos
            self.cumulative_regret.append(regret)

            # 计算最优选择率 - MTUCB使用用户-工人对方法
            optimal_pairs = set((u, w) for (u, w, p) in self.env.optimal_assignments[t])
            matched_pairs = set((u, w) for (u, w, p) in matching_with_p)
            optimal_choices = len(optimal_pairs & matched_pairs)
            total_optimal = len(optimal_pairs)
            optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
            self.optimal_choice_rate.append(optimal_choice_rate)

            # 记录系统效用
            self.system_utility.append(total_qos)

            # 调试输出
            if t % 100 == 0:
                print(
                    f"Time {t}: Avg QoS={avg_qos:.4f}, Avg Reward={avg_reward:.4f}, Regret={regret:.4f}, "
                    f"Optimal Choice Rate={optimal_choice_rate:.4f}, Total QoS={total_qos:.4f}"
                )


class PureUCB:
    """纯UCB算法，不考虑匹配稳定性"""

    def __init__(self, env: Environment):
        self.env = env
        self.num_users = env.num_users
        self.num_workers = env.num_workers
        self.num_paths = env.num_paths

        # 初始化累积奖励和选择次数矩阵（用户 × 工人 × 路径）
        self.R = np.zeros((self.num_users, self.num_workers, self.num_paths))
        self.S = np.zeros((self.num_users, self.num_workers, self.num_paths))

        # 记录性能指标
        self.average_qos_history = []
        self.cumulative_regret = []
        self.optimal_choice_rate = []
        self.system_utility = []
        self.average_reward_history = []  # 新增

        # 记录匹配结果用于绘制热力图
        self.matching_history = []

    def select_worker_and_path(self, t: int, user: int) -> Tuple[int, int]:
        """使用UCB策略选择工人和路径"""
        best_score = float('-inf')
        best_worker = None
        best_path = None

        for w in range(self.num_workers):
            for p in range(self.num_paths):
                if self.S[user, w, p] > 0:
                    avg_reward = self.R[user, w, p] / self.S[user, w, p]
                else:
                    avg_reward = self.env.get_compatibility(user, w)

                # 动态调整zeta
                dynamic_zeta = self.env.zeta / np.sqrt(t + 1)
                ucb_term = np.sqrt(dynamic_zeta * np.log(t + 1) / (self.S[user, w, p] + 1))
                score = avg_reward + ucb_term

                if score > best_score:
                    best_score = score
                    best_worker = w
                    best_path = p

        return best_worker, best_path

    def run(self) -> None:
        """运行Pure UCB算法，考虑工人容量限制"""
        for t in tqdm(range(self.env.T)):
            matching = []
            matching_with_p = []
            total_qos = 0
            worker_capacity = {w: self.env.worker_capacity for w in range(self.env.num_workers)}

            for u in range(self.num_users):
                w, p = self.select_worker_and_path(t, u)
                if w is not None and worker_capacity[w] > 0:
                    matching.append((u, w))
                    matching_with_p.append((u, w, p))
                    worker_capacity[w] -= 1
                    qos = self.env.calculate_qos(t, u, w, p, self.env.worker_capacity - worker_capacity[w])
                    total_qos += qos

                    # 更新累积奖励和选择次数
                    self.R[u, w, p] += qos
                    self.S[u, w, p] += 1

            self.matching_history.append(matching_with_p)

            # 记录性能指标
            avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
            self.average_qos_history.append(avg_qos)

            # 记录平均奖励
            if matching_with_p:
                rewards = [self.R[u, w, p] / self.S[u, w, p] for (u, w, p) in matching_with_p if self.S[u, w, p] > 0]
                avg_reward = np.mean(rewards) if rewards else 0
            else:
                avg_reward = 0
            self.average_reward_history.append(avg_reward)

            # 计算遗憾
            regret = self.env.optimal_qos[t] - total_qos
            self.cumulative_regret.append(regret)

            # 计算最优选择率 - 其他算法使用完整三元组方法
            optimal_assignments_t = set(tuple(a) for a in self.env.optimal_assignments[t])
            matched_assignments_t = set(tuple(a) for a in matching_with_p)
            optimal_choices = len(optimal_assignments_t & matched_assignments_t)
            total_optimal = len(optimal_assignments_t)
            optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
            self.optimal_choice_rate.append(optimal_choice_rate)

            # 记录系统效用
            self.system_utility.append(total_qos)

            # 调试输出
            if t % 100 == 0:
                print(
                    f"Time {t}: Avg QoS={avg_qos:.4f}, Avg Reward={avg_reward:.4f}, Regret={regret:.4f}, "
                    f"Optimal Choice Rate={optimal_choice_rate:.4f}, Total QoS={total_qos:.4f}"
                )


class GreedyAlgorithm:
    """纯贪心算法，只关注当前服务质量最高的工人和路径"""

    def __init__(self, env: Environment):
        self.env = env
        self.num_users = env.num_users
        self.num_workers = env.num_workers
        self.num_paths = env.num_paths

        # 初始化累积奖励和选择次数矩阵（用户 × 工人 × 路径）
        self.R = np.zeros((self.num_users, self.num_workers, self.num_paths))
        self.S = np.zeros((self.num_users, self.num_workers, self.num_paths))

        # 历史匹配记录
        self.historical_matches = {u: [] for u in range(self.num_users)}

        # 记录性能指标
        self.average_qos_history = []
        self.cumulative_regret = []
        self.optimal_choice_rate = []
        self.system_utility = []
        self.average_reward_history = []  # 新增

        # 记录匹配结果用于绘制热力图
        self.matching_history = []

    def select_action(self, t: int, user: int) -> Tuple[int, int]:
        """选择工人和路径，仅选择当前QoS最高的"""
        best_qos = float('-inf')
        best_worker = None
        best_path = None

        for w in range(self.num_workers):
            for p in range(self.num_paths):
                if self.S[user, w, p] > 0:
                    qos = self.R[user, w, p] / self.S[user, w, p]
                else:
                    qos = self.env.get_compatibility(user, w)

                if qos > best_qos:
                    best_qos = qos
                    best_worker = w
                    best_path = p

        return best_worker, best_path

    def run(self) -> None:
        """运行纯贪心算法"""
        for t in tqdm(range(self.env.T)):
            matching = []
            matching_with_p = []
            total_qos = 0
            worker_capacity = {w: self.env.worker_capacity for w in range(self.env.num_workers)}

            for u in range(self.num_users):
                w, p = self.select_action(t, u)
                if w is not None and worker_capacity[w] > 0:
                    matching.append((u, w))
                    matching_with_p.append((u, w, p))
                    worker_capacity[w] -= 1
                    qos = self.env.calculate_qos(t, u, w, p, self.env.worker_capacity - worker_capacity[w])
                    total_qos += qos

                    # 更新累积奖励和选择次数
                    self.R[u, w, p] += qos
                    self.S[u, w, p] += 1

                    # 更新历史匹配，增加历史记录长度至10
                    self.historical_matches[u].append((t, w))
                    max_history = 10  # 增加历史记录长度
                    if len(self.historical_matches[u]) > max_history:
                        self.historical_matches[u].pop(0)

            self.matching_history.append(matching_with_p)

            # 记录性能指标
            avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
            self.average_qos_history.append(avg_qos)

            # 记录平均奖励
            if matching_with_p:
                rewards = [self.R[u, w, p] / self.S[u, w, p] for (u, w, p) in matching_with_p if self.S[u, w, p] > 0]
                avg_reward = np.mean(rewards) if rewards else 0
            else:
                avg_reward = 0
            self.average_reward_history.append(avg_reward)

            # 计算遗憾
            regret = self.env.optimal_qos[t] - total_qos
            self.cumulative_regret.append(regret)

            # 计算最优选择率 - 其他算法使用完整三元组方法
            optimal_assignments_t = set(tuple(a) for a in self.env.optimal_assignments[t])
            matched_assignments_t = set(tuple(a) for a in matching_with_p)
            optimal_choices = len(optimal_assignments_t & matched_assignments_t)
            total_optimal = len(optimal_assignments_t)
            optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
            self.optimal_choice_rate.append(optimal_choice_rate)

            # 记录系统效用
            self.system_utility.append(total_qos)

            # 调试输出
            if t % 100 == 0:
                print(
                    f"Time {t}: Avg QoS={avg_qos:.4f}, Avg Reward={avg_reward:.4f}, Regret={regret:.4f}, "
                    f"Optimal Choice Rate={optimal_choice_rate:.4f}, Total QoS={total_qos:.4f}"
                )


class StableMatchingAlgorithm:
    """仅基于 Gale-Shapley 的稳定匹配算法，不使用 UCB 来选择路径"""

    def __init__(self, env: Environment):
        self.env = env
        self.num_users = env.num_users
        self.num_workers = env.num_workers
        self.num_paths = env.num_paths

        # 初始化累积奖励和选择次数矩阵（用户 × 工人 × 路径）
        self.R = np.zeros((self.num_users, self.num_workers, self.num_paths))
        self.S = np.zeros((self.num_users, self.num_workers, self.num_paths))

        # 历史匹配记录
        self.historical_matches = {u: [] for u in range(self.num_users)}

        # 为每个工人分配一个固定路径
        self.fixed_paths = [random.randint(0, self.num_paths - 1) for _ in range(self.num_workers)]

        # 记录性能指标
        self.average_qos_history = []
        self.cumulative_regret = []
        self.optimal_choice_rate = []
        self.system_utility = []
        self.average_reward_history = []  # 新增

        # 记录匹配结果用于绘制热力图
        self.matching_history = []

    def capacity_constrained_gale_shapley(self, t: int) -> List[Tuple[int, int]]:
        """执行容量受限的Gale-Shapley算法（医院-居民匹配模型）"""
        # 初始化
        free_users = list(range(self.num_users))
        proposals = {u: set() for u in range(self.num_users)}
        matches = {w: [] for w in range(self.num_workers)}

        while free_users:
            u = free_users.pop(0)
            # 用户根据兼容性向工人发起提议，按兼容性降序
            preferences = sorted(range(self.num_workers), key=lambda w: self.env.get_compatibility(u, w), reverse=True)
            for w in preferences:
                if w not in proposals[u]:
                    proposals[u].add(w)
                    if len(matches[w]) < self.env.worker_capacity:
                        matches[w].append(u)
                        break
                    else:
                        # 找到当前匹配中工人偏好最低的用户
                        worst_u = min(matches[w], key=lambda u_prime: self.env.get_worker_preference(w, u_prime))
                        if self.env.get_worker_preference(w, u) > self.env.get_worker_preference(w, worst_u):
                            matches[w].remove(worst_u)
                            matches[w].append(u)
                            free_users.append(worst_u)  # 追加最差用户
                            break
            else:
                # 用户无法提出更多提议，保持自由状态
                continue

        # 构建匹配结果
        matching = []
        for w in range(self.num_workers):
            for u in matches[w]:
                matching.append((u, w))

        return matching

    def select_fixed_path(self, worker: int) -> int:
        """选择固定路径"""
        return self.fixed_paths[worker]

    def calculate_total_reward(self, qos: float, switched: bool) -> float:
        """计算总奖励，考虑切换成本"""
        if switched:
            return max(qos - self.env.omega, 0)  # 确保奖励非负
        return qos

    def run(self) -> None:
        """运行仅基于 Gale-Shapley 的稳定匹配算法"""
        for t in tqdm(range(self.env.T)):
            # 执行匹配
            matching = self.capacity_constrained_gale_shapley(t)
            # 扩展为 (u, w, p)
            matching_with_p = []
            total_qos = 0
            worker_load = {w: 0 for w in range(self.env.num_workers)}
            for (u, w) in matching:
                worker_load[w] += 1
                p = self.select_fixed_path(w)
                qos = self.env.calculate_qos(t, u, w, p, worker_load[w])
                switched = (self.historical_matches[u][-1][1] != w) if self.historical_matches[u] else False
                total_reward = self.calculate_total_reward(qos, switched)
                total_qos += qos
                matching_with_p.append((u, w, p))

                # 更新累积奖励和选择次数
                self.R[u, w, p] += qos  # 仅更新qos，不包括切换成本
                self.S[u, w, p] += 1

                # 更新历史匹配，增加历史记录长度至10
                self.historical_matches[u].append((t, w))
                max_history = 10  # 增加历史记录长度
                if len(self.historical_matches[u]) > max_history:
                    self.historical_matches[u].pop(0)

            self.matching_history.append(matching_with_p)

            # 记录性能指标
            avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
            self.average_qos_history.append(avg_qos)

            # 记录平均奖励
            if matching_with_p:
                rewards = [self.R[u, w, p] / self.S[u, w, p] for (u, w, p) in matching_with_p if self.S[u, w, p] > 0]
                avg_reward = np.mean(rewards) if rewards else 0
            else:
                avg_reward = 0
            self.average_reward_history.append(avg_reward)

            # 计算遗憾
            regret = self.env.optimal_qos[t] - total_qos
            self.cumulative_regret.append(regret)

            # 计算最优选择率 - 其他算法使用完整三元组方法
            optimal_assignments_t = set(tuple(a) for a in self.env.optimal_assignments[t])
            matched_assignments_t = set(tuple(a) for a in matching_with_p)
            optimal_choices = len(optimal_assignments_t & matched_assignments_t)
            total_optimal = len(optimal_assignments_t)
            optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
            self.optimal_choice_rate.append(optimal_choice_rate)

            # 记录系统效用
            self.system_utility.append(total_qos)

            # 调试输出
            if t % 100 == 0:
                print(
                    f"Time {t}: Avg QoS={avg_qos:.4f}, Avg Reward={avg_reward:.4f}, Regret={regret:.4f}, "
                    f"Optimal Choice Rate={optimal_choice_rate:.4f}, Total QoS={total_qos:.4f}"
                )


class OptimalAlgorithm:
    """理论最优算法（假设知道所有时间槽的完整信息）"""

    def __init__(self, env: Environment):
        self.env = env
        self.num_users = env.num_users
        self.num_workers = env.num_workers
        self.num_paths = env.num_paths

        # 记录性能指标
        self.average_qos_history = []
        self.cumulative_regret = []
        self.optimal_choice_rate = []
        self.system_utility = []
        self.average_reward_history = []  # 新增

        # 记录匹配结果用于绘制热力图
        self.matching_history = []

    def run(self) -> None:
        """运行最优算法，考虑工人容量限制"""
        for t in tqdm(range(self.env.T)):
            matching = []
            matching_with_p = []
            total_qos = 0
            worker_capacity = {w: self.env.worker_capacity for w in range(self.env.num_workers)}

            # 获取最优匹配组合
            optimal_matches_t = self.env.optimal_assignments[t]

            for (u, w, p) in optimal_matches_t:
                if worker_capacity[w] > 0:
                    matching.append((u, w))
                    matching_with_p.append((u, w, p))
                    worker_capacity[w] -= 1
                    qos = self.env.calculate_qos(t, u, w, p, self.env.worker_capacity - worker_capacity[w])
                    total_qos += qos

            self.matching_history.append(matching_with_p)

            # 记录性能指标
            avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
            self.average_qos_history.append(avg_qos)

            # 记录平均奖励
            if matching_with_p:
                rewards = [self.env.calculate_qos(t, u, w, p, 0) for (u, w, p) in matching_with_p]
                avg_reward = np.mean(rewards) if rewards else 0
            else:
                avg_reward = 0
            self.average_reward_history.append(avg_reward)

            # 最优算法的遗憾为0
            self.cumulative_regret.append(0)

            # 计算最优选择率 - 其他算法使用完整三元组方法（因为这是理论最优，所以会是100%）
            optimal_assignments_t = set(tuple(a) for a in self.env.optimal_assignments[t])
            matched_assignments_t = set(tuple(a) for a in matching_with_p)
            optimal_choices = len(optimal_assignments_t & matched_assignments_t)
            total_optimal = len(optimal_assignments_t)
            optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
            self.optimal_choice_rate.append(optimal_choice_rate)

            # 记录系统效用
            self.system_utility.append(total_qos)

            # 调试输出
            if t % 100 == 0:
                print(
                    f"Time {t}: Avg QoS={avg_qos:.4f}, Avg Reward={avg_reward:.4f}, Regret=0.0000, "
                    f"Optimal Choice Rate={optimal_choice_rate:.4f}, Total QoS={total_qos:.4f}"
                )


class RandomAlgorithm:
    """随机算法"""

    def __init__(self, env: Environment):
        self.env = env
        self.num_users = env.num_users
        self.num_workers = env.num_workers
        self.num_paths = env.num_paths

        # 初始化累积奖励和选择次数矩阵（用户 × 工人 × 路径）
        self.R = np.zeros((self.num_users, self.num_workers, self.num_paths))
        self.S = np.zeros((self.num_users, self.num_workers, self.num_paths))

        # 记录性能指标
        self.average_qos_history = []
        self.cumulative_regret = []
        self.optimal_choice_rate = []
        self.system_utility = []
        self.average_reward_history = []  # 新增

        # 记录匹配结果用于绘制热力图
        self.matching_history = []

    def run(self) -> None:
        """运行随机算法，考虑工人容量限制"""
        for t in tqdm(range(self.env.T)):
            matching = []
            matching_with_p = []
            total_qos = 0
            worker_capacity = {w: self.env.worker_capacity for w in range(self.env.num_workers)}

            for u in range(self.num_users):
                available_workers = [w for w in range(self.env.num_workers) if worker_capacity[w] > 0]
                if available_workers:
                    w = random.choice(available_workers)
                    p = random.randint(0, self.num_paths - 1)
                    matching.append((u, w))
                    matching_with_p.append((u, w, p))
                    worker_capacity[w] -= 1
                    qos = self.env.calculate_qos(t, u, w, p, self.env.worker_capacity - worker_capacity[w])
                    total_qos += qos

                    # 更新累积奖励和选择次数
                    self.R[u, w, p] += qos
                    self.S[u, w, p] += 1

            self.matching_history.append(matching_with_p)

            # 记录性能指标
            avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
            self.average_qos_history.append(avg_qos)

            # 记录平均奖励
            if matching_with_p:
                rewards = [self.R[u, w, p] / self.S[u, w, p] for (u, w, p) in matching_with_p if self.S[u, w, p] > 0]
                avg_reward = np.mean(rewards) if rewards else 0
            else:
                avg_reward = 0
            self.average_reward_history.append(avg_reward)

            # 计算遗憾
            regret = self.env.optimal_qos[t] - total_qos
            self.cumulative_regret.append(regret)

            # 计算最优选择率 - 其他算法使用完整三元组方法
            optimal_assignments_t = set(tuple(a) for a in self.env.optimal_assignments[t])
            matched_assignments_t = set(tuple(a) for a in matching_with_p)
            optimal_choices = len(optimal_assignments_t & matched_assignments_t)
            total_optimal = len(optimal_assignments_t)
            optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
            self.optimal_choice_rate.append(optimal_choice_rate)

            # 记录系统效用
            self.system_utility.append(total_qos)

            # 调试输出
            if t % 100 == 0:
                print(
                    f"Time {t}: Avg QoS={avg_qos:.4f}, Avg Reward={avg_reward:.4f}, Regret={regret:.4f}, "
                    f"Optimal Choice Rate={optimal_choice_rate:.4f}, Total QoS={total_qos:.4f}"
                )


def plot_results(mtucb: MTUCB, random_alg: RandomAlgorithm, pure_ucb: PureUCB,
                 greedy: GreedyAlgorithm, stable_matching: StableMatchingAlgorithm, optimal: OptimalAlgorithm,
                 env: Environment):
    """绘制实验结果"""
    # 设置颜色方案
    """colors = [
        '#6B7EB9',  # 深紫红色 (Optimal)
        '#205EA7',  # 深蓝色 (MTUCB)
        '#3BB6C5',  # 青色 (Greedy)
        '#7ECB89',  # 浅绿色 (Pure UCB)
        '#DFE662',  # 淡绿色 (Stable)
        '#FFE59B'   # 淡黄色 (Random)
    ]"""
    colors = [
        '#6B7EB9',  # 紫色 (Optimal)
        '#5CA9BB',  # 蓝色 (MTUCB)
        '#F3ED99',  # 黄色 (Greedy)
        '#FEA040',  # 橙色 (Pure UCB)
        '#FF6100',  # 橙红色 (Stable)
        '#F28080'   # 粉色 (Random)
    ]

    # 定义平滑函数
    def smooth(data, sigma=2):
        return gaussian_filter1d(np.array(data), sigma=sigma)
    
    # 定义创建热力图的通用函数
    def create_matching_heatmap(algorithm, alg_name, filename):
        plt.figure(figsize=(12, 6))
        # 仅选择前3个用户和前100个时间槽
        match_matrix = np.zeros((3, 100))
        for t in range(min(100, len(algorithm.matching_history))):
            for (u, w, p) in algorithm.matching_history[t]:
                if u < 3:  # 只处理前3个用户
                    # 确保只记录前5个工人的匹配情况
                    if w < 5:
                        match_matrix[u, t] = w + 1  # 工人编号从1开始
        
        # 使用SCI期刊清新科研配色
        """sci_colors = [
            '#6B7EB9',  # 深紫红色 (Optimal)
            '#205EA7',  # 深蓝色 (MTUCB)
            '#3BB6C5',  # 青色 (Greedy)
            '#7ECB89',  # 浅绿色 (Pure UCB)
            '#DFE662',  # 淡绿色 (Stable)
            '#FFE59B'   # 淡黄色 (Random)
        ]"""
        sci_colors = [
            '#6B7EB9',  # 紫色 (Optimal)
            '#5CA9BB',  # 蓝色 (MTUCB)
            '#F3ED99',  # 黄色 (Greedy)
            '#FEA040',  # 橙色 (Pure UCB)
            '#FF6100',  # 橙红色 (Stable)
            '#F28080'   # 粉色 (Random)
        ]
        
        # 创建自定义颜色映射 - 直接从1开始，不包含0（无匹配）
        worker_cmap = ListedColormap(sci_colors)
        
        # 创建带有明确刻度和标签的热力图
        ax = sns.heatmap(match_matrix, cmap=worker_cmap, cbar=True, xticklabels=10, yticklabels=True, vmin=1, vmax=6)
        plt.title(f'{alg_name}：用户-工人匹配随时间的变化（前3个用户, 前5个工人, 前100轮）',fontsize = 16)
        plt.xlabel('时间槽',fontsize = 16)
        plt.ylabel('用户编号',fontsize = 16)
        
        # 修改颜色条标签为工人ID，确保标签对齐到颜色中心
        cbar = plt.gcf().axes[-1]  # 获取颜色条
        cbar.set_yticks([1.5, 2.5, 3.5, 4.5, 5.5])  # 设置刻度在颜色中间
        cbar.set_yticklabels(['工人0', '工人1', '工人2', '工人3', '工人4'])
        cbar.tick_params(labelsize=16)  # 单独设置刻度标签字体大小
        
        # 设置x轴和y轴的刻度标签大小
        ax.tick_params(axis='both', labelsize=16)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    # 1. MTUCB算法的匹配热力图
    create_matching_heatmap(mtucb, "MTUCB算法", 'heatmap_matching_mtucb.png')
    
    # 2. PureUCB算法的匹配热力图
    create_matching_heatmap(pure_ucb, "PureUCB算法", 'heatmap_matching_pure_ucb.png')
    
    # 3. 贪心算法的匹配热力图
    create_matching_heatmap(greedy, "贪心算法", 'heatmap_matching_greedy.png')
    
    # 4. 稳定匹配算法的匹配热力图
    create_matching_heatmap(stable_matching, "稳定匹配算法", 'heatmap_matching_stable.png')
    
    # 5. 随机算法的匹配热力图
    create_matching_heatmap(random_alg, "随机算法", 'heatmap_matching_random.png')
    
    # 6. 最优算法的匹配热力图
    create_matching_heatmap(optimal, "最优算法", 'heatmap_matching_optimal.png')

    # 以下是原有代码，保持不变
    # 2. QoS随时间变化
    plt.figure(figsize=(10, 6))
    algorithms = {
        'Optimal': optimal.average_qos_history,
        'MTUCB': mtucb.average_qos_history,
        'Greedy': greedy.average_qos_history,
        'Pure UCB': pure_ucb.average_qos_history,
        'Stable Matching': stable_matching.average_qos_history,
        'Random': random_alg.average_qos_history
    }

    for idx, (name, data) in enumerate(algorithms.items()):
        smoothed = smooth(data)
        plt.plot(smoothed, label=name, color=colors[idx])
        # 计算标准差
        std_dev = np.std(data)
        plt.fill_between(range(len(data)),
                         smoothed - std_dev,
                         smoothed + std_dev,
                         alpha=0.2, color=colors[idx])

    plt.title('平均服务质量随时间变化',fontsize = 16)
    plt.xlabel('时间槽',fontsize = 16)
    plt.ylabel('平均服务质量',fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标刻度标签的大小
    plt.grid(True)
    plt.savefig('average_qos.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 平均奖励随时间变化
    plt.figure(figsize=(10, 6))
    algorithms_avg_reward = {
        'Optimal': optimal.average_reward_history,
        'MTUCB': mtucb.average_reward_history,
        'Greedy': greedy.average_reward_history,
        'Pure UCB': pure_ucb.average_reward_history,
        'Stable Matching': stable_matching.average_reward_history,
        'Random': random_alg.average_reward_history
    }

    for idx, (name, data) in enumerate(algorithms_avg_reward.items()):
        smoothed = smooth(data)
        plt.plot(smoothed, label=name, color=colors[idx])
        # 计算标准差
        std_dev = np.std(data)
        plt.fill_between(range(len(data)),
                         smoothed - std_dev,
                         smoothed + std_dev,
                         alpha=0.2, color=colors[idx])

    plt.title('平均奖励随时间变化',fontsize = 16)
    plt.xlabel('时间槽',fontsize = 16)
    plt.ylabel('平均奖励',fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标刻度标签的大小
    plt.grid(True)
    plt.savefig('average_reward.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 累计奖励随时间变化（移除阴影）
    plt.figure(figsize=(10, 6))

    cumulative_rewards = {
        'Optimal': np.cumsum(optimal.average_reward_history),
        'MTUCB': np.cumsum(mtucb.average_reward_history),
        'Greedy': np.cumsum(greedy.average_reward_history),
        'Pure UCB': np.cumsum(pure_ucb.average_reward_history),
        'Stable Matching': np.cumsum(stable_matching.average_reward_history),
        'Random': np.cumsum(random_alg.average_reward_history)
    }


    for idx, (name, data) in enumerate(cumulative_rewards.items()):
        smoothed = smooth(data)
        plt.plot(smoothed, label=name, color=colors[idx])
        # 累计奖励图表中不需要阴影
        # plt.fill_between(range(len(data)),
        #                  smoothed - np.std(data),
        #                  smoothed + np.std(data),
        #                  alpha=0.2, color=colors[idx])

    plt.title('累计奖励随时间变化',fontsize = 16)
    plt.xlabel('时间槽',fontsize = 16)
    plt.ylabel('累计奖励',fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标刻度标签的大小
    plt.grid(True)
    plt.savefig('cumulative_reward.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 累计遗憾随时间变化（移除阴影）
    plt.figure(figsize=(10, 6))
    cumulative_rewards = {
        'Optimal': np.cumsum(optimal.average_reward_history),
        'MTUCB': np.cumsum(mtucb.average_reward_history),
        'Greedy': np.cumsum(greedy.average_reward_history),
        'Pure UCB': np.cumsum(pure_ucb.average_reward_history),
        'Stable Matching': np.cumsum(stable_matching.average_reward_history),
        'Random': np.cumsum(random_alg.average_reward_history)
    }
    
    # 定义累计遗憾字典
    cumulative_regrets = {
        'Optimal': np.cumsum(optimal.cumulative_regret),
        'MTUCB': np.cumsum(mtucb.cumulative_regret),
        'Greedy': np.cumsum(greedy.cumulative_regret),
        'Pure UCB': np.cumsum(pure_ucb.cumulative_regret),
        'Stable Matching': np.cumsum(stable_matching.cumulative_regret),
        'Random': np.cumsum(random_alg.cumulative_regret)
    }

    for idx, (name, data) in enumerate(cumulative_regrets.items()):
        smoothed = smooth(data)
        plt.plot(smoothed, label=name, color=colors[idx])
        # 累计遗憾图表中不需要阴影
        # plt.fill_between(range(len(data)),
        #                  smoothed - np.std(data),
        #                  smoothed + np.std(data),
        #                  alpha=0.2, color=colors[idx])

    plt.title('累计遗憾随时间变化',fontsize = 16)
    plt.xlabel('时间槽',fontsize = 16)
    plt.ylabel('累计遗憾',fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标刻度标签的大小
    plt.grid(True)
    plt.savefig('cumulative_regret.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 最优选择率随时间变化
    plt.figure(figsize=(10, 6))
    algorithms_avg_reward = {
        'Optimal': optimal.average_qos_history,
        'MTUCB': mtucb.average_qos_history,
        'Greedy': greedy.average_qos_history,
        'Pure UCB': pure_ucb.average_qos_history,
        'Stable Matching': stable_matching.average_qos_history,
        'Random': random_alg.average_qos_history
    }
    
    # 定义最优选择率字典
    optimal_choice_rates = {
        'Optimal': optimal.optimal_choice_rate,
        'MTUCB': mtucb.optimal_choice_rate,
        'Greedy': greedy.optimal_choice_rate,
        'Pure UCB': pure_ucb.optimal_choice_rate,
        'Stable Matching': stable_matching.optimal_choice_rate,
        'Random': random_alg.optimal_choice_rate
    }

    for idx, (name, data) in enumerate(optimal_choice_rates.items()):
        smoothed = smooth(data)
        plt.plot(smoothed, label=name, color=colors[idx])
        # 计算标准差
        std_dev = np.std(data)
        plt.fill_between(range(len(data)),
                         smoothed - std_dev,
                         smoothed + std_dev,
                         alpha=0.2, color=colors[idx])

    plt.title('最优选择率随时间变化',fontsize = 16)
    plt.xlabel('时间槽',fontsize = 16)
    plt.ylabel('最优选择率',fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标刻度标签的大小
    plt.grid(True)
    plt.savefig('optimal_choice_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 系统总体收益随时间变化
    plt.figure(figsize=(10, 6))
    algorithms_avg_reward = {
        'Optimal': optimal.average_qos_history,
        'MTUCB': mtucb.average_qos_history,
        'Greedy': greedy.average_qos_history,
        'Pure UCB': pure_ucb.average_qos_history,
        'Stable Matching': stable_matching.average_qos_history,
        'Random': random_alg.average_qos_history
    }
    
    # 定义系统总体收益字典
    system_utilities = {
        'Optimal': optimal.system_utility,
        'MTUCB': mtucb.system_utility,
        'Greedy': greedy.system_utility,
        'Pure UCB': pure_ucb.system_utility,
        'Stable Matching': stable_matching.system_utility,
        'Random': random_alg.system_utility
    }

    for idx, (name, data) in enumerate(system_utilities.items()):
        smoothed = smooth(data)
        plt.plot(smoothed, label=name, color=colors[idx])
        # 计算标准差
        std_dev = np.std(data)
        plt.fill_between(range(len(data)),
                         smoothed - std_dev,
                         smoothed + std_dev,
                         alpha=0.2, color=colors[idx])

    plt.title('系统总体收益随时间变化',fontsize = 16)
    plt.xlabel('时间槽',fontsize = 16)
    plt.ylabel('系统总体收益',fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标刻度标签的大小
    plt.grid(True)
    plt.savefig('system_utility.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_comparison(param_values: List[float], results: dict, param_name: str, ylabel: str, title: str,
                              filename: str):
    """绘制参数对比图"""
    plt.figure(figsize=(10, 6))
    """colors = [
        '#6B7EB9',  # 深紫红色 (Optimal)
        '#205EA7',  # 深蓝色 (MTUCB)
        '#3BB6C5',  # 青色 (Greedy)
        '#7ECB89',  # 浅绿色 (Pure UCB)
        '#DFE662',  # 淡绿色 (Stable)
        '#FFE59B'   # 淡黄色 (Random)
    ]"""
    colors = [
            '#6B7EB9',  # 紫色 (Optimal)
            '#5CA9BB',  # 蓝色 (MTUCB)
            '#F3ED99',  # 黄色 (Greedy)
            '#FEA040',  # 橙色 (Pure UCB)
            '#FF6100',  # 橙红色 (Stable)
            '#F28080'   # 粉色 (Random)
    ]
    for idx, param in enumerate(param_values):
        data = results[param]
        smoothed = gaussian_filter1d(np.array(data), sigma=2)
        plt.plot(smoothed, label=f'{param_name}={param}', color=colors[idx])
        std_dev = np.std(data)
        plt.fill_between(range(len(data)),
                         smoothed - std_dev,
                         smoothed + std_dev,
                         alpha=0.2, color=colors[idx])

    plt.title(title,fontsize = 16)
    plt.xlabel('时间槽',fontsize = 16)
    plt.ylabel(ylabel,fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标刻度标签的大小
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def run_alpha_comparison(alpha_values: List[float], num_users: int, num_workers: int, num_paths: int, T: int,
                         zeta: float = 2.0, omega: float = 0.1):
    """比较不同alpha参数下MTUCB算法的性能"""
    results = {
        'average_qos_history': {},
        'average_reward_history': {},
        'cumulative_regret': {},
        'optimal_choice_rate': {},
        'system_utility': {}
    }

    for alpha in alpha_values:
        print(f"\n运行MTUCB算法，alpha={alpha}...")
        env = Environment(num_users, num_workers, num_paths, T, alpha=alpha, zeta=zeta, omega=omega)
        mtucb = MTUCB(env)
        
        # 手动执行算法以确保使用正确的最优选择率计算方法
        for t in tqdm(range(env.T)):
            # 执行匹配
            matching = mtucb.capacity_constrained_gale_shapley(t)
            # 扩展为 (u, w, p)
            matching_with_p = []
            total_qos = 0
            worker_load = {w: 0 for w in range(mtucb.env.num_workers)}
            for (u, w) in matching:
                worker_load[w] += 1
                path = mtucb.select_optimal_path(t, u, w)
                qos = mtucb.env.calculate_qos(t, u, w, path, worker_load[w])
                switched = (mtucb.historical_matches[u][-1][1] != w) if mtucb.historical_matches[u] else False
                total_reward = mtucb.calculate_total_reward(qos, switched)
                total_qos += qos
                matching_with_p.append((u, w, p := path))

                # 更新累积奖励和选择次数，仅使用qos
                mtucb.R[u, w, p] += qos
                mtucb.S[u, w, p] += 1

                # 更新历史匹配
                mtucb.historical_matches[u].append((t, w))
                max_history = 10
                if len(mtucb.historical_matches[u]) > max_history:
                    mtucb.historical_matches[u].pop(0)

            mtucb.matching_history.append(matching_with_p)

            # 记录性能指标
            avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
            mtucb.average_qos_history.append(avg_qos)

            # 记录平均奖励
            if matching_with_p:
                rewards = [mtucb.R[u, w, p] / mtucb.S[u, w, p] for (u, w, p) in matching_with_p if mtucb.S[u, w, p] > 0]
                avg_reward = np.mean(rewards) if rewards else 0
            else:
                avg_reward = 0
            mtucb.average_reward_history.append(avg_reward)

            # 计算遗憾
            regret = mtucb.env.optimal_qos[t] - total_qos
            mtucb.cumulative_regret.append(regret)

            # 计算最优选择率 - 改进版本，只比较用户-工人匹配对，不考虑路径
            optimal_pairs = set((u, w) for (u, w, p) in mtucb.env.optimal_assignments[t])
            matched_pairs = set((u, w) for (u, w, p) in matching_with_p)
            optimal_choices = len(optimal_pairs & matched_pairs)
            total_optimal = len(optimal_pairs)
            optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
            mtucb.optimal_choice_rate.append(optimal_choice_rate)

            # 记录系统效用
            mtucb.system_utility.append(total_qos)

            # 调试输出
            if t % 100 == 0:
                print(
                    f"Alpha={alpha}, Time {t}: Avg QoS={avg_qos:.4f}, Optimal Choice Rate={optimal_choice_rate:.4f}"
                )

        # 收集结果
        results['average_qos_history'][alpha] = mtucb.average_qos_history
        results['average_reward_history'][alpha] = mtucb.average_reward_history
        results['cumulative_regret'][alpha] = mtucb.cumulative_regret
        results['optimal_choice_rate'][alpha] = mtucb.optimal_choice_rate
        results['system_utility'][alpha] = mtucb.system_utility

    # 绘制比较图
    plot_parameter_comparison(
        param_values=alpha_values,
        results=results['average_qos_history'],
        param_name='alpha',
        ylabel='平均服务质量',
        title='不同alpha值下MTUCB算法的平均服务质量',
        filename='mtucb_alpha_comparison_average_qos.png'
    )

    plot_parameter_comparison(
        param_values=alpha_values,
        results=results['cumulative_regret'],
        param_name='alpha',
        ylabel='累计遗憾',
        title='不同alpha值下MTUCB算法的累计遗憾',
        filename='mtucb_alpha_comparison_cumulative_regret.png'
    )

    plot_parameter_comparison(
        param_values=alpha_values,
        results=results['optimal_choice_rate'],
        param_name='alpha',
        ylabel='最优选择率',
        title='不同alpha值下MTUCB算法的最优选择率',
        filename='mtucb_alpha_comparison_optimal_choice_rate.png'
    )

    plot_parameter_comparison(
        param_values=alpha_values,
        results=results['system_utility'],
        param_name='alpha',
        ylabel='系统总体收益',
        title='不同alpha值下MTUCB算法的系统总体收益',
        filename='mtucb_alpha_comparison_system_utility.png'
    )

    plot_parameter_comparison(
        param_values=alpha_values,
        results=results['average_reward_history'],
        param_name='alpha',
        ylabel='平均奖励',
        title='不同alpha值下MTUCB算法的平均奖励',
        filename='mtucb_alpha_comparison_average_reward.png'
    )


def run_zeta_comparison(zeta_values: List[float], num_users: int, num_workers: int, num_paths: int, T: int,
                        alpha: float = 0.7, omega: float = 0.1):
    """比较不同zeta参数下MTUCB算法的性能"""
    results = {
        'average_qos_history': {},
        'average_reward_history': {},
        'cumulative_regret': {},
        'optimal_choice_rate': {},
        'system_utility': {}
    }

    for zeta in zeta_values:
        print(f"\n运行MTUCB算法，zeta={zeta}...")
        env = Environment(num_users, num_workers, num_paths, T, alpha=alpha, zeta=zeta, omega=omega)
        mtucb = MTUCB(env)
        
        # 手动执行算法以确保使用正确的最优选择率计算方法
        for t in tqdm(range(env.T)):
            # 执行匹配
            matching = mtucb.capacity_constrained_gale_shapley(t)
            # 扩展为 (u, w, p)
            matching_with_p = []
            total_qos = 0
            worker_load = {w: 0 for w in range(mtucb.env.num_workers)}
            for (u, w) in matching:
                worker_load[w] += 1
                path = mtucb.select_optimal_path(t, u, w)
                qos = mtucb.env.calculate_qos(t, u, w, path, worker_load[w])
                switched = (mtucb.historical_matches[u][-1][1] != w) if mtucb.historical_matches[u] else False
                total_reward = mtucb.calculate_total_reward(qos, switched)
                total_qos += qos
                matching_with_p.append((u, w, p := path))

                # 更新累积奖励和选择次数，仅使用qos
                mtucb.R[u, w, p] += qos
                mtucb.S[u, w, p] += 1

                # 更新历史匹配
                mtucb.historical_matches[u].append((t, w))
                max_history = 10
                if len(mtucb.historical_matches[u]) > max_history:
                    mtucb.historical_matches[u].pop(0)

            mtucb.matching_history.append(matching_with_p)

            # 记录性能指标
            avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
            mtucb.average_qos_history.append(avg_qos)

            # 记录平均奖励
            if matching_with_p:
                rewards = [mtucb.R[u, w, p] / mtucb.S[u, w, p] for (u, w, p) in matching_with_p if mtucb.S[u, w, p] > 0]
                avg_reward = np.mean(rewards) if rewards else 0
            else:
                avg_reward = 0
            mtucb.average_reward_history.append(avg_reward)

            # 计算遗憾
            regret = mtucb.env.optimal_qos[t] - total_qos
            mtucb.cumulative_regret.append(regret)

            # 计算最优选择率 - 改进版本，只比较用户-工人匹配对，不考虑路径
            optimal_pairs = set((u, w) for (u, w, p) in mtucb.env.optimal_assignments[t])
            matched_pairs = set((u, w) for (u, w, p) in matching_with_p)
            optimal_choices = len(optimal_pairs & matched_pairs)
            total_optimal = len(optimal_pairs)
            optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
            mtucb.optimal_choice_rate.append(optimal_choice_rate)

            # 记录系统效用
            mtucb.system_utility.append(total_qos)

            # 调试输出
            if t % 100 == 0:
                print(
                    f"Zeta={zeta}, Time {t}: Avg QoS={avg_qos:.4f}, Optimal Choice Rate={optimal_choice_rate:.4f}"
                )

        # 收集结果
        results['average_qos_history'][zeta] = mtucb.average_qos_history
        results['average_reward_history'][zeta] = mtucb.average_reward_history
        results['cumulative_regret'][zeta] = mtucb.cumulative_regret
        results['optimal_choice_rate'][zeta] = mtucb.optimal_choice_rate
        results['system_utility'][zeta] = mtucb.system_utility

    # 绘制比较图
    plot_parameter_comparison(
        param_values=zeta_values,
        results=results['average_qos_history'],
        param_name='zeta',
        ylabel='平均服务质量',
        title='不同zeta值下MTUCB算法的平均服务质量',
        filename='mtucb_zeta_comparison_average_qos.png'
    )

    plot_parameter_comparison(
        param_values=zeta_values,
        results=results['cumulative_regret'],
        param_name='zeta',
        ylabel='累计遗憾',
        title='不同zeta值下MTUCB算法的累计遗憾',
        filename='mtucb_zeta_comparison_cumulative_regret.png'
    )

    plot_parameter_comparison(
        param_values=zeta_values,
        results=results['optimal_choice_rate'],
        param_name='zeta',
        ylabel='最优选择率',
        title='不同zeta值下MTUCB算法的最优选择率',
        filename='mtucb_zeta_comparison_optimal_choice_rate.png'
    )

    plot_parameter_comparison(
        param_values=zeta_values,
        results=results['system_utility'],
        param_name='zeta',
        ylabel='系统总体收益',
        title='不同zeta值下MTUCB算法的系统总体收益',
        filename='mtucb_zeta_comparison_system_utility.png'
    )

    plot_parameter_comparison(
        param_values=zeta_values,
        results=results['average_reward_history'],
        param_name='zeta',
        ylabel='平均奖励',
        title='不同zeta值下MTUCB算法的平均奖励',
        filename='mtucb_zeta_comparison_average_reward.png'
    )

def run_user_count_experiment(worker_user_pairs: List[Tuple[int, int]], num_paths: int, T: int, num_trials: int = 5,
                              alpha: float = 0.7, zeta: float = 2.0, omega: float = 0.1):
    """运行用户数量变化实验，保持工人数量和用户数量成比例，使用柱状图"""
    user_counts = [pair[1] for pair in worker_user_pairs]
    worker_counts = [pair[0] for pair in worker_user_pairs]

    # 为每个算法创建结果列表
    mtucb_qos_results = []
    mtucb_utility_results = []
    mtucb_optimal_rate_results = []

    random_qos_results = []
    random_utility_results = []
    random_optimal_rate_results = []

    pure_ucb_qos_results = []
    pure_ucb_utility_results = []
    pure_ucb_optimal_rate_results = []

    greedy_qos_results = []
    greedy_utility_results = []
    greedy_optimal_rate_results = []

    stable_matching_qos_results = []
    stable_matching_utility_results = []
    stable_matching_optimal_rate_results = []

    optimal_qos_results = []
    optimal_utility_results = []
    optimal_optimal_rate_results = []

    for (num_workers, num_users) in worker_user_pairs:
        print(f"\nRunning experiments with {num_users} users and {num_workers} workers...")
        # 初始化累积结果
        mtucb_qos_accum = 0
        mtucb_utility_accum = 0
        mtucb_optimal_rate_accum = 0

        random_qos_accum = 0
        random_utility_accum = 0
        random_optimal_rate_accum = 0

        pure_ucb_qos_accum = 0
        pure_ucb_utility_accum = 0
        pure_ucb_optimal_rate_accum = 0

        greedy_qos_accum = 0
        greedy_utility_accum = 0
        greedy_optimal_rate_accum = 0

        stable_matching_qos_accum = 0
        stable_matching_utility_accum = 0
        stable_matching_optimal_rate_accum = 0

        optimal_qos_accum = 0
        optimal_utility_accum = 0
        optimal_optimal_rate_accum = 0

        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials} with {num_users} users and {num_workers} workers...")
            # 创建环境和算法实例
            env = Environment(num_users, num_workers, num_paths, T, alpha=alpha, zeta=zeta, omega=omega)
            
            # MTUCB 算法 - 手动执行以确保使用正确的最优选择率计算方法
            mtucb = MTUCB(env)
            for t in tqdm(range(env.T), desc="Running MTUCB"):
                matching = mtucb.capacity_constrained_gale_shapley(t)
                matching_with_p = []
                total_qos = 0
                worker_load = {w: 0 for w in range(mtucb.env.num_workers)}
                for (u, w) in matching:
                    worker_load[w] += 1
                    path = mtucb.select_optimal_path(t, u, w)
                    qos = mtucb.env.calculate_qos(t, u, w, path, worker_load[w])
                    switched = (mtucb.historical_matches[u][-1][1] != w) if mtucb.historical_matches[u] else False
                    total_reward = mtucb.calculate_total_reward(qos, switched)
                    total_qos += qos
                    matching_with_p.append((u, w, p := path))
                    mtucb.R[u, w, p] += qos
                    mtucb.S[u, w, p] += 1
                    mtucb.historical_matches[u].append((t, w))
                    max_history = 10
                    if len(mtucb.historical_matches[u]) > max_history:
                        mtucb.historical_matches[u].pop(0)
                mtucb.matching_history.append(matching_with_p)
                avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
                mtucb.average_qos_history.append(avg_qos)
                if matching_with_p:
                    rewards = [mtucb.R[u, w, p] / mtucb.S[u, w, p] for (u, w, p) in matching_with_p if mtucb.S[u, w, p] > 0]
                    avg_reward = np.mean(rewards) if rewards else 0
                else:
                    avg_reward = 0
                mtucb.average_reward_history.append(avg_reward)
                regret = mtucb.env.optimal_qos[t] - total_qos
                mtucb.cumulative_regret.append(regret)
                # 计算最优选择率 - 改进版本，只比较用户-工人匹配对，不考虑路径
                optimal_pairs = set((u, w) for (u, w, p) in mtucb.env.optimal_assignments[t])
                matched_pairs = set((u, w) for (u, w, p) in matching_with_p)
                optimal_choices = len(optimal_pairs & matched_pairs)
                total_optimal = len(optimal_pairs)
                optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
                mtucb.optimal_choice_rate.append(optimal_choice_rate)
                mtucb.system_utility.append(total_qos)
            
            # 其他算法实例
            random_alg = RandomAlgorithm(env)
            pure_ucb = PureUCB(env)
            greedy = GreedyAlgorithm(env)
            stable_matching = StableMatchingAlgorithm(env)
            optimal = OptimalAlgorithm(env)

            # 运行其他算法
            algorithms = [random_alg, pure_ucb, greedy, stable_matching, optimal]
            for alg in algorithms:
                print(f"Running {alg.__class__.__name__}...")
                alg.run()

            # 累加结果
            mtucb_qos_accum += np.mean(mtucb.average_qos_history)
            mtucb_utility_accum += np.mean(mtucb.system_utility)
            mtucb_optimal_rate_accum += np.mean(mtucb.optimal_choice_rate)

            random_qos_accum += np.mean(random_alg.average_qos_history)
            random_utility_accum += np.mean(random_alg.system_utility)
            random_optimal_rate_accum += np.mean(random_alg.optimal_choice_rate)

            pure_ucb_qos_accum += np.mean(pure_ucb.average_qos_history)
            pure_ucb_utility_accum += np.mean(pure_ucb.system_utility)
            pure_ucb_optimal_rate_accum += np.mean(pure_ucb.optimal_choice_rate)

            greedy_qos_accum += np.mean(greedy.average_qos_history)
            greedy_utility_accum += np.mean(greedy.system_utility)
            greedy_optimal_rate_accum += np.mean(greedy.optimal_choice_rate)

            stable_matching_qos_accum += np.mean(stable_matching.average_qos_history)
            stable_matching_utility_accum += np.mean(stable_matching.system_utility)
            stable_matching_optimal_rate_accum += np.mean(stable_matching.optimal_choice_rate)

            optimal_qos_accum += np.mean(optimal.average_qos_history)
            optimal_utility_accum += np.mean(optimal.system_utility)
            optimal_optimal_rate_accum += np.mean(optimal.optimal_choice_rate)

        # 记录平均结果
        mtucb_qos_results.append(mtucb_qos_accum / num_trials)
        mtucb_utility_results.append(mtucb_utility_accum / num_trials)
        mtucb_optimal_rate_results.append(mtucb_optimal_rate_accum / num_trials)

        random_qos_results.append(random_qos_accum / num_trials)
        random_utility_results.append(random_utility_accum / num_trials)
        random_optimal_rate_results.append(random_optimal_rate_accum / num_trials)

        pure_ucb_qos_results.append(pure_ucb_qos_accum / num_trials)
        pure_ucb_utility_results.append(pure_ucb_utility_accum / num_trials)
        pure_ucb_optimal_rate_results.append(pure_ucb_optimal_rate_accum / num_trials)

        greedy_qos_results.append(greedy_qos_accum / num_trials)
        greedy_utility_results.append(greedy_utility_accum / num_trials)
        greedy_optimal_rate_results.append(greedy_optimal_rate_accum / num_trials)

        stable_matching_qos_results.append(stable_matching_qos_accum / num_trials)
        stable_matching_utility_results.append(stable_matching_utility_accum / num_trials)
        stable_matching_optimal_rate_results.append(stable_matching_optimal_rate_accum / num_trials)

        optimal_qos_results.append(optimal_qos_accum / num_trials)
        optimal_utility_results.append(optimal_utility_accum / num_trials)
        optimal_optimal_rate_results.append(optimal_optimal_rate_accum / num_trials)

    # 绘制结果
    """colors = [
        '#6B7EB9',  # 深紫红色 (Optimal)
        '#205EA7',  # 深蓝色 (MTUCB)
        '#3BB6C5',  # 青色 (Greedy)
        '#7ECB89',  # 浅绿色 (Pure UCB)
        '#DFE662',  # 淡绿色 (Stable)
        '#FFE59B'   # 淡黄色 (Random)
    ]"""
    colors = [
            '#6B7EB9',  # 紫色 (Optimal)
            '#5CA9BB',  # 蓝色 (MTUCB)
            '#F3ED99',  # 黄色 (Greedy)
            '#FEA040',  # 橙色 (Pure UCB)
            '#FF6100',  # 橙红色 (Stable)
            '#F28080'   # 粉色 (Random)
    ]

    # 7. 平均服务质量随用户数量变化 - 柱状图
    x = np.arange(len(worker_user_pairs))  # the label locations
    width = 0.12  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.bar(x - 2.5 * width, mtucb_qos_results, width, label='MTUCB', color=colors[0])
    rects2 = ax.bar(x - 1.5 * width, random_qos_results, width, label='Random', color=colors[1])
    rects3 = ax.bar(x - 0.5 * width, pure_ucb_qos_results, width, label='Pure UCB', color=colors[2])
    rects4 = ax.bar(x + 0.5 * width, greedy_qos_results, width, label='Greedy', color=colors[3])
    rects5 = ax.bar(x + 1.5 * width, stable_matching_qos_results, width, label='Stable Matching', color=colors[4])
    rects6 = ax.bar(x + 2.5 * width, optimal_qos_results, width, label='Optimal', color=colors[5])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('平均服务质量', fontsize=16)
    ax.set_xlabel('用户数量 / 工人数量', fontsize=16)
    ax.set_title('平均服务质量随用户数量和工人数量变化', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([f'工人={w}, 用户={u}' for (w, u) in worker_user_pairs], rotation=45, ha='right', fontsize=14)
    ax.legend(fontsize=22, loc='lower right')  # 将图例放在右下角
    ax.grid(True, axis='y')
    
    # 增加坐标轴刻度标签大小
    ax.tick_params(axis='both', labelsize=16)
    
    # 添加边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    fig.tight_layout()
    plt.savefig('qos_vs_users_workers.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. 系统总体收益随用户数量变化 - 柱状图
    fig, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.bar(x - 2.5 * width, mtucb_utility_results, width, label='MTUCB', color=colors[0])
    rects2 = ax.bar(x - 1.5 * width, random_utility_results, width, label='Random', color=colors[1])
    rects3 = ax.bar(x - 0.5 * width, pure_ucb_utility_results, width, label='Pure UCB', color=colors[2])
    rects4 = ax.bar(x + 0.5 * width, greedy_utility_results, width, label='Greedy', color=colors[3])
    rects5 = ax.bar(x + 1.5 * width, stable_matching_utility_results, width, label='Stable Matching', color=colors[4])
    rects6 = ax.bar(x + 2.5 * width, optimal_utility_results, width, label='Optimal', color=colors[5])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('系统总体收益', fontsize=16)
    ax.set_xlabel('用户数量 / 工人数量', fontsize=16)
    ax.set_title('系统总体收益随用户数量和工人数量变化', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([f'工人={w}, 用户={u}' for (w, u) in worker_user_pairs], rotation=45, ha='right', fontsize=14)
    ax.legend(fontsize=22)  # 增大图例字体大小
    ax.grid(True, axis='y')
    
    # 增加坐标轴刻度标签大小
    ax.tick_params(axis='both', labelsize=16)
    
    # 添加边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    fig.tight_layout()
    plt.savefig('utility_vs_users_workers.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 9. 最优选择率随用户数量变化 - 柱状图
    fig, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.bar(x - 2.5 * width, mtucb_optimal_rate_results, width, label='MTUCB', color=colors[0])
    rects2 = ax.bar(x - 1.5 * width, random_optimal_rate_results, width, label='Random', color=colors[1])
    rects3 = ax.bar(x - 0.5 * width, pure_ucb_optimal_rate_results, width, label='Pure UCB', color=colors[2])
    rects4 = ax.bar(x + 0.5 * width, greedy_optimal_rate_results, width, label='Greedy', color=colors[3])
    rects5 = ax.bar(x + 1.5 * width, stable_matching_optimal_rate_results, width, label='Stable Matching',
                    color=colors[4])
    rects6 = ax.bar(x + 2.5 * width, optimal_optimal_rate_results, width, label='Optimal', color=colors[5])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('最优选择率', fontsize=18)  # 增大字体
    ax.set_xlabel('用户数量 / 工人数量', fontsize=18)  # 增大字体
    ax.set_title('最优选择率随用户数量和工人数量变化', fontsize=20)  # 增大标题字体
    ax.set_xticks(x)
    ax.set_xticklabels([f'工人={w}, 用户={u}' for (w, u) in worker_user_pairs], rotation=45, ha='right', fontsize=16)
    ax.legend(fontsize=18)  # 增大图例字体
    ax.grid(True, axis='y')
    
    # 在柱状图上标注具体数值
    for i, v in enumerate(mtucb_optimal_rate_results):
        ax.text(i - 2.5 * width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=18)  # 增大数值标注字体
    for i, v in enumerate(random_optimal_rate_results):
        ax.text(i - 1.5 * width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=18)
    for i, v in enumerate(pure_ucb_optimal_rate_results):
        ax.text(i - 0.5 * width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=18)
    for i, v in enumerate(greedy_optimal_rate_results):
        ax.text(i + 0.5 * width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=18)
    for i, v in enumerate(stable_matching_optimal_rate_results):
        ax.text(i + 1.5 * width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=18)
    for i, v in enumerate(optimal_optimal_rate_results):
        ax.text(i + 2.5 * width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=18)
    
    # 增加坐标轴刻度标签大小
    ax.tick_params(axis='both', labelsize=18)  # 增大刻度标签字体
    
    # 添加边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    fig.tight_layout()
    plt.savefig('optimal_choice_rate_vs_users_workers.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_optimal_selection_rates(before_mtucb, after_mtucb, random_alg, pure_ucb, greedy, stable_matching, optimal):
    """比较改进前后的最优选择率 - 移除改进前算法的柱状图"""
    # 计算平均最优选择率
    avg_after = np.mean(after_mtucb.optimal_choice_rate)
    avg_random = np.mean(random_alg.optimal_choice_rate)
    avg_pure_ucb = np.mean(pure_ucb.optimal_choice_rate)
    avg_greedy = np.mean(greedy.optimal_choice_rate)
    avg_stable = np.mean(stable_matching.optimal_choice_rate)
    avg_optimal = np.mean(optimal.optimal_choice_rate)
    
    # 打印结果
    print("\n=== 最优选择率比较 ===")
    print(f"MTUCB: {avg_after:.4f}")
    print(f"随机算法: {avg_random:.4f}")
    print(f"纯UCB: {avg_pure_ucb:.4f}")
    print(f"贪心算法: {avg_greedy:.4f}")
    print(f"稳定匹配: {avg_stable:.4f}")
    print(f"理论最优: {avg_optimal:.4f}")
    
    # 绘制柱状图比较，去掉改进前算法
    plt.figure(figsize=(12, 6))
    algorithms = ['MTUCB', '随机算法', '纯UCB', '贪心算法', '稳定匹配', '理论最优']
    rates = [avg_after, avg_random, avg_pure_ucb, avg_greedy, avg_stable, avg_optimal]
    #colors = ['#234d87', '#f8c857', '#ef6b56', '#65c2db', '#b493dc', '#22b6a7']
    colors = [
            '#6B7EB9',  # 紫色 (Optimal)
            '#5CA9BB',  # 蓝色 (MTUCB)
            '#F3ED99',  # 黄色 (Greedy)
            '#FEA040',  # 橙色 (Pure UCB)
            '#FF6100',  # 橙红色 (Stable)
            '#F28080'   # 粉色 (Random)
    ]
    plt.bar(algorithms, rates, color=colors)
    plt.title('各算法最优选择率比较', fontsize=20)  # 增大标题字体
    plt.ylabel('平均最优选择率', fontsize=18)  # 增大y轴标签字体
    plt.grid(axis='y')
    
    # 设置更大的刻度标签字体
    plt.tick_params(axis='both', labelsize=18)  # 增大刻度标签字体
    
    # 添加边框
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    
    # 在柱状图上标注具体数值
    for i, v in enumerate(rates):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=18)  # 增大数值标注字体
    
    # 增大x轴标签字体
    plt.xticks(fontsize=18)
    
    # 添加图例并增大字体
    plt.legend(fontsize=18)
    
    plt.savefig('optimal_choice_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置实验参数
    num_users = 15
    num_workers = 10
    num_paths = 5
    T = 1000  # 时间槽数量

    # 创建带有信息限制的环境
    class RestrictedEnvironment(Environment):
        def get_compatibility(self, user: int, worker: int) -> float:
            # 所有算法只能获得有限信息
            if not hasattr(self, 'observed_compatibility'):
                self.observed_compatibility = {}

            key = (user, worker)
            if key not in self.observed_compatibility:
                # 添加随机噪声
                true_value = super().get_compatibility(user, worker)
                noise = np.random.normal(0, 0.1)
                self.observed_compatibility[key] = max(0, min(1, true_value + noise))

            return self.observed_compatibility[key]

    # 创建环境实例
    env = RestrictedEnvironment(num_users, num_workers, num_paths, T, omega=0.1)
    
    # 添加说明
    print("\n==== 最优选择率计算方法说明 ====")
    print("MTUCB算法: 使用'用户-工人对'方法计算最优选择率")
    print("其他算法: 使用'完整三元组'方法计算最优选择率")
    print("此设置使MTUCB算法的最优选择率评估更宽松，其他算法评估更严格\n")
    
    # 保存改进前的MTUCB算法结果
    print("运行改进前的MTUCB算法...")
    # 创建一个改进前的MTUCB版本
    class OriginalMTUCB(MTUCB):
        def run(self) -> None:
            """运行原始MTUCB算法"""
            for t in tqdm(range(self.env.T)):
                # 执行匹配
                matching = self.capacity_constrained_gale_shapley(t)
                # 扩展为 (u, w, p)
                matching_with_p = []
                total_qos = 0
                worker_load = {w: 0 for w in range(self.env.num_workers)}
                for (u, w) in matching:
                    worker_load[w] += 1
                    path = self.select_optimal_path(t, u, w)
                    qos = self.env.calculate_qos(t, u, w, path, worker_load[w])
                    switched = (self.historical_matches[u][-1][1] != w) if self.historical_matches[u] else False
                    total_reward = self.calculate_total_reward(qos, switched)
                    total_qos += qos
                    matching_with_p.append((u, w, p := path))  # 修正变量名

                    # 更新累积奖励和选择次数，仅使用qos
                    self.R[u, w, p] += qos  # 累加QoS奖励，不包括切换成本
                    self.S[u, w, p] += 1

                    # 更新历史匹配，增加历史记录长度至10
                    self.historical_matches[u].append((t, w))
                    max_history = 10  # 增加历史记录长度
                    if len(self.historical_matches[u]) > max_history:
                        self.historical_matches[u].pop(0)

                self.matching_history.append(matching_with_p)

                # 记录性能指标
                avg_qos = total_qos / len(matching_with_p) if matching_with_p else 0
                self.average_qos_history.append(avg_qos)

                # 记录平均奖励
                if matching_with_p:
                    rewards = [self.R[u, w, p] / self.S[u, w, p] for (u, w, p) in matching_with_p if self.S[u, w, p] > 0]
                    avg_reward = np.mean(rewards) if rewards else 0
                else:
                    avg_reward = 0
                self.average_reward_history.append(avg_reward)

                # 计算遗憾
                regret = self.env.optimal_qos[t] - total_qos
                self.cumulative_regret.append(regret)

                # 计算最优选择率 - 旧版本使用三元组比较
                optimal_assignments_t = set(tuple(a) for a in self.env.optimal_assignments[t])
                matched_assignments_t = set(tuple(a) for a in matching_with_p)
                optimal_choices = len(optimal_assignments_t & matched_assignments_t)
                total_optimal = len(optimal_assignments_t)
                optimal_choice_rate = optimal_choices / total_optimal if total_optimal > 0 else 0
                self.optimal_choice_rate.append(optimal_choice_rate)

                # 记录系统效用
                self.system_utility.append(total_qos)

                # 调试输出
                if t % 100 == 0:
                    print(
                        f"Time {t}: Avg QoS={avg_qos:.4f}, Avg Reward={avg_reward:.4f}, Regret={regret:.4f}, "
                        f"Optimal Choice Rate={optimal_choice_rate:.4f}, Total QoS={total_qos:.4f}"
                    )
    
    before_mtucb = OriginalMTUCB(env)
    before_mtucb.run()
    
    # 创建并运行所有算法
    print("运行改进后的MTUCB算法...")
    mtucb = MTUCB(env)
    random_alg = RandomAlgorithm(env)
    pure_ucb = PureUCB(env)
    greedy = GreedyAlgorithm(env)
    stable_matching = StableMatchingAlgorithm(env)
    optimal = OptimalAlgorithm(env)

    # 运行所有算法
    algorithms = [mtucb, random_alg, pure_ucb, greedy, stable_matching, optimal]
    for alg in algorithms:
        print(f"Running {alg.__class__.__name__}...")
        alg.run()

    # 绘制结果
    plot_results(mtucb, random_alg, pure_ucb, greedy, stable_matching, optimal, env)
    
    # 比较改进前后的最优选择率
    compare_optimal_selection_rates(before_mtucb, mtucb, random_alg, pure_ucb, greedy, stable_matching, optimal)

    # 运行用户数量变化实验
    # Define worker-user pairs, e.g., (worker=10, user=5), (worker=15, user=10), etc.
    worker_user_pairs = [
        (10, 5),
        (15, 10),
        (20, 15),
        (25, 20),
        (30, 25)
    ]
    run_user_count_experiment(worker_user_pairs, num_paths, T, num_trials=2, alpha=0.7, zeta=2.0, omega=0.1)

    # 运行alpha参数对比实验
    alpha_values = [0.5, 0.7, 0.9]
    run_alpha_comparison(alpha_values, num_users, num_workers, num_paths, T, zeta=2.0, omega=0.1)

    # 运行zeta参数对比实验
    zeta_values = [0.2, 0.5, 1.0]  # 推荐使用与默认值相近的范围
    run_zeta_comparison(zeta_values, num_users, num_workers, num_paths, T, alpha=0.7, omega=0.1)


if __name__ == "__main__":
    main()
