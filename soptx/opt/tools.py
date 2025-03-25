from fealpy.backend import backend_manager as bm

from dataclasses import dataclass, field
from time import time
from typing import List, Optional

from fealpy.typing import TensorLike
from fealpy.mesh import StructuredMesh

@dataclass
class OptimizationHistory:
    """优化过程的历史记录"""
    # 密度场历史
    densities: List[TensorLike] = field(default_factory=list)
    # 目标函数值历史
    obj_values: List[float] = field(default_factory=list)
    # 约束函数值历史（如体积分数）
    con_values: List[float] = field(default_factory=list)
    # 迭代时间历史
    iteration_times: List[float] = field(default_factory=list)
    # 优化开始时间
    start_time: float = field(default_factory=time)
    
    def log_iteration(self, 
                     iter_idx: int, 
                     obj_val: float, 
                     volfrac: float, 
                     change: float, 
                     time_cost: float, 
                     density: TensorLike,
                     verbose: bool = True) -> None:
        """记录一次迭代的信息"""
        self.densities.append(bm.copy(density))
        self.obj_values.append(obj_val)
        self.con_values.append(volfrac)
        self.iteration_times.append(time_cost)
        
        if verbose:
            print(f"Iteration: {iter_idx + 1}, "
                  f"Objective: {obj_val:.4f}, "
                  f"Volfrac: {volfrac:.4f}, "
                  f"Change: {change:.4f}, "
                  f"Time: {time_cost:.3f} sec")
    
    def get_total_time(self) -> float:
        """获取总优化时间"""
        return time() - self.start_time
    
    def get_average_iteration_time(self) -> float:
        """获取平均每次迭代时间（排除第一次）"""
        if len(self.iteration_times) <= 1:
            return 0.0
        return sum(self.iteration_times[1:]) / (len(self.iteration_times) - 1)
    
    def print_time_statistics(self) -> None:
        """打印时间统计信息"""
        total_time = self.get_total_time()
        avg_time = self.get_average_iteration_time()
        
        print("\nTime Statistics:")
        print(f"Total optimization time: {total_time:.3f} sec")
        if len(self.iteration_times) > 0:
            print(f"First iteration time: {self.iteration_times[0]:.3f} sec")
        if len(self.iteration_times) > 1:
            print(f"Average iteration time (excluding first): {avg_time:.3f} sec")
            print(f"Number of iterations: {len(self.iteration_times)}")
    
    def get_best_iteration(self, minimize: bool = True) -> int:
        """获取最优迭代的索引"""
        if not self.obj_values:
            return -1
        
        if minimize:
            return self.obj_values.index(min(self.obj_values))
        else:
            return self.obj_values.index(max(self.obj_values))
    
    def get_best_density(self, minimize: bool = True) -> Optional[TensorLike]:
        """获取最优迭代的密度场"""
        best_idx = self.get_best_iteration(minimize)
        if best_idx >= 0 and best_idx < len(self.densities):
            return self.densities[best_idx]
        return None

def save_optimization_history(mesh, history, save_path=None):
    """保存优化过程的所有迭代结果
    
    Parameters
    ----------
    mesh : StructuredMesh or TetrahedronMesh
        有限元网格对象
    history : OptimizationHistory
        优化历史记录，包含每次迭代的密度场
    save_path : str, optional
        保存路径，如不提供则不保存，默认为 None
    """
    if save_path is None:
        return
        
    for i, density in enumerate(history.densities):
        mesh.celldata['density'] = density
        if isinstance(mesh, StructuredMesh):
            mesh.to_vtk(f"{save_path}/density_iter_{i:03d}.vts")
        else:  
            mesh.to_vtk(f"{save_path}/density_iter_{i:03d}.vtu")

def plot_optimization_history(history, save_path=None, show=True, 
                            fontsize=14, title_fontsize=16, 
                            figsize=(12, 8), linewidth=2,
                            ):
    """绘制优化过程中目标函数和约束函数的变化
    
    Parameters
    ----------
    history : OptimizationHistory
        优化历史记录
    save_path : str, optional
        保存路径，如不提供则不保存
    show : bool, optional
        是否显示图像，默认为 True
    fontsize : int, optional
        标签和刻度字体大小，默认为 14
    title_fontsize : int, optional
        标题字体大小，默认为 16
    figsize : tuple, optional
        图形大小，默认为 (12, 8)
    linewidth : float, optional
        线条宽度，默认为 2
    """
    import matplotlib.pyplot as plt
    
    # 准备数据
    iterations = bm.arange(1, len(history.obj_values) + 1)
    obj_values = bm.array(history.obj_values)
    con_values = bm.array(history.con_values)
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': fontsize})
    
    # 绘制目标函数曲线（左轴）
    ax1.set_xlabel('Iteration', fontsize=fontsize)
    ax1.set_ylabel('Compliance, c', color='red', fontsize=fontsize)
    ax1.plot(iterations, obj_values, 'r-', label='c', linewidth=linewidth)
    ax1.tick_params(axis='y', labelcolor='red', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    
    # 创建右轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Volume, v', color='blue', fontsize=fontsize)
    ax2.plot(iterations, con_values, 'b--', label='v', linewidth=linewidth)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=fontsize)
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 添加标题
    plt.title('Optimization History', fontsize=title_fontsize, pad=20)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=fontsize)
    
    # 创建放大子图
    # 找到适合放大的范围
    if len(iterations) > 20:
        start_idx = min(60, len(iterations) - 1)
        if start_idx > 0:  # 确保有足够的数据点
            end_idx = min(120, len(iterations))
            
            # 只有当有足够的数据点时才创建子图
            if end_idx - start_idx > 10:
                sub_ax = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # [left, bottom, width, height]
                sub_ax.plot(iterations[start_idx:end_idx], obj_values[start_idx:end_idx], 'r-', linewidth=linewidth)
                sub_ax.set_xlim(iterations[start_idx], iterations[end_idx-1])
                # 设置 y 轴范围略大于数据范围
                if start_idx < end_idx:
                    y_min = min(obj_values[start_idx:end_idx]) * 0.999
                    y_max = max(obj_values[start_idx:end_idx]) * 1.001
                    sub_ax.set_ylim(y_min, y_max)
                sub_ax.grid(True, linestyle='--', alpha=0.4)
                sub_ax.tick_params(labelsize=fontsize-2)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close()