from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

from typing import Tuple, Callable

class Cantilever3dData1:
    '''
    模型来源论文: An efficient 3D topology optimization code written in Matlab
    '''
    def __init__(self,
                xmin: float=0, xmax: float=60, 
                ymin: float=0, ymax: float=20,
                zmin: float=0, zmax: float=4,
                T: float = -1):
        """
           3------- 7
         / |       /|
        1 ------- 5 |
        |  |      | |
        |  2------|-6
        | /       |/
        0 ------- 4
        位移边界条件: x 坐标为 0 的节点全部固定
        载荷: x 坐标为 xmax, y 坐标为 ymin 的节点施加载荷
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax
        self.T = T 
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, 
               self.ymin, self.ymax, 
               self.zmin, self.zmax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        coord = (
            (bm.abs(x - domain[1]) < self.eps) & 
            (bm.abs(y - domain[2]) < self.eps)
        )
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        # val[coord, 1] = self.T
        val = bm.set_at(val, (coord, 1), self.T)

        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        # 这里仍然是固定左边界的位移
        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    def threshold(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y,
                self.is_dirichlet_boundary_dof_z)

    