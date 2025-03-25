from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from typing import Tuple, Callable

class Bridge2dData1:
    '''
    简支桥梁模型：两端固定支座，中点受力
    '''
    def __init__(self, 
                xmin: float = 0, xmax: float = 1.0, 
                ymin: float = 0, ymax: float = 0.3,
                T: float = 1):
        """
        位移边界条件：桥梁两端有铰支座
        载荷：桥梁中点施加垂直向下的力 T = 1
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.T = T
        self.eps = 1e-12

    def domain(self) -> list:
        box = [self.xmin, self.xmax, self.ymin, self.ymax]
        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        # 载荷施加在桥梁中点
        mid_x = (domain[0] + domain[1]) / 2
        coord = (
            (bm.abs(x - mid_x) < self.eps) & 
            (bm.abs(y - domain[2]) < self.eps)  # 位于底边中点
        )
        
        kwargs = bm.context(points)
        val = bm.zeros(points.shape, **kwargs)
        val[coord, 1] = self.T  # 垂直向下的力
        
        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        kwargs = bm.context(points)
        return bm.zeros(points.shape, **kwargs)
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        # 桥梁左右两端的底部点
        coord_left = (bm.abs(x - domain[0]) < self.eps) & (bm.abs(y - domain[2]) < self.eps)
        coord_right = (bm.abs(x - domain[1]) < self.eps) & (bm.abs(y - domain[2]) < self.eps)
        
        # 铰支座允许旋转但不允许水平位移
        return coord_left | coord_right
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        # 桥梁左右两端的底部点
        coord_left = (bm.abs(x - domain[0]) < self.eps) & (bm.abs(y - domain[2]) < self.eps)
        coord_right = (bm.abs(x - domain[1]) < self.eps) & (bm.abs(y - domain[2]) < self.eps)
        
        # 铰支座允许旋转但不允许垂直位移
        return coord_left | coord_right
    
    @cartesian
    def threshold(self) -> Tuple[Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)
    
    @cartesian
    def is_fixed_point(self, point: TensorLike) -> TensorLike:
        """判断给定坐标点是否需要保持为固体"""
        domain = self.domain()
        x, y = point[..., 0], point[..., 1]
        
        is_bottom = bm.abs(y - domain[2]) < self.eps
        
        is_left = bm.abs(x - domain[0]) < self.eps
        is_right = bm.abs(x - domain[1]) < self.eps
        is_middle = bm.abs(x - (domain[0] + domain[1])/2) < self.eps
        
        return (is_bottom & (is_left | is_right | is_middle))