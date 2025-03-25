"""测试不同后端、优化器、滤波器、网格下的 2D mbb 梁"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            DensityBasedMaterialConfig,
                            DensityBasedMaterialInstance,
                        )
from soptx.pde import MBBBeam2dData1
from soptx.solver import (ElasticFEMSolver, AssemblyMethod)
from soptx.filter import (SensitivityBasicFilter, 
                          DensityBasicFilter, 
                          HeavisideProjectionBasicFilter)
from soptx.opt import (ComplianceObjective, ComplianceConfig,
                       VolumeConstraint, VolumeConfig)
from soptx.opt import OCOptimizer, MMAOptimizer, save_optimization_history, plot_optimization_history

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch', 'jax']
    pde_type: Literal['mbb_beam_2d_1']

    elastic_modulus: float
    poisson_ratio: float
    minimal_modulus: float

    domain_length : float
    domain_width : float

    load : float

    volume_fraction: float
    penalty_factor: float

    mesh_type: Literal['uniform_mesh_2d', 'triangle_mesh']
    nx: int
    ny: int
    hx: float
    hy: float

    p: int
    
    assembly_method: AssemblyMethod
    solver_type: Literal['cg', 'direct'] 
    solver_params: Dict[str, Any]

    diff_mode: Literal['auto', 'manual'] 

    optimizer_type: Literal['oc', 'mma']
    max_iterations: int
    tolerance: float

    filter_type: Literal['None', 'sensitivity', 'density', 'heaviside']
    filter_radius: float

    save_dir: Union[str, Path]

def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')
    elif config.backend == 'jax':
        bm.set_backend('jax')

    if config.pde_type == 'mbb_beam_2d_1':
        pde = MBBBeam2dData1(
                    xmin=0, xmax=config.domain_length,
                    ymin=0, ymax=config.domain_width,
                    T = config.load
                )
        if config.mesh_type == 'uniform_mesh_2d':
            extent = [0, config.nx, 0, config.ny]
            origin = [0.0, 0.0]
            mesh = UniformMesh2d(
                        extent=extent, h=[config.hx, config.hy], origin=origin,
                        ipoints_ordering='yx', device='cpu'
                    )
        elif config.mesh_type == 'triangle_mesh':
            mesh = TriangleMesh.from_box(box=pde.domain(), 
                                        nx=config.nx, ny=config.ny)

    GD = mesh.geo_dimension()
    
    p = config.p
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    space_D = LagrangeFESpace(mesh=mesh, p=0, ctype='D')
    
    material_config = DensityBasedMaterialConfig(
                            elastic_modulus=config.elastic_modulus,            
                            minimal_modulus=config.minimal_modulus,         
                            poisson_ratio=config.poisson_ratio,            
                            plane_assumption="plane_stress",    
                            interpolation_model="SIMP",    
                            penalty_factor=config.penalty_factor
                        )
    
    materials = DensityBasedMaterialInstance(config=material_config)

    solver = ElasticFEMSolver(
                materials=materials,
                tensor_space=tensor_space_C,
                pde=pde,
                assembly_method=config.assembly_method,
                solver_type=config.solver_type,
                solver_params=config.solver_params 
            )
    
    node = mesh.entity('node')
    kwargs = bm.context(node)
    @cartesian
    def density_func(x: TensorLike):
        val = config.volume_fraction * bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)

    obj_config = ComplianceConfig(diff_mode=config.diff_mode)
    objective = ComplianceObjective(solver=solver, config=obj_config)
    cons_config = VolumeConfig(diff_mode=config.diff_mode)
    constraint = VolumeConstraint(solver=solver, 
                                volume_fraction=config.volume_fraction,
                                config=cons_config)
    
    return pde, rho, objective, constraint

def run_diff_mode_test(config: TestConfig) -> Dict[str, Any]:
    """
    测试自动微分和手动微分计算的结果.
    """
    rho, objective, constraint = create_base_components(config)
    mesh = objective.solver.tensor_space.mesh

    if config.filter_type == 'None':
        filter = None
    elif config.filter_type == 'sensitivity':
        filter = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius) 
    elif config.filter_type == 'density':
        filter = DensityBasicFilter(mesh=mesh, rmin=config.filter_radius)
    elif config.filter_type == 'heaviside':
        filter = HeavisideProjectionBasicFilter(mesh=mesh, rmin=config.filter_radius,
                                            beta=1, max_beta=512, continuation_iter=50)   

    if config.optimizer_type == 'oc':
        optimizer = OCOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=filter,
                        options={
                            'max_iterations': config.max_iterations,
                            'tolerance': config.tolerance,
                        }
                    )
        # 设置高级参数 (可选)
        optimizer.options.set_advanced_options(
                                move_limit=0.2,
                                damping_coef=0.5,
                                initial_lambda=1e9,
                                bisection_tol=1e-3
                            )
    elif config.optimizer_type == 'mma':
        NC = mesh.number_of_cells()
        optimizer = MMAOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=filter,
                        options={
                            'max_iterations': config.max_iterations,
                            'tolerance': config.tolerance,
                        }
                    )
        # 设置高级参数 (可选)
        optimizer.options.set_advanced_options(
                                m=1,
                                n=NC,
                                xmin=bm.zeros((NC, 1)),
                                xmax=bm.ones((NC, 1)),
                                a0=1,
                                a=bm.zeros((1, 1)),
                                c=1e4 * bm.ones((1, 1)),
                                d=bm.zeros((1, 1)),
                            )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    rho_opt, history = optimizer.optimize(rho=rho[:])
    
    # Save results
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_optimization_history(mesh, history, str(save_path))
    
    return {
        'optimal_density': rho_opt,
        'history': history,
        'mesh': mesh
    }

def run_basic_filter_test(config: TestConfig) -> Dict[str, Any]:
    """
    测试 filter 类不同滤波器的正确性.
    """
    pde, rho, objective, constraint = create_base_components(config)
    mesh = objective.solver.tensor_space.mesh

    if config.filter_type == 'None':
        filter = None
    elif config.filter_type == 'sensitivity':
        filter = SensitivityBasicFilter(mesh=mesh, rmin=config.filter_radius, domain=pde.domain()) 
    elif config.filter_type == 'density':
        filter = DensityBasicFilter(mesh=mesh, rmin=config.filter_radius, domain=pde.domain())
    elif config.filter_type == 'heaviside':
        filter = HeavisideProjectionBasicFilter(mesh=mesh, rmin=config.filter_radius, domain=pde.domain())  


    if config.optimizer_type == 'oc':
        optimizer = OCOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=filter,
                        options={
                            'max_iterations': config.max_iterations,
                            'tolerance': config.tolerance,
                        }
                    )
        # 设置高级参数 (可选)
        optimizer.options.set_advanced_options(
                                move_limit=0.2,
                                damping_coef=0.5,
                                initial_lambda=1e9,
                                bisection_tol=1e-3
                            )
    elif config.optimizer_type == 'mma':
        NC = mesh.number_of_cells()
        optimizer = MMAOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=filter,
                        options={
                            'max_iterations': config.max_iterations,
                            'tolerance': config.tolerance,
                        }
                    )
        # 设置高级参数 (可选)
        optimizer.options.set_advanced_options(
                                m=1,
                                n=NC,
                                xmin=bm.zeros((NC, 1)),
                                xmax=bm.ones((NC, 1)),
                                a0=1,
                                a=bm.zeros((1, 1)),
                                c=1e4 * bm.ones((1, 1)),
                                d=bm.zeros((1, 1)),
                            )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    rho_opt, history = optimizer.optimize(rho=rho[:])
    
    # Save results
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_optimization_history(mesh, history, str(save_path))
    plot_optimization_history(history, save_path=str(save_path))
    
    return {
        'optimal_density': rho_opt,
        'history': history,
        'mesh': mesh
    }

if __name__ == "__main__":
    base_dir = '/home/heliang/FEALPy_Development/soptx/soptx/vtu'
    '''
    参数来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    # backend = 'numpy'
    # backend = 'pytorch'
    backend = 'jax'
    pde_type = 'mbb_beam_2d_1'
    mesh_type = 'uniform_mesh_2d'
    # mesh_type = 'triangle_mesh'
    nx, ny = 60, 20
    # nx, ny = 150, 50
    # nx ,ny = 300, 100
    optimizer_type = 'oc'
    # optimizer_type = 'mma'
    filter_type = 'sensitivity'
    # filter_type = 'density'
    filter_radius = nx * 0.04
    # filter_type = 'heaviside'
    # filter_radius = nx * 0.03
    config_basic_filter = TestConfig(
        backend=backend,
        pde_type=pde_type,
        elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
        domain_length=nx, domain_width=ny,
        load=-1,
        volume_fraction=0.5,
        penalty_factor=3.0,
        mesh_type=mesh_type, nx=nx, ny=ny, hx=1, hy=1,
        p = 1,
        assembly_method=AssemblyMethod.FAST,
        solver_type='direct', solver_params={'solver_type': 'mumps'},
        # solver_type='cg', solver_params={'maxiter': 2000, 'atol': 1e-12, 'rtol': 1e-12},
        # diff_mode='manual',
        diff_mode='auto',
        optimizer_type=optimizer_type, max_iterations=200, tolerance=0.01,
        filter_type=filter_type, filter_radius=filter_radius,
        save_dir=f'{base_dir}/{backend}_{pde_type}_{mesh_type}_{optimizer_type}_{filter_type}_{nx*ny}',
        )


    # result_11 = run_diff_mode_test(config_sens_filter_auto)
    result2 = run_basic_filter_test(config_basic_filter)
