"""测试不同后端、优化器、滤波器、网格下的 3D 悬臂梁"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh3d, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            DensityBasedMaterialConfig,
                            DensityBasedMaterialInstance,
                        )
from soptx.pde import Cantilever3dData1
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
    device: Literal['cpu', 'cuda']
    pde_type: Literal['cantilever_3d_1']

    elastic_modulus: float
    poisson_ratio: float
    minimal_modulus: float

    domain_length : float
    domain_width : float
    domain_height : float

    load : float

    volume_fraction: float
    penalty_factor: float

    mesh_type: Literal['uniform_mesh_3d', 'tetrahedron_mesh']
    nx: int
    ny: int
    nz: int
    hx: float
    hy: float
    hz: float

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

    if config.pde_type == 'cantilever_3d_1':
        pde = Cantilever3dData1(
                    xmin=0, xmax=config.domain_length,
                    ymin=0, ymax=config.domain_width,
                    zmin=0, zmax=config.domain_height,
                    T = config.load
                )
        if config.mesh_type == 'uniform_mesh_3d':
            extent = [0, config.nx, 0, config.ny, 0, config.nz]
            origin = [0.0, 0.0, 0.0]
            mesh = UniformMesh3d(
                        extent=extent, h=[config.hx, config.hy, config.hz], origin=origin,
                        ipoints_ordering='zyx',
                        device=config.device
                    )
        elif config.mesh_type == 'tetrahedron_mesh':
            mesh = TetrahedronMesh.from_box(box=pde.domain(), 
                                            nx=config.nx, ny=config.ny, nz=config.nz)

    GD = mesh.geo_dimension()
    
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    print(f"CGDOF: {tensor_space_C.number_of_global_dofs()}")
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    print(f"DGDOF: {space_D.number_of_global_dofs()}")
    
    material_config = DensityBasedMaterialConfig(
                            elastic_modulus=config.elastic_modulus,            
                            minimal_modulus=config.minimal_modulus,         
                            poisson_ratio=config.poisson_ratio,            
                            plane_assumption="3D",    
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
    
    # node = mesh.entity('node')
    # kwargs = bm.context(node)
    @cartesian
    def density_func(x: TensorLike):
        val = config.volume_fraction * bm.ones(x.shape[0])
        # val = config.volume_fraction * bm.ones(x.shape[0], **kwargs)
        # val = bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)

    obj_config = ComplianceConfig(diff_mode=config.diff_mode)
    objective = ComplianceObjective(solver=solver, config=obj_config)
    cons_config = VolumeConfig(diff_mode=config.diff_mode)
    constraint = VolumeConstraint(solver=solver, 
                                volume_fraction=config.volume_fraction,
                                config=cons_config)
    
    return pde, rho, objective, constraint

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
    参数来源论文: An efficient 3D topology optimization code written in Matlab
    '''
    # backend = 'numpy'
    backend = 'pytorch'
    # backend = 'jax'
    # device = 'cpu'
    device = 'cuda'
    pde_type = 'cantilever_3d_1'
    # mesh_type = 'tetrahedron_mesh'
    mesh_type = 'uniform_mesh_3d'
    optimizer_type = 'oc'
    filter_type = 'sensitivity'
    nx, ny, nz = 60, 20, 4
    config_basic_filter = TestConfig(
        backend=backend,
        device=device,
        pde_type=pde_type,
        elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
        domain_length=nx, domain_width=ny, domain_height=nz,
        load=-1,
        volume_fraction=0.3,
        penalty_factor=3.0,
        mesh_type=mesh_type, nx=nx, ny=ny, nz=nz, hx=1, hy=1, hz=1,
        p = 1,
        assembly_method=AssemblyMethod.FAST,
        # assembly_method=AssemblyMethod.STANDARD,
        # assembly_method=AssemblyMethod.SYMBOLIC,
        solver_type='direct', solver_params={'solver_type': 'mumps'},
        # solver_type='cg', solver_params={'maxiter': 5000, 'atol': 1e-12, 'rtol': 1e-12},
        diff_mode='manual',
        # diff_mode='auto',
        optimizer_type=optimizer_type, max_iterations=200, tolerance=0.01,
        filter_type=filter_type, filter_radius=1.5,
        save_dir=f'{base_dir}/{device}_{backend}_{pde_type}_{mesh_type}_{optimizer_type}_{filter_type}_{nx*ny*nz}',
    )

    result = run_basic_filter_test(config_basic_filter )
    