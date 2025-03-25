"""测试 solver 模块."""

from dataclasses import dataclass
from typing import Literal, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh2d, TriangleMesh, QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            DensityBasedMaterialConfig,
                            DensityBasedMaterialInstance,
                        )
from soptx.pde import Cantilever2dData1, Cantilever2dData2
from soptx.solver import ElasticFEMSolver, AssemblyMethod
from soptx.utils import timer

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch', 'jax']
    pde_type: Literal['cantilever_2d_1', 'cantilever_2d_2']

    elastic_modulus: float
    poisson_ratio: float
    minimal_modulus: float

    domain_length : float
    domain_width : float

    load : float

    volume_fraction: float
    penalty_factor: float

    mesh_type: Literal['uniform_mesh_2d', 'triangle_mesh', 'quadrangle_mesh']
    nx: int
    ny: int
    hx: int
    hy: int
    
    assembly_method: AssemblyMethod
    solver_type: Literal['cg', 'direct'] 
    solver_params: Dict[str, Any] 

def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')
    elif config.backend == 'jax':
        bm.set_backend('jax')

    if config.pde_type == 'cantilever_2d_2':
        pde = Cantilever2dData2(
                    xmin=0, xmax=config.domain_length,
                    ymin=0, ymax=config.domain_width,
                    T = config.load
                )
        if config.mesh_type == 'triangle_mesh':
            mesh = TriangleMesh.from_box(box=pde.domain(), nx=config.nx, ny=config.ny)
    elif config.pde_type == 'cantilever_2d_1':
        extent = [0, config.nx, 0, config.ny]
        h = [config.hx, config.hy]
        origin = [0.0, 0.0]
        pde = Cantilever2dData1(
                    xmin=0, xmax=extent[1] * h[0],
                    ymin=0, ymax=extent[3] * h[1],
                    T = config.load
                )
        if config.mesh_type == 'uniform_mesh_2d':
            mesh = UniformMesh2d(
                        extent=extent, h=h, origin=origin,
                        ipoints_ordering='yx', device='cpu'
                    )
        elif config.mesh_type == 'triangle_mesh':
            mesh = TriangleMesh.from_box(box=pde.domain(), nx=config.nx, ny=config.ny)
        elif config.mesh_type == 'quadrangle_mesh':
            mesh = QuadrangleMesh.from_box(box=pde.domain(), nx=config.nx, ny=config.ny)

    GD = mesh.geo_dimension()
    print(f"NN: {mesh.number_of_nodes()}")
    print(f"NC: {mesh.number_of_cells()}")
    
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
                            plane_assumption="plane_stress",    
                            interpolation_model="SIMP",    
                            penalty_factor=config.penalty_factor
                        )
    
    materials = DensityBasedMaterialInstance(config=material_config)
    
    node = mesh.entity('node')
    kwargs = bm.context(node)
    @cartesian
    def density_func(x: TensorLike):
        val = config.volume_fraction * bm.ones(x.shape[0], **kwargs)
        return val
    rho = space_D.interpolate(u=density_func)
    
    return materials, tensor_space_C, pde, rho

def run_assmeble_time_test(config: TestConfig):
    """测试 SOPTX 中不同的 assembly_method 的效率."""
    materials, tensor_space_C, pde, rho = create_base_components(config)

    solver = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    for i in range(5):
        # 创建计时器
        t = timer(f"{config.assembly_method}")
        next(t)  # 启动计时器
        solver.update_status(rho[:])
        t.send('准备时间')
        K = solver._assemble_global_stiffness_matrix()
        t.send('组装时间')
        t.send(None)

def run_assmeble_exact_test(config: TestConfig):
    """测试 SOPTX 中不同的 assembly_method 的正确性."""
    materials, tensor_space_C, pde, rho = create_base_components(config)

    solver_sta = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.STANDARD,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    solver_voi = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.VOIGT,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    solver_fas = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.FAST,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    solver_sym = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=AssemblyMethod.SYMBOLIC,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    
    solver_sta.update_status(rho[:])
    K_sta = solver_sta._assemble_global_stiffness_matrix()
    K_sta_full = K_sta.toarray()

    solver_voi.update_status(rho[:])
    K_voi = solver_voi._assemble_global_stiffness_matrix()
    K_voi_full = K_voi.toarray()

    solver_fas.update_status(rho[:])
    K_fas = solver_fas._assemble_global_stiffness_matrix()
    K_fas_full = K_fas.toarray()

    solver_sym.update_status(rho[:])
    K_sym = solver_sym._assemble_global_stiffness_matrix()
    K_sym_full = K_sym.toarray()

    print(f"diff_K1: {bm.sum(bm.abs(K_sta_full - K_voi_full))}")
    print(f"diff_K2: {bm.sum(bm.abs(K_voi_full - K_fas_full))}")
    print(f"diff_K3: {bm.sum(bm.abs(K_fas_full - K_sym_full))}")
    print(f"-------------------------------")

def run_solve_uh_time_test(config: TestConfig):
    """测试求解位移的效率"""
    materials, tensor_space_C, pde, rho = create_base_components(config)

    solver = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    for i in range(3):
        # 创建计时器
        t = timer(f"{config.solver_type} Timing")
        next(t)  # 启动计时器
        solver.update_status(rho[:])
        t.send('准备时间')
        solver_result = solver.solve()
        uh = solver_result.displacement
        t.send('求解时间')
        t.send(None)

def run_solve_uh_exact_test(config: TestConfig):
    """测试求解位移的正确性
    与 Efficient topology optimization in MATLAB using 88 lines of code 比较
    cg 和 MUMPS 求解器的结果一致
    """
    materials, tensor_space_C, pde, rho = create_base_components(config)

    # solver_cg = ElasticFEMSolver(
    #                 materials=materials,
    #                 tensor_space=tensor_space_C,
    #                 pde=pde,
    #                 assembly_method=config.assembly_method,
    #                 solver_type='cg',
    #                 solver_params={'maxiter': 2000, 'atol': 1e-12, 'rtol': 1e-12}, 
    #             )
    # solver_cg.update_status(rho[:])
    # solver_result_cg = solver_cg.solve()
    # uh_cg = solver_result_cg.displacement
    # print(f"uh_cg: {bm.mean(uh_cg):.10f}")
    solver_mumps = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type='direct',
                    solver_params={'solver_type': 'mumps'}, 
                )
    solver_mumps.update_status(rho[:])
    K = solver_mumps._assemble_global_stiffness_matrix()
    K_full = K.toarray()
    print(f"K_full: {bm.sum(bm.abs(K_full)):.10f}")
    solver_result_mumps = solver_mumps.solve()
    uh_mumps = solver_result_mumps.displacement
    print(f"uh_mumps: {bm.mean(uh_mumps):.10f}")
    # diff = bm.max(bm.abs(uh_cg - uh_mumps))
    # print(f"Difference between CG and MUMPS : {diff:.6e}")


if __name__ == "__main__":
    config_assemble_time = TestConfig(
                                    backend='numpy',
                                    pde_type='cantilever_2d_2',
                                    elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                                    domain_length=3.0, domain_width=1.0,
                                    load=2000,
                                    volume_fraction=0.5,
                                    penalty_factor=3.0,
                                    mesh_type='triangle_mesh', nx=300, ny=100, hx=1, hy=1,
                                    assembly_method=AssemblyMethod.FAST,
                                    solver_type='direct', solver_params={'solver_type': 'mumps'},
                                )
    
    # mesh_type = 'uniform_mesh_2d'
    mesh_type = 'triangle_mesh'
    config_assmeble_exact = TestConfig(
                                backend='numpy',
                                pde_type='cantilever_2d_1',
                                elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                                domain_length=8, domain_width=5,
                                load=-1,
                                volume_fraction=0.4,
                                penalty_factor=3.0,
                                mesh_type=mesh_type, nx=8, ny=5, hx=1, hy=1,
                                assembly_method=None,
                                solver_type='direct', 
                                solver_params={'solver_type': 'mumps'},
                                )
    config_cg_solve = TestConfig(
                            backend='numpy',
                            pde_type='cantilever_2d_2',
                            elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                            domain_length=3.0, domain_width=1.0,
                            load=2000,
                            volume_fraction=0.5,
                            penalty_factor=3.0,
                            mesh_type='triangle_mesh', nx=300, ny=100, hx=1, hy=1,
                            assembly_method=AssemblyMethod.STANDARD,
                            solver_type='cg',  
                            solver_params={'maxiter': 1000, 'atol': 1e-8, 'rtol': 1e-8}  
                        )
    config_mumps_solve = TestConfig(
                                backend='numpy',
                                pde_type='cantilever_2d_2',
                                elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                                domain_length=3.0, domain_width=1.0,
                                load=2000,
                                volume_fraction=0.5,
                                penalty_factor=3.0,
                                mesh_type='triangle_mesh', nx=300, ny=100, hx=1, hy=1,
                                assembly_method=AssemblyMethod.STANDARD,
                                solver_type='direct', 
                                solver_params={'solver_type': 'mumps'},
                            )
    config_solve_exact_test = TestConfig(
                            backend='numpy',
                            pde_type='cantilever_2d_1',
                            elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                            domain_length=160, domain_width=100,
                            load=-1,
                            volume_fraction=0.4,
                            penalty_factor=3.0,
                            mesh_type='uniform_mesh_2d', nx=160, ny=100, hx=1, hy=1, 
                            assembly_method=AssemblyMethod.FAST,
                            solver_type=None, 
                            solver_params=None,
                        )
    config_solver_time_test= TestConfig(
                            backend='numpy',
                            pde_type='cantilever_2d_1',
                            elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                            domain_length=160, domain_width=100,
                            load=-1,
                            volume_fraction=0.4,
                            penalty_factor=3.0,
                            mesh_type='triangle_mesh', nx=16, ny=10, hx=1, hy=1,
                            assembly_method=None,
                            solver_type='direct', 
                            solver_params={'solver_type': 'mumps'},
                            )
    
    # result = run_solve_uh_exact_test(config_solve_exact_test)
    result = run_solve_uh_time_test
    # result2 = run_solver_assemble_test(config_solver_assmeble)
    # result3 = run_assmeble_time_test(config_assemble_time)
    # result4 = run_assmeble_exact_test(config_assmeble_exact)