"""测试 opt 模块中的 compliance 和 volume 类."""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            DensityBasedMaterialConfig,
                            DensityBasedMaterialInstance,
                        )
from soptx.pde import Cantilever2dData1, Cantilever2dData2, MBBBeam2dData1
from soptx.solver import (ElasticFEMSolver, AssemblyMethod)
from soptx.opt import ComplianceObjective, VolumeConstraint
from soptx.utils import timer

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch', 'jax']
    pde_type: Literal['cantilever_2d_1', 'cantilever_2d_2', 'mbb_beam_2d_1']

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
    
    assembly_method: AssemblyMethod
    solver_type: Literal['cg', 'direct'] 
    solver_params: Dict[str, Any]

    diff_mode: Literal['auto', 'manual'] 

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
        origin = [0.0, 0.0]
        pde = Cantilever2dData1(
                    xmin=0, xmax=extent[1] * hx,
                    ymin=0, ymax=extent[3] * hy
                )
        if config.mesh_type == 'uniform_mesh_2d':
            mesh = UniformMesh2d(
                        extent=extent, h=[hx, hy], origin=origin,
                        ipoints_ordering='yx', flip_direction=None,
                        device='cpu'
                    )
    elif config.pde_type == 'mbb_beam_2d_1':
        extent = [0, config.nx, 0, config.ny]
        origin = [0.0, 0.0]
        pde = MBBBeam2dData1(
                    xmin=0, xmax=extent[1] * hx,
                    ymin=0, ymax=extent[3] * hy,
                    T=-1,
                )
        if config.mesh_type == 'uniform_mesh_2d':
            mesh = UniformMesh2d(
                        extent=extent, h=[hx, hy], origin=origin,
                        ipoints_ordering='yx', flip_direction=None,
                        device='cpu'
                    )

    GD = mesh.geo_dimension()
    
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
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
    
    array = config.volume_fraction * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho = space_D.function(array)
    
    return solver, rho

def run_compliane_exact_test(config: TestConfig) -> Dict[str, Any]:
    """
    基于 Efficient topology optimization in MATLAB using 88 lines of code 的结果,
        测试 compliance 类计算柔顺度的正确性
    """
    solver, rho = create_base_components(config)

    objective = ComplianceObjective(solver=solver)

    obj_value = objective.fun(rho=rho[:], u=None)
    print(f"Objective function value: {obj_value:.6e}")
    
    ce = objective.get_element_compliance()
    print(f"\nElement compliance information:")
    print(f"- Shape: {ce.shape}:\n {ce}")
    print(f"- Min: {bm.min(ce):.6e}")
    print(f"- Max: {bm.max(ce):.6e}")
    print(f"- Mean: {bm.mean(ce):.6e}")
    
    dce = objective.jac(rho=rho[:], u=None, diff_mode=config.diff_mode)
    print(f"\nElement compliance_diff information:")
    print(f"- Shape: {dce.shape}:\n, {dce}")
    print(f"- Min: {bm.min(dce):.6e}")
    print(f"- Max: {bm.max(dce):.6e}")
    print(f"- Mean: {bm.mean(dce):.6e}")

def run_volume_exact_test(config: TestConfig) -> Dict[str, Any]:
    """
    基于 Efficient topology optimization in MATLAB using 88 lines of code 的结果,
        测试 volume 类计算体积约束的正确性.
    """
    solver, rho = create_base_components(config)

    constraint = VolumeConstraint(solver=solver, 
                                volume_fraction=config.volume_fraction)
    print(f"Constraint type: {constraint.constraint_type}")
    
    cons_value = constraint.fun(rho=rho[:], u=None)
    print(f"Volume constraint value: {cons_value:.6e}")
    
    dge = constraint.jac(rho=rho[:], u=None, diff_mode=config.diff_mode)
    print(f"\nElement volume_diff information:")
    print(f"- Shape: {dge.shape}:\n, {dge}")
    print(f"- Min: {bm.min(dge):.6e}")
    print(f"- Max: {bm.max(dge):.6e}")
    print(f"- Mean: {bm.mean(dge):.6e}")

def run_diff_mode_test(config: TestConfig):
    """测试自动微分的可行性与正确性."""
    solver, rho = create_base_components(config)

    objective = ComplianceObjective(solver=solver)
    constraint = VolumeConstraint(solver=solver, 
                                volume_fraction=config.volume_fraction)

    t = timer(f"dce Timing")
    next(t) 
    dce_auto = objective.jac(rho=rho[:], u=None, diff_mode='auto')
    print(f"dce_auto: {dce_auto}")
    t.send('auto time')
    dce_manual = objective.jac(rho=rho[:], u=None, diff_mode='manual')
    print(f"dce_manual: {dce_manual}")
    t.send('manual time')
    t.send(None)

    t = timer(f"dge Timing")
    next(t)
    dge_auto = constraint.jac(rho=rho[:], u=None, diff_mode='auto')
    print(f"dge_auto: {dge_auto}")
    t.send('auto time')
    dge_manual = constraint.jac(rho=rho[:], u=None, diff_mode='manual')
    print(f"dge_manual: {dge_manual}")
    t.send('manual time')
    t.send(None)

    diff_obj = bm.max(bm.abs(dce_auto - dce_manual))
    print(f"Difference Objective_diff between auto and manual : {diff_obj:.6e}")
    diff_cons = bm.max(bm.abs(dge_auto - dge_manual))
    print(f"Difference Constraint_diff between auto and manual : {diff_cons:.6e}")
    print("--------------------------------")


if __name__ == "__main__":
    config = TestConfig(
                    backend='numpy',
                    pde_type='cantilever_2d_2',
                    elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                    domain_length=3.0, domain_width=1.0,
                    load=2000,
                    volume_fraction=0.5,
                    penalty_factor=3.0,
                    mesh_type='triangle_mesh', nx=300, ny=100, hx=1, hy=1,
                    assembly_method=AssemblyMethod.STANDARD,
                    solver_type='direct', solver_params={'solver_type': 'mumps'},
                    diff_mode='manual',
                )
    pde_type = 'mbb_beam_2d_1'
    optimizer_type = 'oc'
    filter_type = 'sensitivity'
    nx, ny = 15, 5
    hx, hy = 1, 1
    volfrac = 0.5
    config_compliance_volume_exact_test = TestConfig(
                                backend='jax',
                                pde_type=pde_type,
                                elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                                domain_length=nx, domain_width=ny,
                                load=-1,
                                volume_fraction=volfrac,
                                penalty_factor=3.0,
                                mesh_type='uniform_mesh_2d', nx=nx, ny=ny, hx=hx, hy=hy,
                                assembly_method=AssemblyMethod.FAST,
                                solver_type='direct', solver_params={'solver_type': 'mumps'},
                                diff_mode="manual",
                            )
    backend = 'jax'
    # backend = 'pytorch'
    config_diff_mode_test = TestConfig(
                                backend=backend,
                                pde_type='mbb_beam_2d_1',
                                elastic_modulus=1, poisson_ratio=0.3, minimal_modulus=1e-9,
                                domain_length=nx, domain_width=ny,
                                load=-1,
                                volume_fraction=0.5,
                                penalty_factor=3.0,
                                mesh_type='uniform_mesh_2d', nx=15, ny=10, hx=1, hy=1,
                                assembly_method=AssemblyMethod.FAST,
                                solver_type='direct', solver_params={'solver_type': 'mumps'},
                                diff_mode=None,
                            )
    
    # result1 = run_compliane_exact_test(config_compliance_volume_exact_test)
    # result2 = run_volume_exact_test(config_compliance_volume_exact_test)
    result3 = run_diff_mode_test(config_diff_mode_test)
    