from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import UniformMesh2d
from fealpy.decorator import cartesian
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (ElasticMaterialConfig, ElasticMaterialInstance)
from soptx.pde import Cantilever2dData1, MBBBeam2dData1
from soptx.solver import (ElasticFEMSolver, AssemblyMethod)
from soptx.filter import SensitivityBasicFilter
from soptx.opt import ComplianceObjective, VolumeConstraint
from soptx.opt import OCOptimizer, save_optimization_history

domain_length = 160
domain_width = 160
nx, ny = 160, 100
hx, hy = 1, 1
mesh = UniformMesh2d(
            extent=[0, nx, 0, ny], h=[hx, hy], origin=[0, 0],
            ipoints_ordering='yx', flip_direction=None,
            device='cpu'
        )
space_C = LagrangeFESpace(mesh=mesh, p=1, ctype='C')
tensor_space_C = TensorFunctionSpace(scalar_space=space_C, shape=(-1, 2))
space_D = LagrangeFESpace(mesh=mesh, p=0, ctype='D')


elastic_modulus = 1.0
poisson_ratio = 0.3
minimal_modulus = 1e-9
penalty_factor = 3.0
material_config = ElasticMaterialConfig(
                        elastic_modulus=elastic_modulus,            
                        minimal_modulus=minimal_modulus,         
                        poisson_ratio=poisson_ratio,            
                        plane_assumption="plane_stress",    
                        interpolation_model="SIMP",    
                        penalty_factor=penalty_factor
                    )
materials = ElasticMaterialInstance(config=material_config)


load = -1
pde = Cantilever2dData1(
            xmin=0, xmax=domain_length,
            ymin=0, ymax=domain_width,
            T = load
        )
pde = MBBBeam2dData1(
            xmin=0, xmax=domain_length,
            ymin=0, ymax=domain_width,
            T = load
        )
solver = ElasticFEMSolver(
            materials=materials,
            tensor_space=tensor_space_C,
            pde=pde,
            assembly_method=AssemblyMethod.STANDARD,
            solver_type='direct',
            solver_params={'solver_type': 'mumps'} 
        )

volume_fraction = 0.4
objective = ComplianceObjective(solver=solver)
constraint = VolumeConstraint(solver=solver, 
                            volume_fraction=volume_fraction)


filter_radius = 6.0
filter = SensitivityBasicFilter(mesh=mesh, rmin=filter_radius)


max_iterations = 200
tolerance = 0.01
optimizer = OCOptimizer(
                objective=objective,
                constraint=constraint,
                filter=filter,
                options={'max_iterations': max_iterations, 'tolerance': tolerance,}
            )

node = mesh.entity('node')
kwargs = bm.context(node)
@cartesian
def density_func(x: TensorLike):
    val = volume_fraction * bm.ones(x.shape[0], **kwargs)
    return val
rho = space_D.interpolate(u=density_func)
rho_opt, history = optimizer.optimize(rho=rho[:])

from pathlib import Path
save_dir = '/home/heliang/FEALPy_Development/soptx/soptx/vtu/'
save_path = Path(save_dir)
save_path.mkdir(parents=True, exist_ok=True)
save_optimization_history(mesh, history, str(save_path))