from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from soptx.pde import Bridge2dData1
from soptx.material import (
                            LevelSetMaterialConfig,
                            LevelSetMaterialInstance,
                        )
from soptx.solver import ElasticFEMSolver, AssemblyMethod
from soptx.opt import ComplianceObjective, VolumeConstraint

def reinit(rho):

    from scipy.ndimage import distance_transform_edt
    import numpy as np
    
    # 创建填充后的数组
    rho_padded = np.zeros((nely+2, nelx+2))
    rho_padded[1:-1, 1:-1] = rho
    
    # 计算距离变换
    dist_to_struct = distance_transform_edt(1 - rho_padded)  # 从非结构到结构的距离
    dist_to_void = distance_transform_edt(rho_padded)        # 从结构到非结构的距离
    
    # 组合创建水平集函数（结构外为正，结构内为负）
    lsf_padded = (1 - rho_padded) * (dist_to_struct - 0.5) - rho_padded * (dist_to_void - 0.5)
    
    lsf = bm.array((lsf_padded.T).flatten(), **kwargs)
    
    return lsf

def smooth_sens(sens, kernel_size=3, padding_mode='edge'):
    """
    Smooth the sensitivity using convolution with a predefined kernel.

    Parameters:
    - sens ( ndarray - (nely, nelx) ): Sensitivity to be smoothed.
    - kernel_size : The size of the convolution kernel. Default is 3.
    - padding_mode : The mode used for padding. Default is 'edge' 
    which pads with the edge values of the array.

    Returns:
    - smoothed_sens ( ndarray - (nely, nelx) ) : Smoothed sensitivity.
    """
    from scipy.signal import convolve2d
    import numpy as np
    # Convolution filter to smooth the sensitivities
    kernel_value = 1 / (2*kernel_size)
    kernel = kernel_value * bm.array([[0, 1, 0], 
                                        [1, 2, 1], 
                                        [0, 1, 0]], dtype=bm.int32)

    # Apply padding to the sensitivity array
    padded_sens = np.pad(sens, ((1, 1), (1, 1)), mode=padding_mode)

    # Perform the convolution using the padded array and the kernel
    smoothed_sens_np = convolve2d(padded_sens, kernel, mode='valid')

    smoothed_sens = bm.array((smoothed_sens_np.T).flatten(), **kwargs)

    return smoothed_sens

nelx = 6
nely = 3
pde = Bridge2dData1(xmin=0, xmax=6,
                    ymin=0, ymax=3,
                    T=1)
mesh = UniformMesh2d(extent=[0, 6, 0, 3], h=[1, 1], origin=[0, 0],
                    ipoints_ordering='yx', device='cpu')
GD = mesh.geo_dimension()
p = 1
space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
print(f"CGDOF: {tensor_space_C.number_of_global_dofs()}")
space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
print(f"DGDOF: {space_D.number_of_global_dofs()}")

material_config = LevelSetMaterialConfig(
                        elastic_modulus=1,
                        minimal_modulus=1e-5,                 
                        poisson_ratio=0.3,            
                        plane_assumption="plane_stress",    
                    )
materials = LevelSetMaterialInstance(config=material_config)
E = materials.elastic_modulus
nu = materials.poisson_ratio
# lam = materials.lame_lambda
lam = E * nu / ((1 + nu) * (1 - nu))
mu = materials.shear_modulus
nodes = mesh.entity('node')
kwargs = bm.context(nodes)
@cartesian
def density_func(x: TensorLike):
    val = bm.ones(x.shape[0], **kwargs)
    return val
rho = space_D.interpolate(u=density_func)

struct = bm.reshape(rho, (nelx, nely)).T
lsf = reinit(struct)

solver = ElasticFEMSolver(
                materials=materials,
                tensor_space=tensor_space_C,
                pde=pde,
                assembly_method=AssemblyMethod.FAST,
                solver_type='direct',
                solver_params={'solver_type': 'mumps'}, 
            )

shapeSens = bm.zeros_like(struct)
topSens = bm.zeros_like(struct)
objective = ComplianceObjective(solver=solver)
num = 200
ke0 = solver.get_base_local_stiffness_matrix()
ktr0 = solver.get_base_local_trace_matrix()
cell2dof = solver.tensor_space.cell_to_dof()
volreq = 0.3
constraint = VolumeConstraint(solver=solver, volume_fraction=volreq)
for iterNum in range(num):
    solver.update_status(rho[:])
    uh = solver.solve().displacement
    uhe = uh[cell2dof]
    # shapeSens[:] = -objective._compute_element_compliance(u=uh)
    shapeSens[:] = -bm.maximum(struct, 1e-4) * bm.einsum('ci, cik, ck -> c', uhe, ke0, uhe)
    coef = rho[:] * bm.pi/2 * (lam + 2*mu) / mu / (lam + mu)
    topSens[:] = coef * (4*mu * bm.einsum('ci, cik, ck -> c', uhe, ke0, uhe) + \
                            (lam - mu) * bm.einsum('ci, cik, ck -> c', uhe, ktr0, uhe))
    obj_value = objective.fun(rho=rho[:], u=None)
    volfrac = constraint.get_volume_fraction(rho=rho[:])

    print(f'Iter: {iterNum}, Compliance.: {obj_value:.4f}, Volfrac.: {volfrac:.3f}')

    if iterNum > 5 and (abs(volfrac-volreq) < 0.005) and \
        bm.all( bm.abs(objective[iterNum] - objective[iterNum-5:iterNum]) < 0.01 * bm.abs(objective[iterNum]) ):
        break

    if iterNum == 0:
        la = -0.01
        La = 1000
        alpha = 0.9
    else:
        la = la - 1/La * (volfrac - volreq)
        La = alpha * La

    shapeSens = shapeSens - la + 1/La * (volfrac - volreq)
    topSens = topSens + bm.pi * ( la - 1/La * (volfrac - volreq) )

    smooth_shapeSens = smooth_sens(sens=bm.reshape(shapeSens, (nelx, nely)).T)                                
    smooth_topSens = smooth_sens(sens=bm.reshape(topSens, (nelx, nely)).T)

    fixed_nodes_mask = pde.is_fixed_point(nodes)
    fixed_nodes_indices = bm.where(fixed_nodes_mask)[0]

    cell2node = mesh.cell_to_node()
    fixed_cells_mask = bm.any(fixed_nodes_mask[cell2node], axis=1)

    smooth_shapeSens = bm.set_at(smooth_shapeSens, fixed_cells_mask, 0)
    smooth_topSens = bm.set_at(smooth_topSens, fixed_cells_mask, 0)

    v = -smooth_shapeSens
    lsf_2d = bm.reshape(lsf, (nelx+2, nely+2)).T
    g = smooth_topSens*(lsf_2d[1:-1, 1:-1] < 0)
    print("---------------")

