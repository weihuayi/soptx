from typing import Tuple, Set, List
from math import ceil, sqrt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import Mesh
from fealpy.typing import TensorLike
from fealpy.sparse import COOTensor, CSRTensor
from soptx.utils import timer

class FilterMatrixBuilder:
    """负责构建拓扑优化中使用的稀疏过滤矩阵 H"""
    def __init__(self, mesh: Mesh, rmin: float):
        if rmin <= 0:
            raise ValueError("Filter radius must be positive")
        
        self.mesh = mesh
        self.rmin = rmin
        self.device = mesh.device

    def build(self) -> Tuple[CSRTensor, TensorLike]:
        """构建并返回过滤矩阵 H 和其行和 Hs"""
        mesh_keys: Set[str] = set(self.mesh.meshdata.keys())
        
        keys_3d: Set[str] = {'nx', 'ny', 'nz', 'hx', 'hy', 'hz'}
        keys_2d: Set[str] = {'nx', 'ny', 'hx', 'hy'}

        if keys_3d.issubset(mesh_keys) and self.mesh.meshdata['mesh_type'] == 'hexahedron':
            return self._compute_filter_3d(
                            self.rmin,
                            self.mesh.meshdata['nx'], self.mesh.meshdata['ny'], self.mesh.meshdata['nz'],
                            self.mesh.meshdata['hx'], self.mesh.meshdata['hy'], self.mesh.meshdata['hz'],
                        )
        elif keys_2d.issubset(mesh_keys) and self.mesh.meshdata['mesh_type'] in {'quadrangle'}:
            return self._compute_filter_2d(
                                self.rmin,
                                self.mesh.meshdata['nx'], self.mesh.meshdata['ny'],
                                self.mesh.meshdata['hx'], self.mesh.meshdata['hy'], 
                            )
        else:
            return self._compute_filter_general(
                                self.rmin,
                                self.mesh.meshdata['domain'], 
                                self.mesh.entity_barycenter('cell'), 
                            )
        
    def _compute_filter_general(self, 
                                rmin: float,
                                domain: List[float],
                                cell_centers: TensorLike,
                                periodic: List[bool]=[False, False, False],
                                enable_timing: bool = True,
                            ) -> Tuple[COOTensor, TensorLike]:
        """计算任意网格的过滤矩阵, 即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算
        
        Parameters:
        -----------
        rmin: 过滤半径
        cell_centers: 单元中心点坐标, 形状为 (NC, GD)
        domain: 计算域的边界, 
        periodic: 各方向是否周期性, 默认为 [False, False, False]
            
        Returns:
        --------
        H: 过滤矩阵, 形状为 (NC, NC)
        Hs: 过滤矩阵行和, 形状为 (NC, )
        """
        t = None
        if enable_timing:
            t = timer(f"Filter_general")
            next(t)

        # 使用 KD-tree 查询临近点
        # ! 该函数会将目前许需要将变脸 转移到 CPU 上进行计算
        cell_centers = bm.device_put(cell_centers, 'cpu')
        cell_indices, neighbor_indices = bm.query_point(
                                            x=cell_centers, y=cell_centers, h=rmin, 
                                            box_size=domain, mask_self=False, periodic=periodic
                                        )
        if enable_timing:
            t.send('KD-tree 查询时间')

        # 计算节点总数
        NC = cell_centers.shape[0]
        
        # 准备存储过滤器矩阵的数组
        # 预估非零元素的数量（包括对角线元素）
        max_nnz = len(cell_indices) + NC
        iH = bm.zeros(max_nnz, dtype=bm.int32)
        jH = bm.zeros(max_nnz, dtype=bm.int32)
        sH = bm.zeros(max_nnz, dtype=bm.float64)
        
        # 首先添加对角线元素（自身单元）
        for i in range(NC):
            iH[i] = i
            jH[i] = i
            # 自身权重为 rmin（最大权重）
            sH[i] = rmin
        
        # 当前非零元素计数
        nnz = NC
        
        # 填充其余非零元素（邻居点）
        for idx in range(len(cell_indices)):
            i = cell_indices[idx]
            j = neighbor_indices[idx]
            
            # 计算节点间的物理距离
            physical_dist = bm.sqrt(bm.sum((cell_centers[i] - cell_centers[j])**2))
            
            # 计算权重因子
            fac = rmin - physical_dist
            
            if fac > 0:
                iH[nnz] = i
                jH[nnz] = j
                sH[nnz] = fac
                nnz += 1

        if enable_timing:
            t.send('循环计算时间')
        
        # 创建稀疏矩阵（只使用有效的非零元素）
        H = COOTensor(
            indices=bm.astype(bm.stack((iH[:nnz], jH[:nnz]), axis=0), bm.int32),
            values=sH[:nnz],
            spshape=(NC, NC)
        )

        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device='cpu')
        
        H = H.tocsr()
        H = H.device_put(self.device)
        Hs = bm.device_put(Hs, self.device)
        
        # # 转换为 CSR 格式以便于后续操作
        # H = H.tocsr()
        
        # # 计算滤波矩阵行和
        # Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)

        if enable_timing:
            t.send('稀疏矩阵构建时间')
            t.send(None)

        return H, Hs
    
    def _compute_filter_2d(self,
                        rmin: float,     
                        nx: int, ny: int, 
                        hx: float, hy: float,
                        enable_timing: bool = True,
                    ) -> Tuple[CSRTensor, TensorLike]:
        """高性能优化的 2D 滤波矩阵计算 - 分块处理版本
        即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算"""

        NC = self.mesh.number_of_cells()
        expected_NC = nx * ny
        assert NC == expected_NC, (
            f"'_compute_filter_2d' is strictly for simple quadrangle grids. "
            f"Expected {expected_NC} cells (nx*ny), but found {NC}."
        )
        
        
        t = None
        if enable_timing:
            t = timer(f"Filter_2d")
            next(t)
        
        search_radius_x = ceil(rmin/hx)
        search_radius_y = ceil(rmin/hy)
        
        # 批处理单元, 避免一次处理所有单元耗尽内存
        batch_size = min(10000, nx * ny) 
        n_batches = (nx * ny + batch_size - 1) // batch_size 
        
        # 创建一个映射函数，从线性索引转换为 2D 坐标
        def linear_to_2d(linear_idx):
            i = linear_idx // ny  
            j = linear_idx % ny   
            return i, j
        
        # 预计算每个格子的物理坐标 
        all_coords = bm.zeros((nx * ny, 2), dtype=bm.float64, device='cpu')

        for idx in range(nx * ny):
            i, j = linear_to_2d(idx)
            all_coords[idx, 0] = i * hx  
            all_coords[idx, 1] = j * hy  
        
        if enable_timing:
            t.send('预处理')
        
        # 初始化存储结果的列表
        all_rows = [] 
        all_cols = []  
        all_vals = []  
        
        # 分批处理所有单元
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, nx * ny)
            
            batch_rows = []
            batch_cols = []
            batch_vals = []
            
            # 获取当前批次单元的物理坐标
            batch_coords = all_coords[start_idx:end_idx]
            
            # 处理当前批次中的每个单元
            for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
                i, j = linear_to_2d(global_idx)
                row = global_idx  
                
                # 计算搜索范围 - 与原始函数完全相同
                ii1 = max(0, i - (search_radius_x - 1))
                ii2 = min(nx, i + search_radius_x)
                jj1 = max(0, j - (search_radius_y - 1))
                jj2 = min(ny, j + search_radius_y)
                
                # 创建搜索范围内所有单元的线性索引
                search_indices = []
                for ii in range(ii1, ii2):
                    for jj in range(jj1, jj2):
                        col = ii * ny + jj
                        search_indices.append(col)
                
                if not search_indices:
                    continue
                
                # 获取搜索单元的物理坐标
                search_coords = all_coords[search_indices]
                
                # 计算与当前单元的距离
                current_coords = batch_coords[local_idx].reshape(1, 2)  
                diffs = search_coords - current_coords 
                squared_dists = bm.sum(diffs * diffs, axis=1)  
                distances = bm.sqrt(squared_dists)
                
                # 计算滤波因子
                factors = rmin - distances 
                valid_mask = factors > 0  
                
                # 只保留有效的单元对
                if bm.any(valid_mask):
                    valid_cols = bm.array(search_indices, device='cpu')[valid_mask]
                    valid_factors = factors[valid_mask]
                    
                    # 收集结果
                    batch_rows.extend([row] * len(valid_cols))
                    batch_cols.extend(valid_cols.tolist())
                    batch_vals.extend(valid_factors.tolist())
            
            # 将当前批次结果添加到总结果
            all_rows.extend(batch_rows)
            all_cols.extend(batch_cols)
            all_vals.extend(batch_vals)

        
        if enable_timing:
            t.send('计算距离和过滤矩阵')
        
        if all_rows:
            iH = bm.tensor(all_rows, dtype=bm.int32, device='cpu')
            jH = bm.tensor(all_cols, dtype=bm.int32, device='cpu')
            sH = bm.tensor(all_vals, dtype=bm.float64, device='cpu')
        else:
            iH = bm.tensor([], dtype=bm.int32, device='cpu')
            jH = bm.tensor([], dtype=bm.int32, device='cpu')
            sH = bm.tensor([], dtype=bm.float64, device='cpu')
        
        H = COOTensor(
            indices=bm.stack((iH, jH), axis=0),
            values=sH,
            spshape=(nx * ny, nx * ny)
        )
        
        # ! PyTorch 后端对 COOTensor 的支持更好
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device='cpu')
        
        H = H.tocsr()
        H = H.device_put(self.device)
        Hs = bm.device_put(Hs, self.device)

        if enable_timing:
            t.send('矩阵构建')
            t.send(None)
        
        return H, Hs

    def _compute_filter_2d_math(self,
                                rmin: float, 
                                nx: int, ny: int, 
                                hx: float, hy: float,
                                enable_timing: bool = True,
                            ) -> Tuple[COOTensor, TensorLike]:
        """计算 2D 滤波矩阵 - 数学原理的直接实现版本
            即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算"""

        NC = self.mesh.number_of_cells()
        expected_NC = nx * ny
        assert NC == expected_NC, (
            f"'_compute_filter_2d' is strictly for simple quadrangle grids. "
            f"Expected {expected_NC} cells (nx*ny), but found {NC}."
        )

        t = None
        if enable_timing:
            t = timer(f"Filter_2d_math")
            next(t)

        min_h = min(hx, hy)
        max_cells = ceil(rmin/min_h)
        nfilter = int(nx * ny * ((2 * (max_cells - 1) + 1) ** 2))
        
        iH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')   
        jH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')   
        sH = bm.zeros(nfilter, dtype=bm.float64, device='cpu') 
        cc = 0

        if enable_timing:
            t.send('预处理')

        for i in range(nx):
            for j in range(ny):
                # 单元的编号顺序: y->x 
                row = i * ny + j
                # 根据物理距离计算搜索范围
                kk1 = int(max(i - (ceil(rmin/hx) - 1), 0))
                kk2 = int(min(i + ceil(rmin/hx), nx))
                ll1 = int(max(j - (ceil(rmin/hy) - 1), 0))
                ll2 = int(min(j + ceil(rmin/hy), ny))
                
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        # 单元的编号顺序: y->x 
                        col = k * ny + l
                        # 计算实际物理距离
                        physical_dist = sqrt((i - k)**2 * hx**2 + (j - l)**2 * hy**2)
                        fac = rmin - physical_dist
                        if fac > 0:
                            # ! bm.set_at 的性能相比直接赋值更慢
                            # iH[cc] = row
                            # jH[cc] = col
                            # sH[cc] = max(0.0, fac)
                            iH = bm.set_at(iH, cc, row)
                            jH = bm.set_at(jH, cc, col)
                            sH = bm.set_at(sH, cc, max(0.0, fac))
                            cc += 1

        if enable_timing:
            t.send('计算距离和过滤矩阵')

        H = COOTensor(
                indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
                values=sH[:cc],
                spshape=(nx * ny, nx * ny)
            )
        
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device='cpu')
        
        H = H.tocsr()
        H = H.device_put(self.device)
        Hs = bm.device_put(Hs, self.device)

        if enable_timing:
            t.send('矩阵构建')
            t.send(None)
        
        return H, Hs

    def _compute_filter_3d(self,
                        rmin: float, 
                        nx: int, ny: int, nz: int, 
                        hx: float, hy: float, hz: float,
                        enable_timing: bool = True,
                    ) -> Tuple[COOTensor, TensorLike]:
        """高性能优化的 3D 过滤矩阵计算 - 分块处理版本"""
        
        t = None
        if enable_timing:
            t = timer(f"Filter_3d")
            next(t)
        
        search_radius_x = ceil(rmin/hx)
        search_radius_y = ceil(rmin/hy)
        search_radius_z = ceil(rmin/hz)
        
        # 批处理单元, 避免一次处理所有单元耗尽内存
        batch_size = min(10000, nx * ny * nz)  
        n_batches = (nx * ny * nz + batch_size - 1) // batch_size
        
        # 创建一个映射函数，从线性索引转换为 3D 坐标
        def linear_to_3d(linear_idx):
            i = linear_idx // (ny * nz)
            j = (linear_idx % (ny * nz)) // nz
            k = linear_idx % nz
            return i, j, k
        
        # 预计算每个格子的物理坐标 
        all_coords = bm.zeros((nx * ny * nz, 3), dtype=bm.float64, device='cpu')
        
        for idx in range(nx * ny * nz):
            i, j, k = linear_to_3d(idx)
            all_coords[idx, 0] = i * hx
            all_coords[idx, 1] = j * hy
            all_coords[idx, 2] = k * hz
                
        if enable_timing:
            t.send('预处理')

        # 初始化存储结果的列表
        all_rows = []
        all_cols = []
        all_vals = []
        
        # 分批处理所有单元
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, nx * ny * nz)
            
            batch_rows = []
            batch_cols = []
            batch_vals = []
            
            # 获取当前批次单元的坐标
            batch_coords = all_coords[start_idx:end_idx]
            
            # 处理当前批次中的每个单元
            for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
                i, j, k = linear_to_3d(global_idx)
                row = global_idx
                
                # 计算搜索范围 - 与原始函数完全相同
                ii1 = max(0, i - (search_radius_x - 1))
                ii2 = min(nx, i + search_radius_x)
                jj1 = max(0, j - (search_radius_y - 1))
                jj2 = min(ny, j + search_radius_y)
                kk1 = max(0, k - (search_radius_z - 1))
                kk2 = min(nz, k + search_radius_z)
                
                # 创建搜索范围内所有单元的线性索引
                search_indices = []
                for ii in range(ii1, ii2):
                    for jj in range(jj1, jj2):
                        for kk in range(kk1, kk2):
                            col = kk + jj * nz + ii * ny * nz
                            search_indices.append(col)
                
                if not search_indices:
                    continue
                    
                # 获取搜索单元的物理坐标
                search_coords = all_coords[search_indices]
                
                # 计算与当前单元的距离
                current_coords = batch_coords[local_idx].reshape(1, 3) 
                diffs = search_coords - current_coords  
                squared_dists = bm.sum(diffs * diffs, axis=1) 
                distances = bm.sqrt(squared_dists) 
                
                # 计算滤波因子
                factors = rmin - distances 
                valid_mask = factors > 0  
                
                if bm.any(valid_mask):
                    valid_cols = bm.array(search_indices, device='cpu')[valid_mask]
                    valid_factors = factors[valid_mask]
                    
                    # 收集结果
                    batch_rows.extend([row] * len(valid_cols))
                    batch_cols.extend(valid_cols.tolist())
                    batch_vals.extend(valid_factors.tolist())
            
            # 添加批次结果到总结果
            all_rows.extend(batch_rows)
            all_cols.extend(batch_cols)
            all_vals.extend(batch_vals)
                
        if enable_timing:
            t.send('计算距离和过滤因子')

        if all_rows:
            iH = bm.tensor(all_rows, dtype=bm.int32, device='cpu')
            jH = bm.tensor(all_cols, dtype=bm.int32, device='cpu')
            sH = bm.tensor(all_vals, dtype=bm.float64, device='cpu')
        else:
            # 处理边缘情况
            iH = bm.tensor([], dtype=bm.int32, device='cpu')
            jH = bm.tensor([], dtype=bm.int32, device='cpu')
            sH = bm.tensor([], dtype=bm.float64, device='cpu')
        
        H = COOTensor(
            indices=bm.stack((iH, jH), axis=0),
            values=sH,
            spshape=(nx * ny * nz, nx * ny * nz)
        )
        
        H = H.tocsr()
        H = H.device_put(self.device)
        
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device=self.device)

        if enable_timing:
            t.send('矩阵构建')
            t.send(None)
        
        return H, Hs
    
    def _compute_filter_3d_math(self,
                                rmin: float, 
                                nx: int, ny: int, nz: int, 
                                hx: float, hy: float, hz: float,
                                enable_timing: bool = True,
                            ) -> Tuple[COOTensor, TensorLike]:
        """计算 3D 滤波矩阵 - 数学原理的直接实现版本"""

        t = None
        if enable_timing:
            t = timer(f"Filter_3d_math")
            next(t)

        min_h = min(hx, hy, hz)
        max_cells = ceil(rmin/min_h)
        nfilter = nx * ny * nz * ((2 * (max_cells - 1) + 1) ** 3)
        
        iH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')
        jH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')
        sH = bm.zeros(nfilter, dtype=bm.float64, device='cpu')
        cc = 0

        if enable_timing:
            t.send('预处理')

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 单元的编号顺序: z -> y -> x
                    row = k + j * nz + i * ny * nz
                    ii1 = int(max(i - (ceil(rmin/hx) - 1), 0))
                    ii2 = int(min(i + ceil(rmin/hx), nx))
                    jj1 = int(max(j - (ceil(rmin/hy) - 1), 0))
                    jj2 = int(min(j + ceil(rmin/hy), ny))
                    kk1 = int(max(k - (ceil(rmin/hz) - 1), 0))
                    kk2 = int(min(k + ceil(rmin/hz), nz))
                    
                    for ii in range(ii1, ii2):
                        for jj in range(jj1, jj2):
                            for kk in range(kk1, kk2):
                                # 单元的编号顺序: z -> y -> x
                                col = kk + jj * nz + ii * ny * nz
                                # 计算实际物理距离 
                                physical_dist = sqrt(
                                                    (i - ii)**2 * hx**2 + 
                                                    (j - jj)**2 * hy**2 + 
                                                    (k - kk)**2 * hz**2
                                                )
                                fac = rmin - physical_dist
                                if fac > 0:
                                    # ! bm.set_at 的性能相比直接赋值更慢
                                    # iH[cc] = row
                                    # jH[cc] = col
                                    # sH[cc] = max(0.0, fac)
                                    iH = bm.set_at(iH, cc, row)
                                    jH = bm.set_at(jH, cc, col)
                                    sH = bm.set_at(sH, cc, max(0.0, fac))
                                    cc += 1

        if enable_timing:
            t.send('计算距离和过滤因子')

        H = COOTensor(
            indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
            values=sH[:cc],
            spshape=(nx * ny * nz, nx * ny * nz)
        )
        H = H.tocsr()
        H = H.device_put(self.device)
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device=self.device)

        if enable_timing:
            t.send('矩阵构建')
            t.send(None)

        return H, Hs