%% 参数定义
nelx = 80;
nely = 64;
Vmax = 0.5;
tau = 2e-4;

E0 = 1;
Emin = 1e-4;
nu = 0.3;
nvol = 100;
dt = 0.1;
d = -0.02;
p = 4;
phi = ones((nely+1)*(nelx+1), 1);
str = ones(nely, nelx);
volInit = sum(str(:)) / (nelx * nely);

%% 有限元分析准备
% 对于位移场
A11 = [12 3 -6 3;  3 12  3  0; -6  3 12 -3; 3 0 -3 12];
A12 = [-6 -3 0 3; -3 -6 -3 -6;  0 -3 -6  3; 3 -6 3 -6];
B11 = [4  3 -2 9;  3 -4 -9  4; -2 -9 -4 -3; 9 4 -3 -4];
B12 = [2 -3 4 -0; -3  2  9 -2;  4  9  2  3; -9 -2 3 2];
KE = 1 / (1-nu^2) / 24 * [A11 A12; A12' A11] + nu * [B11 B12; B12' B11];

% 对于拓扑导数
a1 = 3*(1-nu) / (2*(1+nu)*(7-5*nu)) * (-(1-14*nu+15*nu^2)*E0) / (1-2*nu)^2;
a2 = 3*(1-nu) / (2*(1+nu)*(7-5*nu)) * 5*E0;
A = (a1+2*a2)/24 * ([A11 A12; A12' A11] + (a1/(a1+2*a2))*[B11 B12; B12' B11]);
nodenrs = reshape(1:(1+nelx)*(1+nely), 1+nely, 1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1) + 1, nelx*nely, 1);
edofMat = repmat(edofVec, 1, 8) + repmat([0 1 2*nely+[2 3 0 1] -2 -1], nelx*nely, 1);
iK = reshape(kron(edofMat, ones(8, 1))', 64*nelx*nely, 1);
jK = reshape(kron(edofMat, ones(1, 8))', 64*nelx*nely, 1);

% 对于反应扩散方程



