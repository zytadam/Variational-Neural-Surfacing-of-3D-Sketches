import torch
import numpy as np
import scipy as sp
from sksparse.cholmod import cholesky
import time

class PoissonSolver:
    L = None
    P = None
    D = None
    
    def __init__(self, res=8, size=3, device="cpu"):
        self.res = res
        self.size = size
        self.spacing = size / res
        self.device = device

    def L_and_D(self):
        D = self.derivative_3d()
        return D.T @ D, D
    
    def derivative_3d(self):
        I = sp.sparse.identity(self.res, format='csr')

        off_diag = 0.5 / self.spacing * np.ones(self.res)

        diagonals = [-off_diag, off_diag]
        D = sp.sparse.diags(diagonals, [-1, 1], shape=(self.res, self.res), format='lil')

        # Apply periodic boundary conditions
        D[0, -1] = -0.5
        D[-1, 0] = 0.5

        Dx = sp.sparse.kron(sp.sparse.kron(I, I), D)
        Dy = sp.sparse.kron(sp.sparse.kron(I, D), I)
        Dz = sp.sparse.kron(sp.sparse.kron(D, I), I)

        derivative_3d = sp.sparse.vstack([Dx, Dy, Dz], format='csr')

        return derivative_3d
    
    def fft_poisson_solver(self, f):
        h = self.spacing
        n = self.res

        f = f.view(n, n, n)
        # Fourier transform of the source term
        f_hat = torch.fft.fftn(f, dim=(-3, -2, -1))
        
        # Generate wave numbers (kz, ky, kx) for each dimension, reflecting the (z, y, x) layout
        k = torch.fft.fftfreq(n, d=h, device=self.device) * 2 * torch.pi
        kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")
        
        # Correct wave number factor using the sine formulation
        k2 = (4 / h**2) * (torch.sin(kx * h / 2)**2 + torch.sin(ky * h / 2)**2 + torch.sin(kz * h / 2)**2)
        
        # Avoid division by zero at the zero frequency
        k2[0, 0, 0] = 1  # Temporary non-zero value to prevent division by zero
        phi_hat = f_hat / k2
        phi_hat[0, 0, 0] = 0  # Set the mean of phi to zero

        # Inverse FFT to get phi in the spatial domain
        phi = torch.fft.ifftn(phi_hat, dim=(-3, -2, -1)).real
        phi = phi.view(-1)

        return phi
    
    def fft_divergence(self, v):
        h = self.spacing
        n = self.res

        v = v.view(n, n, n, 3)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        vx_hat = torch.fft.fftn(vx, dim=(-3, -2, -1))
        vy_hat = torch.fft.fftn(vy, dim=(-3, -2, -1))
        vz_hat = torch.fft.fftn(vz, dim=(-3, -2, -1))

        k = torch.fft.fftfreq(n, d=h) * 2 * torch.pi
        kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")

        div_hat = 1j * (torch.sin(kx * h) * vx_hat + torch.sin(ky * h) * vy_hat + torch.sin(kz * h) * vz_hat) / h

        return div_hat
    
    def fft_div_poisson_solver(self, v):
        '''v in shape (..., n**3, 3)'''

        h = self.spacing
        n = self.res
        
        # Generate wave numbers (kz, ky, kx) for each dimension, reflecting the (z, y, x) layout
        k = torch.fft.fftfreq(n, d=h, device=self.device) * 2 * torch.pi
        kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")

        v = v.view(*v.shape[:-2], n, n, n, 3)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        vx_hat = torch.fft.fftn(vx, dim=(-3, -2, -1))
        vy_hat = torch.fft.fftn(vy, dim=(-3, -2, -1))
        vz_hat = torch.fft.fftn(vz, dim=(-3, -2, -1))

        f_hat = 1j * (torch.sin(kx * h) * vx_hat + torch.sin(ky * h) * vy_hat + torch.sin(kz * h) * vz_hat) / h

        # Correct wave number factor using the sine formulation
        k2 = (4 / h**2) * (torch.sin(kx * h / 2)**2 + torch.sin(ky * h / 2)**2 + torch.sin(kz * h / 2)**2)

        # Avoid division by zero at the zero frequency
        k2[0, 0, 0] = 1  # Temporary non-zero value to prevent division by zero
        phi_hat = f_hat / k2
        phi_hat[0, 0, 0] = 0  # Set the mean of phi to zero

        # Inverse FFT to get phi in the spatial domain
        phi = torch.fft.ifftn(phi_hat, dim=(-3, -2, -1)).real
        phi = phi.view(*phi.shape[:-3], n**3)

        return phi


    def cholesky_laplacian(self):
        h = self.spacing
        n = self.res

        D = PoissonSolver.D

        # Laplacian = - sp.sparse.linalg.LaplacianNd((n,n,n),boundary_conditions="neumann").tosparse().tocsc() / h**2
        Laplacian = D.T @ D
        Laplacian_cut = Laplacian[:-1,:-1]
        factor = cholesky(Laplacian_cut)
        L = factor.L()
        P = factor.P()
        L = torch.tensor(L.todense(), dtype=torch.float32, device=self.device)
        P = list(P)
        
        return L, P
    
    def cholesky_decompose(self, A):
        factor = cholesky(A)
        L = factor.L()
        P = factor.P()
        L = torch.tensor(L.todense(), dtype=torch.float32, device=self.device)
        P = list(P)
        return L, P
    
    
    def cholesky_solver(self, A, f):
        if PoissonSolver.D is None:
            PoissonSolver.D = self.generateD_sparse()
        if PoissonSolver.L is None or PoissonSolver.P is None:
            A = A[:-1,:-1]
            A = sp.sparse.csc_array(A.detach().cpu().numpy())
            PoissonSolver.L, PoissonSolver.P = self.cholesky_decompose(A)
        
        L = PoissonSolver.L
        P = PoissonSolver.P
        P_inv = list(np.argsort(P))

        x = torch.cholesky_solve(f[:-1][P].unsqueeze(-1), L, upper=False).squeeze(-1)
        x = torch.cat([x[P_inv], torch.tensor([0.0], dtype=torch.float32, device=self.device)], dim=-1)

        return x

    def cholesky_poisson_solver(self, n):
        if PoissonSolver.D is None:
            PoissonSolver.D = self.generateD_sparse()
        if PoissonSolver.L is None or PoissonSolver.P is None:
            PoissonSolver.L, PoissonSolver.P = self.cholesky_laplacian()

        L = PoissonSolver.L
        P = PoissonSolver.P
        P_inv = list(np.argsort(P))

        D = self.sparse_to_tensor(PoissonSolver.D).to(self.device)

        divN = (D.T @ n).squeeze(-1)
        x = torch.cholesky_solve(divN[:-1][P].unsqueeze(-1), L, upper=False).squeeze(-1)
        x = torch.cat([x[P_inv], torch.tensor([0.0], dtype=torch.float32, device=self.device)], dim=-1)

        return x
    
    def generateD_sparse(self):
        h = self.spacing
        n = self.res
        # D_phi  in   R^(V*3)
        # phi    in   R^V
        D_x = sp.sparse.lil_array((n**3, n**3), dtype=np.float32)
        D_y = sp.sparse.lil_array((n**3, n**3), dtype=np.float32)
        D_z = sp.sparse.lil_array((n**3, n**3), dtype=np.float32)
        # weight = 1 / (2 * h)
        weight = 1 / h

        for i in range(n**3):
            # x direction
            if i % n == 0:
                D_x[i, i] = -weight
                D_x[i, i+1] = weight
            elif i % n == n-1:
                D_x[i, i] = weight
                D_x[i, i-1] = -weight
            else:
                D_x[i, i+1] = weight/2
                D_x[i, i-1] = -weight/2

            # y direction
            if (i // n) % n == 0:
                D_y[i, i] = -weight
                D_y[i, i+n] = weight
            elif (i // n) % n == n-1:
                D_y[i, i] = weight
                D_y[i, i-n] = -weight
            else:
                D_y[i, i+n] = weight/2
                D_y[i, i-n] = -weight/2

            # z direction
            if i < n*n:
                D_z[i, i] = -weight
                D_z[i, i+n*n] = weight
            elif i >= n*n*(n-1) and i < n*n*n:
                D_z[i, i] = weight
                D_z[i, i-n*n] = -weight
            else:
                D_z[i, i+n*n] = weight/2
                D_z[i, i-n*n] = -weight/2

        return sp.sparse.vstack([D_x, D_y, D_z]).tocsr()
    
    def sparse_to_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)



if __name__ == "__main__":
    res = 32
    size = 4
    # np.random.seed(seed=1)

    ps = PoissonSolver(res, size)
    device = "cuda:0"
    
    n = ps.res
    h = ps.spacing

    N = torch.rand((3*n**3,1), dtype=torch.float32)
    D = ps.generateD_sparse()
    L = (D.T @ D)[:-1,:-1]
    L = torch.tensor(L.todense(),dtype=torch.float32)
    divN = ps.sparse_to_tensor(D).T @ N

    print(type(L), L.shape)
    print(divN.shape)
    LU, pivots = torch.linalg.lu_factor(L)
    print(LU.shape)
    v = torch.linalg.lu_solve(LU, pivots, divN)
