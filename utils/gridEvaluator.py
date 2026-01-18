import torch
import numpy as np
import scipy as sp
import pyvista as pv
from skimage import measure
from utils.utils_DPSR import grid_interp
import open3d as o3d
import os

class GridEvaluator:
    
    def __init__(self, grid_size, grid_res, device):
        self.grid_size = grid_size
        self.grid_res = grid_res
        self.device = device

        self.grid_values = None
        self.iso_value = None
        self.M_inter = None
        self.grid_points = None

        self.curve_network = None

        self.ras_p = None

    def set_grid_values(self, grid_values, iso_value):
        self.grid_values = grid_values
        self.iso_value = iso_value

    def get_grid_values(self):
        return self.grid_values, self.iso_value

    def generate_voxel_centers(self, jitter=True):
        n = self.grid_res
        s = self.grid_size
        h = s / n
        # Create a 1D grid from -s/2 to s/2 for each axis
        grid_1d = np.linspace(-s/2 + s/(2*n), s/2 - s/(2*n), n, dtype=np.float32)

        # Create 3D meshgrid for x, y, z coordinates
        z, y, x = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

        # Stack the coordinates to get a (n*n*n, 3) array where each row is (x, y, z)
        voxel_centers = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Generate a jittering offset
        if jitter:
            offset = np.random.uniform(0, h, size=3).astype(np.float32)
            self.offset = offset
            voxel_centers = voxel_centers + offset

        return voxel_centers
    
    def generate_unit_voxel_centers(self, jitter=True):
        n = self.grid_res
        s = 1
        h = s / n
        eps = 1e-6
        # Create a 1D grid from -s/2 to s/2 for each axis
        grid_1d = np.linspace(0+eps, 1-eps, n, dtype=np.float32)

        # Create 3D meshgrid for x, y, z coordinates
        z, y, x = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')

        # Stack the coordinates to get a (n*n*n, 3) array where each row is (x, y, z)
        voxel_centers = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Generate a jittering offset
        if jitter:
            offset = np.random.uniform(0, h, size=3).astype(np.float32)
            self.offset = offset
            voxel_centers = voxel_centers + offset

        return voxel_centers


    def trilinear_weights_tensor_sparse(self, sampled_points, jitter=True):
        device = self.device
        grid_size = self.grid_size
        grid_n = self.grid_res

        # Jittering
        if jitter:
            sampled_points = sampled_points - torch.as_tensor(self.offset, dtype=torch.float32, device=device)

        m = sampled_points.shape[0]
        h = grid_size / grid_n

        min_coord = -grid_size / 2 + h / 2
        frac_index = (sampled_points - min_coord) / h
        indices = torch.clamp(frac_index.floor().long(), 0, grid_n - 2)

        # Precompute octant offsets
        octant_offsets = torch.tensor(
            [[(i >> j) & 1 for j in range(3)] for i in range(8)],
            device=device,
            dtype=torch.float
        ).unsqueeze(0)  # Shape: (1, 8, 3)

        # Calculate weights for each octant
        delta = frac_index - indices.float()
        delta_expanded = delta.unsqueeze(1)  # Shape: (m, 1, 3)
        W = torch.prod(octant_offsets * delta_expanded + (1 - octant_offsets) * (1 - delta_expanded), dim=-1)

        # Calculate flattened indices for each corner of the cube
        indices_expanded = indices.unsqueeze(1) + octant_offsets  # Shape: (m, 8, 3)
        I = (indices_expanded[:, :, 2] * grid_n * grid_n +
            indices_expanded[:, :, 1] * grid_n +
            indices_expanded[:, :, 0]).long()

        # Construct the sparse weight matrix using torch sparse tensor
        batch_indices = torch.arange(m, device=device).unsqueeze(1).expand(-1, 8).flatten()
        I = I.flatten()
        W = W.flatten()

        M = torch.sparse_coo_tensor(
            torch.stack([batch_indices, I]),
            W,
            torch.Size([m, grid_n ** 3])
        )

        return M
    
    def trilinear_weights_array_sparse(self, sampled_points, jitter=True):
        grid_size = self.grid_size
        grid_n = self.grid_res

        # Jittering
        if jitter:
            sampled_points = sampled_points - self.offset

        m = sampled_points.shape[0]
        h = grid_size / grid_n

        min_coord = -grid_size / 2 + h / 2
        frac_index = (sampled_points - min_coord) / h
        indices = np.clip(np.floor(frac_index).astype(int), 0, grid_n - 2)

        # Precompute octant offsets
        octant_offsets = np.array(
            [[(i >> j) & 1 for j in range(3)] for i in range(8)],
            dtype=np.float32
        )[np.newaxis, :]  # Shape: (1, 8, 3)

        # Calculate weights for each octant
        delta = frac_index - indices.astype(np.float32)
        delta_expanded = delta[:, np.newaxis, :]  # Shape: (m, 1, 3)
        W = np.prod(octant_offsets * delta_expanded + (1 - octant_offsets) * (1 - delta_expanded), axis=-1)

        # Calculate flattened indices for each corner of the cube
        indices_expanded = indices[:, np.newaxis, :] + octant_offsets  # Shape: (m, 8, 3)
        I = (indices_expanded[:, :, 2] * grid_n * grid_n +
            indices_expanded[:, :, 1] * grid_n +
            indices_expanded[:, :, 0]).astype(int)

        # Construct the sparse weight matrix using scipy
        batch_indices = np.repeat(np.arange(m), 8)
        I = I.flatten()
        W = W.flatten()

        M = sp.sparse.csr_matrix((W, (batch_indices, I)), shape=(m, grid_n ** 3))

        return M

    def prepare_interpolation_matrix(self, sampled_points, jitter=True):
        if jitter:
            self.M_inter = self.trilinear_weights_tensor_sparse(sampled_points, jitter=True)
        elif self.M_inter is None:
            self.M_inter = self.trilinear_weights_tensor_sparse(sampled_points, jitter=False)
        
    def get_interpolation_matrix(self):
        return self.M_inter.detach()
    
    def evaluate_points(self, grid_values, sampled_points, jitter=True):
        self.prepare_interpolation_matrix(sampled_points, jitter)
        M = self.M_inter.detach()
        return M@grid_values
    
    def set_curve_points(self, curve_points):
        self.curve_points = curve_points
    
    def plot_grid_values(self, path, batch_id, show=False):
        if path is not None:
            os.makedirs(path, exist_ok=True)
            if batch_id == 0:
                self.save_curve(self.curve_points, path)
        
        if self.grid_values is None:
            return

        verts, faces, normals, values = measure.marching_cubes(self.grid_values, self.iso_value)
        verts = verts / self.grid_res

        if path is not None:
            os.makedirs(path, exist_ok=True)
            self.save_mesh(verts, faces, os.path.join(path, f"mesh_{batch_id}.ply"))
            if batch_id == 0:
                self.save_curve(self.curve_points, path)

        if show:
            faces = np.array([[3]+list(f) for f in faces])
            mesh = pv.PolyData(verts, faces)

            x, y, z = np.meshgrid(*[np.linspace(0, 1, self.grid_res)]*3, indexing='ij')
            grid_points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

            iso_samples = self.sample_iso_mesh()

            plot = pv.Plotter()
            plot.add_mesh(mesh, opacity=0.5)
            # plot.add_points(grid_points, scalars=self.grid_values.flatten(), point_size=0, cmap="coolwarm")
            plot.add_points(self.curve_points, color="red", point_size=5)
            # plot.add_points(iso_samples, color="green", point_size=5)
            plot.show()

    def save_mesh(self, verts, faces, filename="."):
        verts = self.curve_network.rescale_verts(verts)
        # filename = os.path.join(filename, "mesh.ply")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(filename, mesh)
        print(f"Mesh saved to {filename}")
    
    def save_curve(self, curve_points, filename="."):
        curve_points = self.curve_network.rescale_verts(curve_points)
        filename = os.path.join(filename, "curve.ply")
        curve = o3d.geometry.PointCloud()
        curve.points = o3d.utility.Vector3dVector(curve_points)
        o3d.io.write_point_cloud(filename, curve)
        print(f"Curve saved to {filename}")

    def sample_iso_surface(self, n=50000, eps=0.1):
        v = self.grid_values

        in_range = v - self.iso_value > 0
        in_range = in_range.astype(int)
        relevant_cell = (
            in_range[:-1, :-1, :-1] +
            in_range[1:, :-1, :-1] +
            in_range[:-1, 1:, :-1] +
            in_range[:-1, :-1, 1:] +
            in_range[1:, 1:, :-1] +
            in_range[1:, :-1, 1:] +
            in_range[:-1, 1:, 1:] +
            in_range[1:, 1:, 1:]
        )
        relevant_indices = np.argwhere((relevant_cell > -1) & (relevant_cell < 9))
        samples = []
        n_samples = 0
        while n_samples < n:
            sample_indices = np.random.choice(relevant_indices.shape[0], n)
            sample_indices = relevant_indices[sample_indices]
            sample_coords = np.random.rand(n, 3)
            sample_points = (sample_indices + sample_coords) / self.grid_res
            sample_values = self.grid_interp(sample_points).reshape(-1)
            mask = np.abs(sample_values) < eps
            sample_points = sample_points[mask]
            samples.append(sample_points)
            n_samples += sample_points.shape[0]
        return np.vstack(samples)
    
    def grid_interp(self, points):
        assert self.grid_values is not None, "Grid values not set"
        assert points.min() >= 0 and points.max() <= 1, "Points out of range"

        grid_values = torch.as_tensor(self.grid_values, device="cpu").unsqueeze(-1)
        query_points = torch.as_tensor(points, device="cpu")
        query_values = grid_interp(grid_values, query_points, batched=False)
        return query_values.numpy()

    def sample_iso_mesh(self, n=15000, sigma=0.0):
        verts, faces, normals, values = measure.marching_cubes(self.grid_values, self.iso_value)
        verts = verts / self.grid_res
        # faces = np.array([[3]+list(f) for f in faces])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        samples = mesh.sample_points_uniformly(n)
        samples = np.asarray(samples.points)
        noise = np.random.normal(0, 1/self.grid_res, size=samples.shape)
        samples = samples + sigma * noise
        return samples
    
    def save_grid_values(self, path):
        if self.grid_values is not None:
            np.save(path, self.grid_values)

    def load_grid_values(self, path):
        self.grid_values = np.load(path)

    def set_curve_network(self, curve_network):
        self.curve_network = curve_network

    def save_ras_p(self, path):
        if self.ras_p is not None:
            np.save(path, self.ras_p)
    

if __name__ == "__main__":
    device = "cpu"
    grid_size = 1
    grid_res = 32
    evaluator = GridEvaluator(grid_size, grid_res, device)

    grid_values = np.random.rand(grid_res, grid_res, grid_res)
    iso_value = 0.5
    evaluator.set_grid_values(grid_values, iso_value)

    evaluator.sample_iso_surface()

    query_points = np.random.rand(100, 3)
    evaluator.grid_interp(query_points)