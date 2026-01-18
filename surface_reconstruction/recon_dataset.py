import torch.utils.data as data
import numpy as np
import scipy as sp
import scipy.spatial as spatial
import open3d as o3d
import trimesh
import torch
import utils.utils as utils
import utils.gridEvaluator as grid_eval
import utils.poissonSolver as PS
import utils.curveNetwork as CN


class SuperDataset(data.Dataset):
    # A class to generate synthetic examples of basic shapes.
    # Generates clean and noisy point clouds sampled  + samples on a grid with their distance to the surface (not used in DiGS paper)
    def __init__(self, file_path, n_points, n_samples=128, res=128, sample_type='grid', sapmling_std=0.005,
                 requires_dist=False, requires_curvatures=False, grid_range=2, evaluator=None):
        self.file_path = file_path
        self.n_points = n_points
        self.n_samples = n_samples
        # self.grid_res = res
        # self.sample_type = sample_type  # grid | gaussian | combined
        # self.sampling_std = sapmling_std
        # self.requires_dist = requires_dist
        # self.nonmnfld_dist, self.nonmnfld_n, self.mnfld_curvs = None, None, None
        # self.requires_curvatures = requires_curvatures  # assumes a subdirectory names "estimated props" in dataset path
        # load data
        self.o3d_point_cloud = o3d.io.read_point_cloud(self.file_path)
        self.curve_network = None
        self.grid_range = grid_range

        # extract center and scale points and normals
        # self.points, self.mnfld_n = self.get_mnfld_points()
        # self.bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).transpose()
        # self.bbox_trimesh = trimesh.PointCloud(self.points).bounding_box.copy()

        # self.point_idxs = np.arange(self.points.shape[0], dtype=np.int32)
        # record sigma
        # self.sample_gaussian_noise_around_shape()

        self.grid_n = res
        self.grid_values = None
        self.evaluator = evaluator

        self.iteration = 0
    
    def set_iteration(self, iteration):
        self.iteration = iteration  # Update iteration

    def get_mnfld_points(self):
        # Returns points on the manifold
        points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32)
        if normals.shape[0] == 0:
            normals = np.zeros_like(points)
        # center and scale data/point cloud
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        self.scale = np.abs(points).max()
        points = points / self.scale / (4. * self.grid_range)
        points = points * 0.8 + 0.5
        return points, normals

    def sample_gaussian_noise_around_shape(self):
        kd_tree = spatial.KDTree(self.points)
        # query each point for sigma
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas
        return

    def __getitem__(self, index):
        nonmnfld_points = np.random.uniform(0., 1.,
                                            size=(self.n_points, 3)).astype(np.float32)  # (n_points, 3)
        
        curve_points, curve_noise_points, curve_tangents = self.curve_network.get_uniform_samples(10000, sigma=0.002)
        self.evaluator.set_curve_points(curve_points)

        if self.evaluator.grid_values is None or self.iteration <= 100:
            iso_points = np.random.uniform(0., 1.,
                                            size=(0, 3)).astype(np.float32)
            knn_idx = np.zeros((0, 8), dtype=np.int32)
            
        else:
            iso_points = self.evaluator.sample_iso_mesh(15000, sigma=0.005).astype(np.float32)

            # Distance mask
            C = curve_points.astype(np.float64)
            pcd_C = o3d.geometry.PointCloud()
            pcd_C.points = o3d.utility.Vector3dVector(C)
            C_sub = curve_noise_points.astype(np.float64)
            pcd_C_sub = o3d.geometry.PointCloud()
            pcd_C_sub.points = o3d.utility.Vector3dVector(C_sub)
            
            S = iso_points.astype(np.float64)
            pcd_S = o3d.geometry.PointCloud()
            pcd_S.points = o3d.utility.Vector3dVector(S)
            D = np.asarray(pcd_S.compute_point_cloud_distance(pcd_C))
            Dmax = D.max()
            asc_indices = np.argsort(D)

            r = ((self.iteration - 100) / 700)**1
            n = int(15000 * r)
            # iso_points = iso_points[(D < r * Dmax) & (D > 0.001)]
            # iso_points = iso_points[(D < r * Dmax)]
            # iso_points = iso_points[asc_indices[:n]]
            # print(iso_points.shape)

            knn_idx = []
            S_tree = spatial.cKDTree(np.concatenate([iso_points], axis=0))
            for p in np.array_split(curve_noise_points, 100, axis=0):
                _, index = S_tree.query(p, k=8)
                knn_idx.append(index)
            knn_idx = np.concatenate(knn_idx, axis=0)

        return {'nonmnfld_points': nonmnfld_points, 'iso_points':iso_points, 'curve_points': curve_points, 'sub_curve_points': curve_noise_points, 'curve_tangents': curve_tangents, 'knn_index': knn_idx}

    def get_train_data(self, batch_size):
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:batch_size]
        manifold_points = self.points[mnfld_idx]  # (n_points, 3)
        near_points = (manifold_points + self.sigmas[mnfld_idx] * np.random.randn(manifold_points.shape[0],
                                                                                  manifold_points.shape[1])).astype(
            np.float32)

        return manifold_points, near_points, self.points

    def gen_new_data(self, dense_pts):
        self.points = dense_pts
        kd_tree = spatial.KDTree(self.points)
        # query each point for sigma^2
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas

    def __len__(self):
        return self.n_samples
    
    def set_model(self, model):
        self.model = model 
    
    def set_grid_values(self, values):
        self.grid_values = values[0]
        self.iso_value = values[1]

    def set_grid_evaluator(self, evaluator):
        self.evaluator = evaluator

    def set_curve_network(self, curve_network):
        self.curve_network = curve_network

class SuperDataLoader(data.DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.iteration = 0  # Track iteration count

    def __iter__(self):
        dataset = self.dataset  # Reference to dataset
        for batch in super().__iter__():  # Get batch from original DataLoader
            if hasattr(dataset, "set_iteration"):  # Ensure dataset supports iteration tracking
                dataset.set_iteration(self.iteration)
            self.iteration += 1
            yield batch