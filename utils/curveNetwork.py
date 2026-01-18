import numpy as np
import os
import pyvista as pv
class CurveNetwork():
    def __init__(self, file_path):
        self.verts = None
        self.lines = None
        self.lens = None
        self.tans = None
        if os.path.splitext(file_path)[1] == '.obj':
            self.load_from_obj(file_path)
        elif os.path.splitext(file_path)[1] == '.ply':
            self.load_from_ply(file_path)
        self.scale, self.trans = self.fit_into_box(1, 0.7)

    def load_from_ply(self, file_path):
        verts = []
        lines = []
        lens = []

        with open(file_path, 'r') as f:
            while True:
                line = f.readline().strip()
                if line.startswith('element vertex'):
                    n_verts = int(line.split()[-1])
                elif line.startswith('element face'):
                    n_faces = int(line.split()[-1])
                elif line.startswith('end_header'):
                    break
            file_lines = f.readlines()
            for i in range(n_verts):
                verts.append(list(map(float, file_lines[i].split())))

            for i in range(n_verts, n_verts + n_faces):
                l = file_lines[i].split()
                idxs = list(map(int, l[1:3]))
                len = float(l[3])
                lines.append(idxs)
                lens.append(len)

        self.verts = np.array(verts)
        self.lines = np.array(lines)
        self.lens = np.array(lens)
        self.tans = self.verts[self.lines[:, 1]] - self.verts[self.lines[:, 0]]
        self.tans /= self.lens[:, None]

        print(f'Loaded {self.verts.shape[0]} vertices and {self.lines.shape[0]} lines from {file_path}')

    def load_from_obj(self, file_path):
        verts = []
        lines = []

        with open(file_path, 'r') as f:
            line = f.readline().strip()
            while line:
                if line.startswith('v '):
                    verts.append(list(map(float, line.split()[1:])))
                elif line.startswith('l '):
                    l = line.split()
                    idxs = list(map(int, l[1:3]))
                    lines.append(idxs)
                line = f.readline().strip()
        
        self.verts = np.array(verts)
        self.lines = np.array(lines) - 1
        self.lens = np.linalg.norm(self.verts[self.lines[:, 1]] - self.verts[self.lines[:, 0]], axis=1)
        self.tans = self.verts[self.lines[:, 1]] - self.verts[self.lines[:, 0]]
        self.tans /= self.lens[:, None]

        print(f'Loaded {self.verts.shape[0]} vertices and {self.lines.shape[0]} lines from {file_path}')

    
    def fit_into_box(self, box_size=1, ratio=0.8):
        center = self.verts.mean(axis=0)
        
        scale = np.abs(self.verts).max()
        scale = 0.5 * 1/scale * box_size * ratio
        trans = box_size / 2 - center * scale

        self.verts = self.verts
        self.verts = self.verts * scale + trans
        self.lens = self.lens * scale

        return scale, trans
    
    def get_uniform_samples(self, n_samples, sigma=0.0):
        weights = self.lens / self.lens.sum()
        idxs = np.random.choice(np.arange(len(self.lens)), size=n_samples, p=weights)
        locs = np.random.rand(n_samples)
        sample_points = self.verts[self.lines[idxs, 0]]*(1-locs[:, None]) + self.verts[self.lines[idxs, 1]]*locs[:, None]
        sample_tangents = self.tans[idxs]

        sample_points = sample_points
        u = np.zeros_like(sample_tangents)
        mask = np.abs(sample_tangents[:, 0]) > np.abs(sample_tangents[:, 1])
        u[mask] = np.stack([-sample_tangents[mask, 2], np.zeros(mask.sum()), sample_tangents[mask, 0]], axis=1)
        u[~mask] = np.stack([np.zeros((~mask).sum()), sample_tangents[~mask, 2], -sample_tangents[~mask, 1]], axis=1)
        u = u / np.linalg.norm(u, axis=1)[:, None]
        v = np.cross(sample_tangents, u)
        a, b = np.random.normal(0, sigma, size=(2, n_samples))
        sample_noise_points = sample_points + a[:, None]*u + b[:, None]*v    
        return sample_points.astype(np.float32), sample_noise_points.astype(np.float32), sample_tangents.astype(np.float32)
    
    def rescale_verts(self, verts):
        return (verts - self.trans) / self.scale

    
if __name__ == "__main__":
    
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(root, 'data/sdf/input')
    
    files = list()
    for f in sorted(os.listdir(input_path)):
        if os.path.splitext(f)[1] in ['.obj']:
            files.append(f)
    file = files[0]
    file_path = os.path.join(input_path, file)

    cnet = CurveNetwork(file_path)
    points, tangents = cnet.get_uniform_samples(50000)
    print(points.shape, tangents.shape)

    plot = pv.Plotter()
    plot.add_points(points, color='r')
    # plot.add_arrows(points, tangents, mag=0.1, color='b')
    plot.show()