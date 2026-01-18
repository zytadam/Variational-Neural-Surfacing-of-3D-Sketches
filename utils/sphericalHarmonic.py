import torch
import numpy as np

class SphericalHarmonic:

    def __init__(self, device):

        self.f_0 = torch.tensor([0,0,0,0,np.sqrt(7/12),0,0,0,np.sqrt(5/12)], dtype=torch.float32, device=device)

        self.L_x = torch.tensor([[0, 0, 0, 0, 0, 0, 0,-np.sqrt(2), 0],
                        [0, 0, 0, 0, 0, 0,-np.sqrt(7/2), 0,-np.sqrt(2)],
                        [0, 0, 0, 0, 0,-3/np.sqrt(2), 0,-np.sqrt(7/2), 0],
                        [0, 0, 0, 0,-np.sqrt(10), 0, -3/np.sqrt(2), 0, 0],
                        [0, 0, 0, np.sqrt(10), 0, 0, 0, 0, 0],
                        [0, 0, 3/np.sqrt(2), 0, 0, 0, 0, 0, 0],
                        [0, np.sqrt(7/2), 0, 3/np.sqrt(2), 0, 0, 0, 0, 0],
                        [np.sqrt(2), 0, np.sqrt(7/2), 0, 0, 0, 0, 0, 0],
                        [0, np.sqrt(2), 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)

        self.L_y = torch.tensor([[0, np.sqrt(2), 0, 0, 0, 0, 0, 0, 0],
                        [-np.sqrt(2), 0, np.sqrt(7/2), 0, 0, 0, 0, 0, 0],
                        [0,-np.sqrt(7/2), 0, 3/np.sqrt(2), 0, 0, 0, 0, 0],
                        [0, 0,-3/np.sqrt(2), 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0,-np.sqrt(10), 0, 0, 0],
                        [0, 0, 0, 0, np.sqrt(10), 0,-3/np.sqrt(2), 0, 0],
                        [0, 0, 0, 0, 0, 3/np.sqrt(2), 0,-np.sqrt(7/2), 0],
                        [0, 0, 0, 0, 0, 0, np.sqrt(7/2), 0,-np.sqrt(2)],
                        [0, 0, 0, 0, 0, 0, 0, np.sqrt(2), 0]], dtype=torch.float32, device=device)

        self.L_z = torch.tensor([[ 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [ 0, 0, 0, 0, 0, 0, 0, 3, 0],
                        [ 0, 0, 0, 0, 0, 0, 2, 0, 0],
                        [ 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0,-1, 0, 0, 0, 0, 0],
                        [ 0, 0,-2, 0, 0, 0, 0, 0, 0],
                        [ 0,-3, 0, 0, 0, 0, 0, 0, 0],
                        [-4, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)
        
        self.YZ = torch.linalg.matrix_exp(torch.pi/2 * self.L_x)
        
        
    def to(self, device):
        self.L_x.to(device)
        self.L_y.to(device)
        self.L_z.to(device)
        self.YZ.to(device)
        self.f_0.to(device)
        
    
    def v2f(self, v):
        """
        Convert a rotation vector to a 9D spherical harmonic vector.
        """

        v_x, v_y, v_z = v[..., 0], v[..., 1], v[..., 2]

        exp_vL = torch.matrix_exp(v_x.unsqueeze(-1).unsqueeze(-1) * -self.L_x +
                                v_y.unsqueeze(-1).unsqueeze(-1) * -self.L_y +
                                v_z.unsqueeze(-1).unsqueeze(-1) * self.L_z)

        # Compute the 9D SH vector for all matrices
        # f_0 should be broadcasted across the batch
        f = torch.matmul(exp_vL, self.f_0.unsqueeze(-1)).squeeze(-1)

        return f
    
    
    def R2v(self, R, eps=1e-12):
        """
        Convert a rotation matrix to axis angle.
        """
        if not (R.shape[-2:] == (3, 3)):
            raise ValueError("The input must be a valid batch of 3x3 rotation matrices.")
        
        det_R = torch.det(R)  # Shape: (b, n)
        negative_det_mask = det_R < 0  # Shape: (b, n)
        flip = torch.tensor([-1.0, 1.0, 1.0], device=R.device).expand_as(R)
        
        # Apply the mask to flip the first column where determinant is negative
        R = torch.where(negative_det_mask.unsqueeze(-1).unsqueeze(-1), R * flip, R)
        
        # Compute the rotation axis using the skew-symmetric components of R
        axis = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                            R[..., 0, 2] - R[..., 2, 0],
                            R[..., 1, 0] - R[..., 0, 1]], dim=-1) / 2
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)

        # If axis_norm is zero, apply a 90-degree rotation along the z-axis
        zero_mask = axis_norm < eps
        z_rotation_90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=R.device, dtype=R.dtype)
        R = torch.where(zero_mask.unsqueeze(-1), torch.matmul(R, z_rotation_90), R)

        # Recompute the rotation axis using the skew-symmetric components of the updated R
        axis = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                            R[..., 0, 2] - R[..., 2, 0],
                            R[..., 1, 0] - R[..., 0, 1]], dim=-1) / 2
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)

        axis = torch.nn.functional.normalize(axis, dim=-1)

        # Compute the cosine of the angle from the trace of R
        cos_theta = (torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2
        sin_theta = axis_norm.squeeze(-1)
        theta = torch.atan2(sin_theta, cos_theta)
        v = axis * theta.unsqueeze(-1)

        return v
    
    def R2f(self, R, eps=1e-12):
        v = self.R2v(R, eps=eps)
        f = self.v2f(v)
        return f
    
    def t2v(self, t):
        t = torch.nn.functional.normalize(t, dim=-1)

        z = torch.tensor([0.0, 0.0, 1.0], device=t.device).expand(t.shape[:-1] + (3,))
        rotation_axes = torch.cross(z, t, dim=-1)
        
        sin_angles = torch.norm(rotation_axes, dim=-1, keepdim=True)
        cos_angles = torch.sum(z * t, dim=-1, keepdim=True)
        angles = torch.atan2(sin_angles, cos_angles)

        # Return the axis-angle representation (axis * angle)
        v = rotation_axes * angles

        return v
    
    def alignment_loss(self, f, t):
        v = -self.t2v(t)
        v_x, v_y, v_z = v[..., 0], v[..., 1], v[..., 2]
        W = torch.matrix_exp(v_x.unsqueeze(-1).unsqueeze(-1) * -self.L_x +
                                v_y.unsqueeze(-1).unsqueeze(-1) * -self.L_y +
                                v_z.unsqueeze(-1).unsqueeze(-1) * self.L_z)
        
        Wf = torch.matmul(W, f.unsqueeze(-1)).squeeze(-1)
        u = (Wf - self.f_0)[...,1:8]
        loss = torch.linalg.norm(u, dim=-1)**2
        return loss
    
    def rotation_components(self, f, n):
        v = -self.t2v(n)
        v_x, v_y, v_z = v[..., 0], v[..., 1], v[..., 2]
        W = torch.matrix_exp(v_x.unsqueeze(-1).unsqueeze(-1) * self.L_x +
                                v_y.unsqueeze(-1).unsqueeze(-1) * self.L_y +
                                v_z.unsqueeze(-1).unsqueeze(-1) * self.L_z)
        Wf = torch.matmul(W, f.unsqueeze(-1)).squeeze(-1)
        u = (Wf - self.f_0)[...,[0,8]]
        return u
    
    def R2v_woexp(self, R, eps=1e-12):
        """
        Convert a rotation matrix to axis angle.
        """
        if not (R.shape[-2:] == (3, 3)):
            raise ValueError("The input must be a valid batch of 3x3 rotation matrices.")
        
        det_R = torch.det(R)  # Shape: (b, n)
        negative_det_mask = det_R < 0  # Shape: (b, n)
        flip = torch.tensor([-1.0, 1.0, 1.0], device=R.device).expand_as(R)
        
        # Apply the mask to flip the first column where determinant is negative
        R = torch.where(negative_det_mask.unsqueeze(-1).unsqueeze(-1), R * flip, R)
        
        # Compute the rotation axis using the skew-symmetric components of R
        axis = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                            R[..., 0, 2] - R[..., 2, 0],
                            R[..., 1, 0] - R[..., 0, 1]], dim=-1) / 2
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)

        # If axis_norm is zero, apply a 90-degree rotation along the z-axis
        zero_mask = axis_norm < eps
        x_rotation_90 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=R.device, dtype=R.dtype)
        R = torch.where(zero_mask.unsqueeze(-1), torch.matmul(R, x_rotation_90), R)

        # Recompute the rotation axis using the skew-symmetric components of the updated R
        axis = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                            R[..., 0, 2] - R[..., 2, 0],
                            R[..., 1, 0] - R[..., 0, 1]], dim=-1) / 2
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)
        axis = torch.nn.functional.normalize(axis, dim=-1)

        # Compute the cosine of the angle from the trace of R
        cos_theta = (torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2
        sin_theta = axis_norm.squeeze(-1)

        return axis, cos_theta, sin_theta
    
    def R2f_woexp(self, R, eps=1e-12):

        def cos2A(cosA, sinA):
            return cosA**2 - sinA**2
        def sin2A(cosA, sinA):
            return 2*cosA*sinA
        def cos3A(cosA):
            return 4*cosA**3 - 3*cosA
        def sin3A(sinA):
            return 3*sinA - 4*sinA**3
        def cos4A(sinA):
            return 8*sinA**4 - 8*sinA**2 + 1
        def sin4A(cosA, sinA):
            return 4*sinA*cosA - 8*sinA**3 * cosA
        
        def band_coeff(cosA, sinA):
            cosAs = torch.stack([cosA, cos2A(cosA, sinA), cos3A(cosA), cos4A(sinA)], dim=-1)
            sinAs = torch.stack([sinA, sin2A(cosA, sinA), sin3A(sinA), sin4A(cosA, sinA)], dim=-1)
            one = torch.tensor(1., dtype=R.dtype, device=R.device).expand(cosAs.shape[:-1] + (1,))
            zero = torch.tensor(0., dtype=R.dtype, device=R.device).expand(cosAs.shape[:-1] + (1,))
            
            coeff_cos = torch.cat([torch.flip(cosAs, dims=[-1]), one, cosAs], dim=-1)
            coeff_sin = torch.cat([-torch.flip(sinAs, dims=[-1]), zero, sinAs], dim=-1)
            return coeff_cos, coeff_sin

        # Axis-angle to spherical coordinates
        v, cos_r, sin_r = self.R2v_woexp(R, eps=eps)
        v_x, v_y, v_z = v[..., 0], v[..., 1], v[..., 2]
        v_x2 = v_x**2
        v_y2 = v_y**2

        cos_e = v_z
        sin_e = torch.sqrt(v_x2 + v_y2)
        sin_e = torch.where(sin_e < eps, sin_e + eps, sin_e)

        cos_a = v_x / sin_e
        sin_a = v_y / sin_e

        c_cos_r, c_sin_r = band_coeff(cos_r, sin_r)
        c_cos_e, c_sin_e = band_coeff(cos_e, sin_e)
        c_cos_a, c_sin_a = band_coeff(cos_a, sin_a)

        f_0 = self.f_0
        f_1 = torch.matmul(c_cos_a * f_0 + c_sin_a * torch.flip(f_0, dims=[-1]), self.YZ)
        f_2 = torch.matmul(c_cos_e * f_1 + c_sin_e * torch.flip(f_1, dims=[-1]), self.YZ.T)
        f_3 = torch.matmul(c_cos_r * f_2 - c_sin_r * torch.flip(f_2, dims=[-1]), self.YZ)
        f_4 = torch.matmul(c_cos_e * f_3 - c_sin_e * torch.flip(f_3, dims=[-1]), self.YZ.T)
        f = c_cos_a * f_4 - c_sin_a * torch.flip(f_4, dims=[-1])
        
        
        # body = torch.tensor([0.,0.,0.,(7/12)**.5,0.,0.,0.], dtype=R.dtype, device=R.device).expand(v.shape[:-1] + (7,))
        # tail = (v_z * cos4A(sin_r)* (5/12)**.5).unsqueeze(-1)
        # head = (v_z * sin4A(cos_r, sin_r) * (5/12)**.5).unsqueeze(-1)
        # f = torch.cat([head, body, tail], dim=-1)
        
        return f
        


if __name__ == "__main__":
    SH = SphericalHarmonic(device=torch.device('cpu'))
    # t = torch.tensor([[1.,1.,1.],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
    # f = SH.f_0.expand(t.shape[:-1] + (9,))
    # loss = SH.alignment_loss(f, t)
    # print(loss)

    R = torch.tensor([
        # [
        #     [0.866, -0.5, 0],
        #     [0.5, 0.866, 0],
        #     [0, 0, 1]
        # ],
        # [
        #     [1.0, 0.0, 0.0],
        #     [0.0, 0.0, -1.0],
        #     [0.0, 1.0, 0.0]
        # ],
        # [
        #     [1.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0],
        #     [0.0, 0.0, 1.0]
        # ],
        # [
        #     [0.7071,   -0.7071,         0],
        #     [0.7071,    0.7071,         0],
        #     [0,         0,         1.0000]
        # ],
        [
            [ 0.0188, -0.9998,  0.0000],
            [-0.9998, -0.0188,  0.0000],
            [ 0.0000, -0.0000,  1.0000]
        ],
        # [
        #     [-0.0188, -0.9998,  0.0000],
        #     [ 0.9998, -0.0188,  0.0000],
        #     [ 0.0000, -0.0000,  1.0000]
        # ],
    ], dtype=torch.float32, requires_grad=True)


    # f0 = SH.R2f_woexp(R)
    f1 = SH.R2f(R)
    f_bar = SH.f_0.expand_as(f1)

    # loss0 = torch.norm(f0 - f_bar, dim=-1).sum()
    loss1 = torch.norm(f1 - f_bar, dim=-1).sum()

    # loss0.backward()
    loss1.backward()
    print(f1)
    # print(f_bar)
    # print(loss1)
    print(R.grad)
