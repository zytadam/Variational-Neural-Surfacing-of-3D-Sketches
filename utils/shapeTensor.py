import torch

class ShapeTensor:
    def __init__(self, dtype=torch.float32, device="cpu"):
        self.device = device
        self.dtype = dtype

    def shape_tensor_3D_to_2D(self, normals, shape_tensors):
        """
        Convert 3D shape tensor to 2D shape tensor
        Args:
            normals: (..., 3) tensor
            shape_tensors: (..., 3, 3) tensor
        Returns:
            shape_tensor_2d: (..., 2, 2) tensor
        """
        n = torch.nn.functional.normalize(normals, p=2, dim=-1)
        u = self.perpendicular_vector(n)
        v = torch.cross(n, u, dim=-1)
        J = torch.stack([u, v], dim=-1)
        # print(J.shape)
        s2 = torch.matmul(torch.matmul(J.transpose(-1, -2), shape_tensors), J)
        return s2, J
    
    def decompose_shape_tensor(self, normals, hessians):
        """
        Compute the eigendecomposition of shape tensor from normals and hessians
        Args:
            normals: (..., 3) tensor
            hessians: (..., 3, 3) tensor
        Returns:
            shape_tensor: (..., 3, 3) tensor
        """
        n = torch.nn.functional.normalize(normals, p=2, dim=-1)
        u = self.perpendicular_vector(n)
        v = torch.cross(n, u, dim=-1)
        J = torch.stack([u, v], dim=-1)
        # print(J.shape)
        h2 = torch.matmul(torch.matmul(J.transpose(-1, -2), hessians), J)
        # print(h2.shape)
        val2d, vec2d = self.decompose2D(h2)
        vec3d = torch.matmul(J, vec2d)

        vals = torch.stack([torch.zeros_like(val2d[..., 0], dtype=self.dtype, device=self.device), val2d[..., 0], val2d[..., 1]], dim=-1)
        vecs = torch.stack([n, vec3d[..., 0], vec3d[..., 1]], dim=-1)

        return vals, vecs
    
    def perpendicular_vector(self, n):
        """
        Compute perpendicular vector to the input vectors.

        Args:
            n (torch.Tensor): Input tensor of shape (..., 3).

        Returns:
            torch.Tensor: Perpendicular vectors of shape (..., 3).
        """
        # Choose a reference vector that is unlikely to be parallel to v
        ref_vector = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype, requires_grad=False).expand_as(n).clone()
        # print(ref_vector.shape)
        
        # Check if n is aligned with ref_vector, switch to [0, 1, 0] if necessary
        aligned = torch.abs(n[..., 0]) > torch.abs(n[..., 1])
        ref_vector[aligned] = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype, requires_grad=False)
        # print(ref_vector.shape)
        # Compute the cross product
        perp = torch.cross(n, ref_vector, dim=-1)
        # print(perp.shape)
        
        return perp
    
    def decompose2D(self, A):
        # input (..., 2, 2)

        trA = A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        detA = torch.linalg.det(A)
        gapA = torch.sqrt(torch.clamp(trA ** 2 - 4 * detA, min=1e-6))

        lambda1 = (trA + gapA) / 2
        lambda2 = (trA - gapA) / 2

        I = torch.eye(2, device=self.device, requires_grad=False)
        V1 = A - lambda2[..., None, None] * I
        # V2 = A - lambda1[..., None, None] * I
        
        i1 = torch.argmax(torch.linalg.norm(V1, ord=2, dim=-2), dim=-1).unsqueeze(-1).unsqueeze(-1).expand(*V1.shape[:-1], 1)
        # i2 = torch.argmax(torch.linalg.norm(V2, ord=2, dim=-2), dim=-1).unsqueeze(-1).unsqueeze(-1).expand(*V2.shape[:-1], 1)
        v1 = torch.gather(V1, dim=-1, index=i1).squeeze(-1)
        # v2 = torch.gather(V2, dim=-1, index=i2)

        v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)
        v2 = torch.stack([-v1[..., 1], v1[..., 0]], dim=-1)

        return torch.stack([lambda1, lambda2], dim=-1), torch.stack([v1, v2], dim=-1)

    def alignment_loss(self, normals, shape_tensors, tangents):
        """
        Compute alignment loss between shape tensors and tangents
        Args:
            shape_tensors: (..., 3, 3) tensor
            tangents: (..., 3) tensor
        Returns:
            loss: (..., ) tensor
        """
        _, vecs = self.decompose_shape_tensor(normals, shape_tensors)
        J = vecs[..., :, 1:]
        t_2d = torch.matmul(J.transpose(-1, -2), tangents.unsqueeze(-1)).squeeze(-1)
        t_2d = torch.nn.functional.normalize(t_2d, p=2, dim=-1)
        theta = torch.atan2(t_2d[..., 1], t_2d[..., 0])
        loss = torch.abs(torch.cos(4*theta) - 1)

        return loss

    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = ShapeTensor(device=device)
    
    # normal = torch.tensor([[-0.14420263, -0.13337484, 0.5616307],
    #                        [-0.08954914, 0.1160154, -0.7712222]], dtype=torch.float32, device=device)
    # hessian = torch.tensor([[[0.833367,  0.6076003, 1.4381562],
    #                         [0.6076003, 0.8479664, 1.2376046],
    #                         [1.4381562, 1.2376046, 1.2389835]],
    #                         [[ 0.5128148,  -0.23061274, -2.157321  ],
    #                         [-0.23061274,  0.5943284,   2.7933636 ],
    #                         [-2.157321,    2.7933636,   1.5119982 ]]], dtype=torch.float32, device=device)
    
    # vals, vecs = st.decompose_shape_tensor(normal, hessian)
    # print(vals)
    # print(vecs)

    normal = torch.tensor([[-0.0167, -0.8361,  0.5248]], dtype=torch.float32, device=device)
    hessian = torch.tensor([[[ 1.1106, -0.0159,  0.0101],
        [-0.0159,  0.3143,  0.5001],
        [ 0.0101,  0.5001,  0.7971]]], dtype=torch.float32, device=device)
    
    H_2d = st.shape_tensor_3D_to_2D(normal, hessian)
    print(H_2d)
    
    vals, vecs = st.decompose2D(H_2d)
    print(vals)
    print(vecs)

    vals = vals[...,[1,0]]
    vecs = vecs[...,:,[1,0]]
    vecs[...,:,1] = -vecs[...,:,1]
    print(vals)
    print(vecs)

    print(torch.linalg.matmul(vecs, torch.linalg.matmul(torch.diag_embed(vals), vecs.transpose(-1, -2))))