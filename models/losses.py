import torch
import torch.nn as nn
import utils.utils as utils
import scipy as sp
import numpy as np
import open3d as o3d
import utils.sphericalHarmonic as SH
import utils.poissonSolver as PS
import utils.shapeTensor as ST
from models import DPSR
from utils.utils_DPSR import grid_interp

def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs', min=1.0):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - min).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - min).square()).mean()

    return eikonal_term


def relax_eikonal_loss(nonmnfld_grad, mnfld_grad, min=.8, max=0.1, eikonal_type='abs', udf=False):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    grad_norm = all_grads.norm(2, dim=-1) + 1e-12
    if udf:
        pass
    else:
        term = torch.relu(-(grad_norm - min))
    if eikonal_type == 'abs':
        eikonal_term = term.abs().mean()
    else:
        eikonal_term = term.square().mean()
    return eikonal_term

def gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad):
    device = morse_nonmnfld_grad.device
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, morse_nonmnfld_grad[:, :, :, None]), dim=-1)
    zero_grad = torch.zeros(
        (morse_nonmnfld_grad.shape[0], morse_nonmnfld_grad.shape[1], 1, 1),
        device=device)
    zero_grad = torch.cat((morse_nonmnfld_grad[:, :, None, :], zero_grad), dim=-1)
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)
    K_G = (-1. / (morse_nonmnfld_grad.norm(dim=-1) ** 2 + 1e-12)) * torch.det(
        nonmnfld_hessian_term)

    return K_G

def mean_curvature(hessian, grad):
    device = grad.device
    GHG_T = torch.einsum('bni,bnij,bnj->bn', grad, hessian, grad)
    Trace_H = torch.einsum('bnii->bn', hessian)
    grad_norm = grad.norm(dim=-1)
    K_M = (GHG_T - grad_norm ** 2 * Trace_H)/(2*grad_norm**2 + 1e-12)
    return K_M


def latent_rg_loss(latent_reg, device):
    # compute the VAE latent representation regularization loss
    if latent_reg is not None:
        reg_loss = latent_reg.mean()
    else:
        reg_loss = torch.tensor([0.0], device=device)

    return reg_loss


class FlowLoss(nn.Module):
    def __init__(self, weights=None, loss_type='siren_supervised', div_decay='none',
                 div_type='l1', bidirectional_morse=True, udf=False):
        super().__init__()
        if weights is None:
            weights = [3e3, 1e2, 1e2, 5e1, 1e2, 1e1]
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_morse = True if 'morse' in self.loss_type else False
        self.bidirectional_morse = bidirectional_morse
        self.udf = udf

    def forward(self, output_pred, mnfld_points, nonmnfld_points, curve_points=None, sub_curve_points=None, curve_tangents=None, knn_idx=None, grid_evaluator=None, iter=None):
        dims = nonmnfld_points.shape[-1]
        device = nonmnfld_points.device

        #########################################
        # Compute required terms
        #########################################

        nonmanifold_pred = output_pred["nonmanifold_pnts_pred"]
        near_pred = output_pred['curve_points_pred']
        sub_curve_pred = output_pred['sub_curve_points_pred']
        iso_pred = output_pred['manifold_pnts_pred']
        latent_reg = output_pred["latent_reg"]

        curv_term = torch.tensor([0.0], device=device)
        latent_reg_term = torch.tensor([0.0], device=device)

        if grid_evaluator is not None:
            evaluator = grid_evaluator

        near_points = curve_points
        near_tangents = curve_tangents
        iso_points = mnfld_points

        if iso_points is None or iter <= 200:
            iso_points = None
            ps_points = torch.cat([near_points], dim=1)
            ps_normals = torch.cat([near_pred], dim=1)
        else:
            ps_points = torch.cat([near_points, iso_points], dim=1)
            ps_normals = torch.cat([near_pred, iso_pred], dim=1)

        inter_term = eikonal_term = smooth_term = tangent_term = iso_term = cc_term = torch.tensor([0.0], device=device)



        # Poisson data loss
        n = evaluator.grid_res
        s = evaluator.grid_size
        h = s/n

        eval_points = torch.cat([near_points, sub_curve_points], dim=1)

        dpsr = DPSR.DPSR(res=(n, n, n), sig=3., shift=True, scale=True).to(device)
        if iter <= 600 and iter > 100:
            v_grid, v_eval = dpsr(torch.clamp(ps_points, 0., 1.), ps_normals, eval_points, avg=False, normalize=False)
        else:
            v_grid, v_eval = dpsr(torch.clamp(ps_points, 0., 1.), ps_normals, eval_points, avg=False, normalize=True)

        v_grid = v_grid.squeeze(0)
        v_eval = v_eval.squeeze(0)
        v_curve = v_eval[:near_points.shape[1]]
        v_sub_curve = v_eval[near_points.shape[1]:]
        sdf_term = (v_curve**2).mean()

        evaluator.set_grid_values(v_grid.detach().cpu().numpy(), 0.0)
        evaluator.ras_p = dpsr.ras_p

        if iso_points is not None:
            # print("KNN index: ", knn_idx.shape)
            # print("Iso points: ", iso_points.shape)

            knn_points = utils.knn_gather(torch.cat([iso_points], dim=1), knn_idx)
            # print("KNN points: ", knn_points.shape)

            v = sub_curve_points[:, :, None, :] - knn_points
            v1 = v[:, :, :8, :].mean(-2)
            v2 = v[:, :, :4, :].mean(-2)
            v3 = v[:, :, 0, :]
            
            v_sub_curve = v_sub_curve.unsqueeze(0).unsqueeze(-1)
            loss_v1 = torch.linalg.norm((v1 - v_sub_curve * sub_curve_pred), ord=2, dim=-1).mean()
            loss_v2 = torch.linalg.norm((v2 - v_sub_curve * sub_curve_pred), ord=2, dim=-1).mean()
            loss_v3 = torch.linalg.norm((v3 - v_sub_curve * sub_curve_pred), ord=2, dim=-1).mean()
            cc_term = loss_v1 + loss_v2 + loss_v3


        if iso_points is not None:
            # print("Iso preds NaN: ", torch.isnan(iso_pred).any().item())
            # print("Iso preds finite: ", torch.isfinite(iso_pred).all().item())

            iso_dx = utils.gradient(iso_points, iso_pred[:, :, 0])
            iso_dy = utils.gradient(iso_points, iso_pred[:, :, 1])
            iso_dz = utils.gradient(iso_points, iso_pred[:, :, 2])
            iso_hessian = torch.stack((iso_dx, iso_dy, iso_dz), dim=-1)
            iso_hessian_hat = 0.5 * (iso_hessian + torch.transpose(iso_hessian, -2, -1))
            n_hat = torch.nn.functional.normalize(iso_pred, dim=-1)
            norm = torch.linalg.norm(iso_pred, dim=-1, keepdim=True).unsqueeze(-1) + 1e-8
            P = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0) - n_hat.unsqueeze(-1) * n_hat.unsqueeze(-2)
            S = P @ iso_hessian_hat @ P
            S = S / norm

            # print("Shape tensor NaN: ", torch.isnan(S).any().item())
            # print("Shape tensor finite: ", torch.isfinite(S).all().item())

            # Projected dS
            # dS = torch.transpose(dS, -2, -1)
            ntn = torch.einsum('...i,...j->...ij', iso_pred, iso_pred)
            M_proj = torch.eye(3, device=device).expand_as(ntn) - ntn
            # dS_proj = torch.linalg.matmul(dS, M_proj)
            # dS_proj_loss = torch.norm(dS_proj, p=1, dim=(-2, -1))


            # Curvature loss
            # K_G = gaussian_curvature(iso_hessian, iso_pred)
            # KG_loss = K_G**2
            # KG_loss = torch.abs(K_G)

            # K_M = mean_curvature(iso_hessian, iso_pred)
            # KM_loss = K_M**2

            st = ST.ShapeTensor(device=device)
            S_2d, J = st.shape_tensor_3D_to_2D(iso_pred, S)
            # iso_vals, iso_vecs = st.decompose_shape_tensor(iso_pred, S)
            t1 = J[..., :, 0]
            t2 = J[..., :, 1]
            
            ds0 = utils.gradient(iso_points, S_2d[:, :, 0, 0])
            ds1 = utils.gradient(iso_points, S_2d[:, :, 0, 1])
            ds2 = utils.gradient(iso_points, S_2d[:, :, 1, 0])
            ds3 = utils.gradient(iso_points, S_2d[:, :, 1, 1])
            ds = torch.stack((ds0, ds1, ds2, ds3), dim=-1)

            # projected ds 2d
            ds = torch.transpose(ds, -2, -1)
            ds_proj = torch.linalg.matmul(ds, P)

            # dn_proj = torch.linalg.matmul(S, M_proj)

            # test_loss = torch.linalg.matrix_norm(ds)
            # test_loss = torch.abs(ds).sum(dim=(-2, -1))
            # test_loss = torch.abs(ds_proj).sum(dim=(-2, -1))
            ds_proj_loss = torch.linalg.matrix_norm(ds_proj)
            ds_loss = torch.linalg.matrix_norm(ds)

            # test_loss = torch.linalg.matrix_norm(iso_hessian_hat)
            S_loss = torch.linalg.matrix_norm(S)
            # test_loss = KM_loss
            flat_loss = torch.abs(S_2d).sum(dim=(-2, -1))

            # iso_vals = iso_vals.detach()
            # k_diff = torch.abs(iso_vals[:, :, 1] - iso_vals[:, :, 2])

            
            

            # Distance to curve points
            C = curve_points.squeeze(0).detach().cpu().numpy().astype(np.float64)
            pcd_C = o3d.geometry.PointCloud()
            pcd_C.points = o3d.utility.Vector3dVector(C)
            
            S = iso_points.squeeze(0).detach().cpu().numpy().astype(np.float64)
            pcd_S = o3d.geometry.PointCloud()
            pcd_S.points = o3d.utility.Vector3dVector(S)
            D = np.asarray(pcd_S.compute_point_cloud_distance(pcd_C))
            D_tensor = torch.tensor(D, dtype=iso_points.dtype, device=iso_points.device)
            D_max = D_tensor.max()
            D_mean = D_tensor.mean()

            # iso_term = dS_loss.squeeze(0).mean()
            # iso_term = mvs_loss.squeeze(0)[D > 0.1].mean()
            # iso_term = dSde_loss.squeeze(0)[D > 0.1].mean()
            # iso_term = (K_G**2).squeeze(0).mean()
            # iso_term = (torch.abs(K_G)).squeeze(0)[D > 0.1].mean()
            # iso_term = (torch.abs(K_M)).squeeze(0)[D > 0.05].mean()
            # iso_term = df_loss.squeeze(0).mean()
            # iso_term = dS_proj_loss.squeeze(0).mean()
            # iso_term = du_loss.squeeze(0).mean()
            # iso_term = df_proj_loss.squeeze(0).mean()
            # iso_term = test_loss.squeeze(0)[D > 0.01].mean()
            # iso_term = test_loss.squeeze(0).mean()

            density_scale = (533.3 * D_max**2 - 16.7 * D_max + 0.5) * D_mean

            if iter <= 600:
                iso_term = (0.5*S_loss + 0.5*ds_loss).squeeze(0).mean() * density_scale
            else:
                if (D > 0.004).sum() > 0:
                    iso_term = (ds_loss).squeeze(0)[D > 0.004].mean() * density_scale
                # else:
                #     iso_term = (ds_proj_loss).squeeze(0).mean() * D_mean

            # import ipdb
            # ipdb.set_trace()


        # eikonal_term = eikonal_loss(nonmnfld_grad=iso_pred, mnfld_grad=near_pred, eikonal_type='abs')
        if iter <= 600:
            if iso_points is not None:
            # if False:
                eikonal_term = relax_eikonal_loss(nonmnfld_grad=iso_pred, mnfld_grad=near_pred, min=0.5, eikonal_type='abs')
            else:
                eikonal_term = relax_eikonal_loss(nonmnfld_grad=None, mnfld_grad=near_pred, min=0.5, eikonal_type='abs')
        else:
            eikonal_term = relax_eikonal_loss(nonmnfld_grad=iso_pred, mnfld_grad=near_pred, eikonal_type='abs', min=0.5)

        inter_term = nonmanifold_pred.norm(2, dim=-1).mean()

        smooth_term = iso_term

        # if iter <= 600:
        #     cc_term = torch.tensor([0.0], device=device)
        
        # Tangent aligning term
        if near_tangents != None:
            # Normal align
            normalized_pred = torch.nn.functional.normalize(near_pred, dim=-1)
            # tangent_term = torch.sum(normalized_pred * near_tangents, dim=-1)**2
            near_points = sub_curve_points
            near_pred = sub_curve_pred
            # Frame align
            near_dx = utils.gradient(near_points, near_pred[:, :, 0])
            near_dy = utils.gradient(near_points, near_pred[:, :, 1])
            near_dz = utils.gradient(near_points, near_pred[:, :, 2])
            near_hessian = torch.stack((near_dx, near_dy, near_dz), dim=-1)
            near_hessian = 0.5 * (near_hessian + torch.transpose(near_hessian, -2, -1))
            n_hat = torch.nn.functional.normalize(near_pred, dim=-1)
            near_P = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0) - n_hat.unsqueeze(-1) * n_hat.unsqueeze(-2)
            near_S = near_P @ near_hessian @ near_P

            # _, near_vecs = torch.linalg.eigh(near_S)
            # _, near_vecs = st.decompose_shape_tensor(near_pred, near_S)

            # f = sh.R2f(near_vecs)
            # tangent_term = sh.alignment_loss(f, near_tangents)
            # tangent_term = torch.mean(tangent_term)
            st = ST.ShapeTensor(device=device)
            tangent_term = st.alignment_loss(n_hat, near_S, near_tangents)

            if torch.isnan(tangent_term).any().item():
                indinces = utils.check_nan(tangent_term)
                import ipdb
                ipdb.set_trace()
                # utils.print_nan(near_S, indinces, "Shape tensor")
                # utils.print_nan(n_hat, indinces, "Normal")

            tangent_term = torch.mean(tangent_term)

        # Curl free term
        non_mnfld_dx = utils.gradient(nonmnfld_points, nonmanifold_pred[:, :, 0])
        non_mnfld_dy = utils.gradient(nonmnfld_points, nonmanifold_pred[:, :, 1])
        non_mnfld_dz = utils.gradient(nonmnfld_points, nonmanifold_pred[:, :, 2])

        Ry = non_mnfld_dz[:, :, 1]
        Qz = non_mnfld_dy[:, :, 2]
        Pz = non_mnfld_dx[:, :, 2]
        Rx = non_mnfld_dz[:, :, 0]
        Qx = non_mnfld_dy[:, :, 0]
        Py = non_mnfld_dx[:, :, 1]

        curl_term = torch.stack((Ry-Qz, Pz-Rx, Qx-Py), dim=-1)
        curl_term = torch.linalg.norm(curl_term, dim=-1)**2
        curl_term = torch.mean(curl_term)


        #########################################
        # Losses
        #########################################

        # losses used in the paper
        # if self.loss_type == 'siren_supervised':  # SIREN loss
        #     loss = self.weights[0] * sdf_term
        # if self.loss_type == 'siren_supervised_normal':  # SIREN loss
        #     loss = self.weights[0] * sdf_term + self.weights[1] * normal_term
        # else:
        #     print(self.loss_type)
        #     raise Warning("unrecognized loss type")
        loss = self.weights[0] * sdf_term\
            + self.weights[1] * curl_term\
            + self.weights[2] * inter_term\
            + self.weights[3] * eikonal_term\
            + self.weights[4] * tangent_term\
            + self.weights[5] * smooth_term\
            # + 1e3 * cc_term
            
            

        # If multiple surface reconstruction, then latent and latent_reg are defined so reg_term need to be used
        if latent_reg is not None:
            loss += self.weights[6] * latent_reg_term

        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': curl_term, 'latent_reg_term': latent_reg_term,
                'eikonal_term': eikonal_term, 'normals_loss': inter_term, 'div_loss': tangent_term,
                'curv_loss': curv_term.mean(), 'morse_term': smooth_term}
    
    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.75, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we
        elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we
        elif self.div_decay == 'step':  # change weight at s
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")