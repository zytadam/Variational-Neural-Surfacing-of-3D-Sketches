import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.optim as optim
from torchinfo import summary

from models import Network, FlowLoss
import utils.utils as utils
import utils.visualizations as vis
import utils.gridEvaluator as grid_eval
import surface_recon_args
import surface_reconstruction.recon_dataset as dataset
import utils.curveNetwork as CN

# get training parameters
args = surface_recon_args.get_args()
file_name = os.path.splitext(args.data_path.split('/')[-1])[0]
logdir = os.path.join(args.logdir, file_name)
os.makedirs(logdir, exist_ok=True)

# set up logging
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(logdir, args)
os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
os.system('cp %s %s' % ('recon_dataset.py', logdir))  # backup the current training file
os.system('cp %s %s' % ('../models/overfit_network.py', logdir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the losses files

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# get data loaders
utils.same_seed(args.seed)

# get model
net = Network(in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim, nl=args.nl,
              decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type,
              sphere_init_params=args.sphere_init_params, udf=args.udf)

# Load model
model_name = "coffee_base"
# net.load_state_dict(torch.load(os.path.join(args.logdir, model_name, 'trained_models', '999.pth')))

net.to(device)
summary(net.decoder, (1, 1024, 3))

# train_set = dataset.ReconDataset(args.data_path, args.n_points, args.n_samples, args.grid_res)
curve_net = CN.CurveNetwork(args.data_path)
train_set = dataset.SuperDataset(args.data_path, args.n_points, args.n_samples, args.grid_res, grid_range=args.grid_size/2)
evaluator = grid_eval.GridEvaluator(grid_res=args.grid_res, grid_size=args.grid_size, device=device)
# evaluator.load_grid_values(os.path.join(args.logdir, model_name, 'grid_values', '999.npy'))
evaluator.set_curve_network(curve_net)
train_set.set_grid_evaluator(evaluator)
train_set.set_curve_network(curve_net)

# train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
#                                                pin_memory=True)
train_dataloader = dataset.SuperDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                               pin_memory=True)

n_parameters = utils.count_parameters(net)
utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

# Setup Adam optimizers
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
# optimizer = optim.LBFGS(net.parameters(), lr=args.lr, line_search_fn='strong_wolfe')
n_iterations = args.n_samples * (args.num_epochs)
print('n_iterations: ', n_iterations)

net.to(device)


criterion = FlowLoss(weights=args.loss_weights, loss_type=args.loss_type, div_decay=args.morse_decay,
                      div_type=args.morse_type, bidirectional_morse=args.bidirectional_morse, udf=args.udf)

num_batches = len(train_dataloader)
refine_flag = True
min_cd = np.inf
max_f1 = -np.inf
# For each epoch
for epoch in range(args.num_epochs):
    # For each batch in the dataloader
    for batch_idx, data in enumerate(train_dataloader):
        if (batch_idx % 100 == 0 or batch_idx == len(train_dataloader) - 1):
            torch.save(net.state_dict(), os.path.join(model_outdir, str(batch_idx) + '.pth'))

            os.makedirs(os.path.join(logdir, 'grid_values'), exist_ok=True)
            evaluator.save_grid_values(path=os.path.join(logdir, 'grid_values', str(batch_idx) + '.npy'))
            evaluator.plot_grid_values(path=os.path.join(logdir, 'mesh'), batch_id=batch_idx, show=False)
            os.makedirs(os.path.join(logdir, 'ras_p'), exist_ok=True)
            evaluator.save_ras_p(path=os.path.join(logdir, 'ras_p', str(batch_idx) + '.npy'))

            # output_dir = os.path.join(logdir, 'vis')
            # os.makedirs(output_dir, exist_ok=True)
            # vis.plot_cuts_iso(net.decoder, save_path=os.path.join(output_dir, str(batch_idx) + '.html'))
            # try:
            #     shapename = file_name
            #     output_dir = os.path.join(logdir, 'result_meshes')
            #     os.makedirs(output_dir, exist_ok=True)
            #     cp, scale, bbox = train_set.cp, train_set.scale, train_set.bbox
            #     mesh_dict = None
            #     if args.udf:
            #         res_dict = utils.udf2mesh(net.decoder, None,
            #                                   args.grid_res,
            #                                   translate=-cp,
            #                                   scale=1 / scale,
            #                                   get_mesh=True, device=device, bbox=bbox)
            #         mesh = res_dict['mesh']
            #         mesh = utils.normalize_mesh_export(mesh)
            #     else:
            #         mesh = utils.implicit2mesh(net.decoder, None,
            #                                    args.grid_res,
            #                                    translate=-cp,
            #                                    scale=1 / scale,
            #                                    get_mesh=True, device=device, bbox=bbox)

            #     pred_mesh = mesh.copy()
            #     output_ply_filepath = os.path.join(output_dir,
            #                                        shapename + '_iter_{}.ply'.format(batch_idx))

            #     print('Saving to ', output_ply_filepath)
            #     mesh.export(output_ply_filepath)
            # except Exception as e:
            #     print(e)
            #     print('Could not generate mesh\n')

        

        nonmnfld_points, curve_points, curve_tangents, iso_points, sub_curve_points, knn_idx = data['nonmnfld_points'].to(device), data['curve_points'].to(device), \
            data["curve_tangents"].to(device), data["iso_points"].to(device), data["sub_curve_points"].to(device), data["knn_index"].to(device)

        # path = rf"iso_points_{batch_idx}"
        # np.save(path, iso_points.detach().cpu().numpy())

        nonmnfld_points.requires_grad_()
        curve_points.requires_grad_()
        sub_curve_points.requires_grad_()
        iso_points.requires_grad_()

        if iso_points.numel() == 0:
            iso_points = None
            knn_idx = None

        net.zero_grad()
        net.train()


        output_pred = net(nonmnfld_points, iso_points, curve_points, sub_curve_points)

        loss_dict = criterion(output_pred, iso_points, nonmnfld_points, 
                                 curve_points, sub_curve_points, curve_tangents, knn_idx=knn_idx, grid_evaluator=evaluator, iter=batch_idx)
        
        loss_dict["loss"].backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)

        optimizer.step()

        if (batch_idx % 100 == 0 or batch_idx == len(train_dataloader) - 1):

            os.makedirs(os.path.join(logdir, 'samplings'), exist_ok=True)
            curve_pos = curve_points.squeeze(0).detach().cpu().numpy()
            curve_vec = output_pred['curve_points_pred'].squeeze(0).detach().cpu().numpy()
            np.save(os.path.join(logdir, 'samplings', str(batch_idx) + '_curve_pos.npy'), curve_pos)
            np.save(os.path.join(logdir, 'samplings', str(batch_idx) + '_curve_vec.npy'), curve_vec)
            if iso_points is not None:
                iso_pos = iso_points.squeeze(0).detach().cpu().numpy()
                iso_vec = output_pred['manifold_pnts_pred'].squeeze(0).detach().cpu().numpy()
                np.save(os.path.join(logdir, 'samplings', str(batch_idx) + '_iso_pos.npy'), iso_pos)
                np.save(os.path.join(logdir, 'samplings', str(batch_idx) + '_iso_vec.npy'), iso_vec)
        

        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        loss_dict["lr"] = lr
        utils.log_losses(log_writer_train, epoch, batch_idx, num_batches, loss_dict, args.batch_size)


        # Output training stats
        if batch_idx % 10 == 0:
            # print("Iso samples num: ", iso_points.shape[1] if iso_points is not None else 0)
            weights = criterion.weights
            utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Data: {:.5f} + '
                             'L_Curl: {:.5f} + L_Inter: {:.5f} + L_Eknl: {:.5f} + L_Align: {:.5f} + L_Smooth: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                loss_dict["loss"].item(), weights[0] * loss_dict["sdf_term"].item(),
                       weights[1] * loss_dict["inter_term"].item(),
                       weights[2] * loss_dict["normals_loss"].item(), weights[3] * loss_dict["eikonal_term"].item(),
                       weights[4] * loss_dict["div_loss"].item(), weights[5] * loss_dict['morse_term'].item(),
            ),
                log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Data: {:.5f},  '
                             'L_Curl: {:.5f},  L_Inter: {:.5f},  L_Eknl: {:.5f}, L_Align: {:.5f}, L_Smooth: {:.5f}'.format(
                epoch, batch_idx * args.batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
                loss_dict["div_loss"].item(), loss_dict['morse_term'].item()),
                log_file)
            utils.log_string('', log_file)
            

        criterion.update_morse_weight(epoch * args.n_samples + batch_idx, args.num_epochs * args.n_samples,
                                      args.decay_params)  # assumes batch size of 1
