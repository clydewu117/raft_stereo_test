from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from raft_stereo import RAFTStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


def validate_osu(model, iters=32, mixed_prec=False):
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.cuda()
    model.eval()
    aug_params = {}
    val_dataset = datasets.OSU(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    with torch.no_grad():
        for val_id in range(len(val_dataset)):
            _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with autocast(enabled=mixed_prec):
                start = time.time()
                _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
                end = time.time()

            if val_id > 50:
                elapsed_list.append(end - start)
            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

            assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
            epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

            epe_flattened = epe.flatten()
            val = valid_gt.flatten() >= 0.5

            out = (epe_flattened > 3.0)
            image_out = out[val].float().mean().item()
            image_epe = epe_flattened[val].mean().item()
            if val_id < 9 or (val_id + 1) % 10 == 0:
                logging.info(
                    f"OSU Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")
            epe_list.append(epe_flattened[val].mean().item())
            out_list.append(out[val].cpu().numpy())

            del image1, image2, flow_gt, valid_gt, flow_pr, epe
            torch.cuda.empty_cache()

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation OSU: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'osu-epe': epe, 'osu-d1': d1}


@torch.no_grad()
def validate_things(model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    epe_by_iter_2d = []
    d1_by_iter_2d = []
    for val_id in range(len(val_dataset)):
        epe_by_iter = [[] for _ in range(iters)]
        d1_by_iter = [[] for _ in range(iters)]
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_predictions = model(image1, image2, iters=iters, test_mode=False)

        for i, flow_pr in enumerate(flow_predictions):  # iterate over flow_predictions (1 per iteration)
            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
            assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
            epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()  # calculate EPE

            epe_flattened = epe.flatten()
            val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

            out = (epe_flattened > 2.0)
            image_out = out[val].float().mean().item()  # D1 for the current iteration
            image_epe = epe_flattened[val].mean().item()  # EPE for the current iteration

            # Append results to the lists
            epe_by_iter[i].append(image_epe)
            d1_by_iter[i].append(image_out)

            logging.info(f"Middlebury Iter {val_id + 1}, Iteration {i + 1}/{iters} - EPE: {round(image_epe, 4)} D1: {round(image_out, 4)}")

        avg_epe_by_iter = [np.mean(epe_list) for epe_list in epe_by_iter]
        avg_d1_by_iter = [100 * np.mean(d1_list) for d1_list in d1_by_iter]

        print(f"Validation Middlebury{split}: EPE {avg_epe_by_iter}, D1 {avg_d1_by_iter}")
        epe_by_iter_2d.append(avg_epe_by_iter)
        d1_by_iter_2d.append(avg_d1_by_iter)

    np_epe = np.array(epe_by_iter_2d)  # shape(10, 32)
    np_d1 = np.array(d1_by_iter_2d)  # shape(10, 32)

    final_epe_by_iter = np.mean(np_epe, axis=0)
    final_d1_by_iter = np.mean(np_d1, axis=0)

    print(f"final EPE by iters: {final_epe_by_iter}")
    print(f"final D1 by iters: {final_d1_by_iter}")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", "osu", "things"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0, 1, 2, 3])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'osu':
        validate_osu(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        validate_things(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
