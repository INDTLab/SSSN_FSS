r""" ASSN testing code """
import argparse
import os

import torch.nn.functional as F
import torch.nn as nn
import torch
from config.config import get_cfg_defaults

from model.assn import ASSN
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from model.siamese import SiameseNetwork
import heapq

def test(model, siamese_model, dataloader, nshot):
    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # 1. ASSN forward pass
        batch = utils.to_cuda(batch)

        score_list = []
        support_images = []
        support_masks = []
        for i in range(9):
            score_list.append(siamese_model(batch['query_img'], batch['support_imgs'][:, i])) 
        arr_max = heapq.nlargest(nshot, score_list)
        index_max = map(score_list.index, arr_max)
        for i in index_max:
            support_images.append(batch['support_imgs'][:, i]) 
            support_masks.append(batch['support_masks'][:, i])
        support_images = torch.stack(support_images, dim=1)
        support_masks = torch.stack(support_masks, dim = 1)
        batch['support_imgs'] = support_images.cuda()
        batch['support_masks'] = support_masks.cuda()
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='ASSN Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../Datasets_ASSN')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(args.load, 'config.yaml'))
    cfg.freeze()

    Logger.initialize(args, training=False, cfg=cfg, benchmark=cfg.TRAIN.BENCHMARK, logpath=args.logpath)

    # Model initialization
    model = ASSN(cfg, False)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(os.path.join(args.load, 'best_model.pt'))['state_dict'])

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset.initialize(benchmark=cfg.TRAIN.BENCHMARK, img_size=cfg.TRAIN.IMG_SIZE, datapath=args.datapath, use_original_imgsize=False)
    dataloader_test = FSSDataset.build_dataloader(cfg.TRAIN.BENCHMARK, args.bsz, args.nworker, cfg.TRAIN.FOLD, 'test', args.nshot)


    siamese_model = SiameseNetwork()
    siamese_model.to(device)
    siamese_model=torch.load('./siamese_model.pt',map_location='cuda')
    siamese_model.eval()
    
    # Test ASSN
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, siamese_model, dataloader_test, args.nshot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (cfg.TRAIN.FOLD, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
