r""" training (validation) code """
import argparse
import os, datetime

import torch.optim as optim
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
from config.config import get_cfg_defaults
from model.assn import ASSN
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from model.siamese import SiameseNetwork, train_siamese
import heapq
from torch.profiler import profile, record_function, ProfilerActivity


def train(epoch, model, siamese_model, dataloader, optimizer, training):

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    siamese_model.eval()
    average_meter = AverageMeter(dataloader.dataset)
   
    for idx, batch in enumerate(dataloader):
        
        # 1. ASSN forward pass
        batch = utils.to_cuda(batch)
        with torch.no_grad():
            query_img = batch['query_img'].cuda()
            support_imgs = batch['support_imgs'].cuda()
            score_list = []
            for i in range(9): # n=9
                score = siamese_model(query_img, support_imgs[:, i])
                score_list.append(score)
            
                score_tensor = torch.stack(score_list, dim=1)
                top_scores, top_indices = torch.topk(score_tensor, k = 1, dim=1)
                reshaped_indices = top_indices.view(-1)
                batch_index = torch.arange(reshaped_indices.shape[0])
                reshaped_indices = [batch_index, reshaped_indices]
                support_images = support_imgs[reshaped_indices]
                support_masks = batch['support_masks'][reshaped_indices]
                
        logit_mask = model(batch['query_img'], support_images.squeeze(1), support_masks.squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

def siamese(epoch, model, train_loader):
    train_siamese(model, device, train_loader, optimizer, epoch)
    scheduler.step()

    return model

def split_params(model):
    encoder_param = []
    decoder_param = []

    for name, param in model.named_parameters():
        if 'decoder' in name:
            decoder_param.append(param)
        else:
            encoder_param.append(param)

    return encoder_param, decoder_param

    
if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='ASSN Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../Datasets_ASSN')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    
    cfg = get_cfg_defaults()
    
    if args.load is None:
        cfg.merge_from_file(args.config)
    else:
        cfg.merge_from_file(os.path.join('logs', args.load, 'config.yaml'))
        # Load from specified path
    cfg.freeze()

    logpath = args.logpath if args.load is None else args.load
    Logger.initialize(args, training=True, cfg=cfg, benchmark=cfg.TRAIN.BENCHMARK, logpath=logpath)

    # Model initialization
    model = ASSN(cfg, False)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    encoder_param, decoder_param = split_params(model)
    optimizer = optim.AdamW([
        {"params": encoder_param, "lr": cfg.TRAIN.LR, "weight_decay": cfg.TRAIN.WEIGHT_DECAY},
        {"params": decoder_param, "lr": cfg.TRAIN.DECODER_LR, "weight_decay": cfg.TRAIN.DECODER_WEIGHT_DECAY},
    ])
    
    if cfg.TRAIN.LR_SCHEDULER == 'constant':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[] if cfg.TRAIN.MILESTONES is None else cfg.TRAIN.MILESTONES, gamma=1.)
    elif cfg.TRAIN.LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.NITER, eta_min=1e-6)
    else:
        raise NotImplementedError('Invalid learning rate scheduler.')
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(benchmark=cfg.TRAIN.BENCHMARK, img_size=cfg.TRAIN.IMG_SIZE, datapath=args.datapath, use_original_imgsize=False,
        apply_cats_augmentation=cfg.TRAIN.CATS_AUGMENTATIONS, apply_pfenet_augmentation=cfg.TRAIN.PFENET_AUGMENTATIONS)
    dataloader_trn = FSSDataset.build_dataloader(cfg.TRAIN.BENCHMARK, cfg.TRAIN.BSZ, cfg.SYSTEM.NUM_WORKERS, cfg.TRAIN.FOLD, 'trn')
    dataloader_val = FSSDataset.build_dataloader(cfg.TRAIN.BENCHMARK, cfg.TRAIN.BSZ, cfg.SYSTEM.NUM_WORKERS, cfg.TRAIN.FOLD, 'val')
    dataloader_trn_siamese = FSSDataset.build_dataloader_siamese(cfg.TRAIN.SIA_BSZ, cfg.SYSTEM.NUM_WORKERS, 'trn')


    # Train VAT
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    start_epoch = 0

    if args.load is not None:
        model, optimizer, scheduler, start_epoch, best_val_miou =\
             Logger.load_checkpoint(model, optimizer, scheduler)
             
    siamese_model = SiameseNetwork()
    siamese_model = nn.DataParallel(siamese_model)
    siamese_model.to(device)
    optimizer_siamese = optim.Adadelta(siamese_model.parameters(), lr=0.1)
    scheduler_siamese = StepLR(optimizer_siamese, step_size=1, gamma=0.7)


    for epoch in range(start_epoch, cfg.TRAIN.NITER):
        tic = datetime.datetime.now()
        siamese_model = siamese(epoch, siamese_model, dataloader_trn_siamese)

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, siamese_model, dataloader_trn, optimizer, training=True)
        scheduler.step()

        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, siamese_model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(epoch, model, optimizer, scheduler, best_val_miou)
            torch.save(siamese_model,'siamese_model.pt')
        Logger.save_recent_model(epoch, model, optimizer, scheduler, best_val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        toc = datetime.datetime.now()
        Logger.info('Time: #param.:%s' % (toc-tic))
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
    


