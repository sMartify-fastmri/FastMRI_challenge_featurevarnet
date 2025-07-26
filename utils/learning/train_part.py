import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy
import argparse

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss, center_crop
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

# feature_varnet 모델들을 import하기 위한 try-except 구문
try:
    from utils.model.feature_varnet import (
        AttentionFeatureVarNet_n_sh_w,
        E2EVarNet,
        FeatureVarNet_n_sh_w,
        FeatureVarNet_sh_w,
        FIVarNet,
        IFVarNet,
        GTX1080OptimizedFIVarNet,  # 새로운 GTX 1080 최적화 모델
    )
    FEATURE_VARNET_AVAILABLE = True
    print("Feature VarNet models successfully imported!")
except ImportError as e:
    print(f"Warning: feature_varnet modules not available: {e}")
    FEATURE_VARNET_AVAILABLE = False

class FeatureVarNetWrapper(nn.Module):
    """Feature VarNet 모델들의 출력에 center_crop을 적용하는 wrapper 클래스"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, masked_kspace, mask):
        output = self.model(masked_kspace, mask)
        # 384x384로 center crop 적용
        output = center_crop(output, 384, 384)
        return output

def create_model(args):
    """선택된 feature_model_type에 따라 적절한 feature varnet 모델을 생성합니다."""
    
    if not FEATURE_VARNET_AVAILABLE:
        raise ImportError("feature_varnet modules are not available. Please check the import path.")
    
    # acceleration은 일반적으로 4를 기본값으로 사용
    acceleration = 4
    
    if args.feature_model_type == "basic":
        print(f"BUILDING FEATURE VARNET WITH WEIGHT SHARING (basic), chans={args.chans}")
        base_model = FeatureVarNet_sh_w(
            num_cascades=args.cascade,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
        )
    elif args.feature_model_type == "FI":
        print(f"BUILDING FI VARNET, chans={args.chans}")
        base_model = FIVarNet(
            num_cascades=args.cascade,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            acceleration=acceleration,
        )
    elif args.feature_model_type == "Attention":
        print(f"BUILDING ATTENTION FEATURE VARNET WITH WEIGHT SHARING, chans={args.chans}")
        base_model = AttentionFeatureVarNet_n_sh_w(
            num_cascades=args.cascade,
            pools=args.pools,
            chans=args.chans,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            acceleration=acceleration,
        )
    else:
        raise ValueError(f"Unrecognized feature_model_type: {args.feature_model_type}")
    
    # feature_varnet 모델들에 center_crop wrapper 적용
    model = FeatureVarNetWrapper(base_model)
    print("Applied center_crop wrapper to feature_varnet model")
    
    return model


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
    else:
        print('CUDA not available, using CPU')

    # 새로운 모델 생성 함수 사용
    model = create_model(args)
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    # MRAugment DataAugmentor 생성 (epoch별 제어)
    augmentor = None
    augmentor_enabled = None  # 나중에 활성화할 augmentor
    current_epoch = [0]  # mutable object to store current epoch
    
    try:
        # MRAugment import 시도
        import sys
        import os
        mraugment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'MRAugment')
        if mraugment_path not in sys.path:
            sys.path.insert(1, mraugment_path)
        
        from mraugment.data_augment import DataAugmentor
        
        # MRAugment compatibility: max_epochs attribute 추가
        if not hasattr(args, 'max_epochs') and hasattr(args, 'num_epochs'):
            args.max_epochs = args.num_epochs
        
        # aug_start_epoch가 설정되어 있으면 해당 시점부터 augmentation 활성화
        if hasattr(args, 'aug_start_epoch') and args.aug_start_epoch > 0:
            # epoch을 반환하는 함수 정의
            current_epoch_fn = lambda: current_epoch[0]
            
            # 나중에 사용할 augmentor를 미리 생성 (args를 복사해서 aug_on=True로 설정)
            args_with_aug = argparse.Namespace(**vars(args))
            args_with_aug.aug_on = True
            
            augmentor_enabled = DataAugmentor(args_with_aug, current_epoch_fn)
            print(f"MRAugment DataAugmentor prepared to start from epoch {args.aug_start_epoch}")
            print(f"Augmentation settings (will be activated later):")
            print(f"  - aug_strength: {args_with_aug.aug_strength}")
            print(f"  - aug_schedule: {args_with_aug.aug_schedule}")
            print(f"  - aug_delay: {args_with_aug.aug_delay}")
            print(f"  - max_epochs: {args_with_aug.max_epochs}")
        elif hasattr(args, 'aug_on') and args.aug_on:
            # 기존 방식: 처음부터 augmentation 활성화
            current_epoch_fn = lambda: current_epoch[0]
            augmentor = DataAugmentor(args, current_epoch_fn)
            print(f"MRAugment DataAugmentor created with aug_on={args.aug_on}")
        else:
            print("MRAugment available but data augmentation is disabled")
            
    except ImportError as e:
        print(f"MRAugment not available: {e}")
    except Exception as e:
        print(f"Error setting up MRAugment: {e}")

    # 초기 train_loader 생성 (augmentation 없이)
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True, augmentor=augmentor)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # MRAugment 활성화 체크 및 train_loader 재생성
        if (augmentor_enabled is not None and 
            hasattr(args, 'aug_start_epoch') and 
            epoch == args.aug_start_epoch and 
            augmentor is None):
            
            print(f"\n🚀 Activating MRAugment from epoch {epoch}!")
            augmentor = augmentor_enabled
            # train_loader를 augmentation이 활성화된 상태로 재생성
            train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True, augmentor=augmentor)
            print(f"✅ Train loader recreated with MRAugment enabled\n")
        
        # Update current epoch for MRAugment scheduler
        if augmentor is not None:
            current_epoch[0] = epoch
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
