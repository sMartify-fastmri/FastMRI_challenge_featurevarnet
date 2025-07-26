import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

# fastMRI feature_varnet 경로 추가
fastmri_path = os.path.join(os.path.dirname(os.getcwd()), 'fastMRI')
if fastmri_path not in sys.path:
    sys.path.insert(1, fastmri_path)

feature_varnet_path = os.path.join(fastmri_path, 'fastmri_examples/feature_varnet')
if feature_varnet_path not in sys.path:
    sys.path.insert(1, feature_varnet_path)

# MRAugment 경로 추가
mraugment_path = os.path.join(os.path.dirname(os.getcwd()), 'MRAugment')
if mraugment_path not in sys.path:
    sys.path.insert(1, mraugment_path)

from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

# MRAugment import
try:
    from mraugment.data_augment import DataAugmentor
    MRAUGMENT_AVAILABLE = True
    print("MRAugment successfully imported!")
except ImportError as e:
    print(f"Warning: MRAugment not available: {e}")
    MRAUGMENT_AVAILABLE = False


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=10, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    # Feature VarNet 관련 arguments (기존 CLI 호환)
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--feature_model_type', type=str, default='basic', choices=['basic', 'FI', 'Attention'], 
                        help='Type of Feature VarNet: basic (FeatureVarNet_sh_w), FI (FIVarNet), Attention (AttentionFeatureVarNet)')  ## feature varnet type
    
    # feature_varnet에서 추가로 필요한 파라미터들
    parser.add_argument('--pools', type=int, default=4, help='Number of pooling layers for U-Net (for feature_varnet)')
    parser.add_argument('--sens_pools', type=int, default=4, help='Number of pooling layers for sensitivity estimation U-Net (for feature_varnet)')
    
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    # MRAugment 관련 arguments 추가
    if MRAUGMENT_AVAILABLE:
        parser = DataAugmentor.add_augmentation_specific_args(parser)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    # Feature 모델 타입에 따라 네트워크 이름 수정
    if hasattr(args, 'feature_model_type'):
        args.net_name = Path(f"{args.net_name}_{args.feature_model_type}")
    
    # MRAugment가 활성화된 경우 네트워크 이름에 표시
    if MRAUGMENT_AVAILABLE and hasattr(args, 'aug_on') and args.aug_on:
        args.net_name = Path(f"{args.net_name}_mraugment")

    result_base = Path('../result') / args.net_name
    args.exp_dir = result_base / 'checkpoints'
    args.val_dir = result_base / 'reconstructions_val'
    args.main_dir = result_base / Path(__file__).name
    args.val_loss_dir = result_base

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
