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
    parser.add_argument('-e', '--num-epochs', type=int, default=70, help='Number of epochs (20 without aug + 50 with aug)')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=10, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    # GTX 1080 최적화된 FI VarNet 설정
    parser.add_argument('--model-type', type=str, default='feature_varnet', choices=['e2e_varnet', 'feature_varnet'], 
                        help='Model type')
    parser.add_argument('--varnet-type', type=str, default='fi_varnet', 
                        choices=['fi_varnet', 'if_varnet', 'attention_feature_varnet_sh_w', 'feature_varnet_n_sh_w', 'feature_varnet_sh_w', 'e2e_varnet'],
                        help='VarNet type')
    parser.add_argument('--cascade', type=int, default=5, help='Number of cascades (reduced for GTX 1080)') ## GTX 1080 최적화
    parser.add_argument('--chans', type=int, default=8, help='Number of channels for cascade U-Net') ## 사용자 요구사항
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net') ## GTX 1080 최적화
    parser.add_argument('--feature_model_type', type=str, default='FI', choices=['basic', 'FI', 'Attention'], 
                        help='Type of Feature VarNet: basic (FeatureVarNet_sh_w), FI (FIVarNet), Attention (AttentionFeatureVarNet)')
    
    # feature_varnet에서 추가로 필요한 파라미터들
    parser.add_argument('--pools', type=int, default=4, help='Number of pooling layers for U-Net (for feature_varnet)')
    parser.add_argument('--sens_pools', type=int, default=4, help='Number of pooling layers for sensitivity estimation U-Net (for feature_varnet)')
    
    # Image layer 설정 (사용자 요구사항: 2개 layer, 12 channel)
    parser.add_argument('--image_layers', type=int, default=2, help='Number of image layers')
    parser.add_argument('--image_chans', type=int, default=12, help='Number of channels for image layers')
    
    # Feature layer 설정 (사용자 요구사항: 3개 layer, 2개 attention + 1개 일반)
    parser.add_argument('--feature_layers', type=int, default=3, help='Number of feature layers')
    parser.add_argument('--attention_layers', type=int, default=2, help='Number of attention layers in feature layers')
    
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    # MRAugment 활성화 시점 설정
    parser.add_argument('--aug-start-epoch', type=int, default=20, help='Epoch to start MRAugment (default: 20)')
    
    # MRAugment 관련 arguments 추가
    if MRAUGMENT_AVAILABLE:
        parser = DataAugmentor.add_augmentation_specific_args(parser)
        # 기본값으로 aug_on을 False로 설정 (20 에폭 후에 활성화할 예정)
        parser.set_defaults(aug_on=False)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    # Feature 모델 타입에 따라 네트워크 이름 수정
    if hasattr(args, 'feature_model_type'):
        net_name = f"{args.net_name}_{args.feature_model_type}"
        # GTX 1080 최적화 설정 추가
        net_name += f"_c{args.cascade}_ch{args.chans}_sch{args.sens_chans}"
        # Feature layer 구성 정보 추가
        net_name += f"_fl{args.feature_layers}_att{args.attention_layers}"
        # Image layer 구성 정보 추가
        net_name += f"_il{args.image_layers}_ich{args.image_chans}"
        args.net_name = Path(net_name)
    
    # MRAugment 예정 표시
    if MRAUGMENT_AVAILABLE:
        args.net_name = Path(f"{args.net_name}_mraugment_epoch{args.aug_start_epoch}")
        print(f"MRAugment will be activated from epoch {args.aug_start_epoch}")

    result_base = Path('../result') / args.net_name
    args.exp_dir = result_base / 'checkpoints'
    args.val_dir = result_base / 'reconstructions_val'
    args.main_dir = result_base / Path(__file__).name
    args.val_loss_dir = result_base

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training configuration:")
    print(f"  Model: {args.varnet_type}")
    print(f"  Cascades: {args.cascade}")
    print(f"  Channels: {args.chans}")
    print(f"  Sensitivity Channels: {args.sens_chans}")
    print(f"  Feature Layers: {args.feature_layers} (Attention: {args.attention_layers})")
    print(f"  Image Layers: {args.image_layers} (Channels: {args.image_chans})")
    print(f"  Total Epochs: {args.num_epochs}")
    print(f"  MRAugment starts from epoch: {args.aug_start_epoch}")
    print(f"  GTX 1080 optimized settings applied")

    train(args)
