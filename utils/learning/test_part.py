import numpy as np
import torch
import torch.nn as nn
import os

from collections import defaultdict
from utils.common.utils import save_reconstructions, center_crop
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet

# feature_varnet 모델들을 import하기 위한 try-except 구문
try:
    from feature_varnet import (
        AttentionFeatureVarNet_n_sh_w,
        E2EVarNet,
        FeatureVarNet_n_sh_w,
        FeatureVarNet_sh_w,
        FIVarNet,
        IFVarNet,
    )
    FEATURE_VARNET_AVAILABLE = True
except ImportError:
    print("Warning: feature_varnet modules not available. Only E2E VarNet will be supported.")
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
    """선택된 모델 타입에 따라 적절한 모델을 생성합니다."""
    
    if args.model_type == 'e2e_varnet':
        # 기존 E2E VarNet 사용 (이미 center_crop이 적용됨)
        model = VarNet(num_cascades=args.cascade, 
                       chans=args.chans, 
                       sens_chans=args.sens_chans)
        print(f"Created E2E VarNet with cascades={args.cascade}, chans={args.chans}, sens_chans={args.sens_chans}")
        return model
        
    elif args.model_type == 'feature_varnet':
        if not FEATURE_VARNET_AVAILABLE:
            raise ImportError("feature_varnet modules are not available. Please check the import path.")
        
        # acceleration은 일반적으로 4를 기본값으로 사용
        acceleration = 4
        
        if args.varnet_type == "fi_varnet":
            print(f"BUILDING FI VARNET, chans={args.chans}")
            model = FIVarNet(
                num_cascades=args.cascade,
                pools=args.pools,
                chans=args.chans,
                sens_pools=args.sens_pools,
                sens_chans=args.sens_chans,
                acceleration=acceleration,
            )
        elif args.varnet_type == "if_varnet":
            print(f"BUILDING IF VARNET, chans={args.chans}")
            model = IFVarNet(
                num_cascades=args.cascade,
                pools=args.pools,
                chans=args.chans,
                sens_pools=args.sens_pools,
                sens_chans=args.sens_chans,
                acceleration=acceleration,
            )
        elif args.varnet_type == "attention_feature_varnet_sh_w":
            print(f"BUILDING ATTENTION FEATURE VARNET WITH WEIGHT SHARING, chans={args.chans}")
            model = AttentionFeatureVarNet_n_sh_w(
                num_cascades=args.cascade,
                pools=args.pools,
                chans=args.chans,
                sens_pools=args.sens_pools,
                sens_chans=args.sens_chans,
                acceleration=acceleration,
            )
        elif args.varnet_type == "feature_varnet_n_sh_w":
            print(f"BUILDING FEATURE VARNET WITHOUT WEIGHT SHARING, chans={args.chans}")
            model = FeatureVarNet_n_sh_w(
                num_cascades=args.cascade,
                pools=args.pools,
                chans=args.chans,
                sens_pools=args.sens_pools,
                sens_chans=args.sens_chans,
            )
        elif args.varnet_type == "feature_varnet_sh_w":
            print(f"BUILDING FEATURE VARNET WITH WEIGHT SHARING, chans={args.chans}")
            model = FeatureVarNet_sh_w(
                num_cascades=args.cascade,
                pools=args.pools,
                chans=args.chans,
                sens_pools=args.sens_pools,
                sens_chans=args.sens_chans,
            )
        elif args.varnet_type == "e2e_varnet":
            print(f"BUILDING E2E VARNET (from feature_varnet), chans={args.chans}")
            model = E2EVarNet(
                num_cascades=args.cascade,
                pools=args.pools,
                chans=args.chans,
                sens_pools=args.sens_pools,
                sens_chans=args.sens_chans,
            )
        else:
            raise ValueError(f"Unrecognized varnet_type: {args.varnet_type}")
        
        print(f"Created feature_varnet model: {args.varnet_type}")
        return model
    else:
        raise ValueError(f"Unrecognized model_type: {args.model_type}")

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)
            
            # feature_varnet 모델의 경우 center_crop 적용
            if hasattr(args, 'model_type') and args.model_type == 'feature_varnet':
                output = center_crop(output, 384, 384)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    # 새로운 모델 생성 함수 사용
    model = create_model(args)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    
    # 체크포인트의 키에서 'model.' 접두사 제거
    state_dict = checkpoint['model']
    if any(key.startswith('model.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # 'model.' 제거 (6글자)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)