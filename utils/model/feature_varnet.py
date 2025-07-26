"""
Feature VarNet 모델 for FastMRI Challenge
fastMRI의 feature_varnet 모델을 FastMRI_challenge에서 사용할 수 있도록 wrapper 제공
"""

import sys
import os
from pathlib import Path

# fastMRI 패키지를 import하기 위해 경로 추가
fastmri_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'fastMRI')
if fastmri_path not in sys.path:
    sys.path.insert(0, fastmri_path)

import torch
import torch.nn as nn
from fastmri.models.feature_varnet import (
    FIVarNet,
    IFVarNet, 
    FeatureVarNet_sh_w,
    FeatureVarNet_n_sh_w,
    AttentionFeatureVarNet_n_sh_w,
    E2EVarNet
)
from fastmri.coil_combine import rss
from fastmri.math import complex_abs
from fastmri.fftc import ifft2c_new as ifft2c


class FeatureVarNetWrapper(nn.Module):
    """
    Feature VarNet wrapper to match VarNet interface in FastMRI_challenge
    """
    
    def __init__(self, 
                 model_type="FeatureVarNet_sh_w",
                 num_cascades=12, 
                 chans=18, 
                 sens_chans=8,
                 pools=4,
                 sens_pools=4,
                 acceleration=4,
                 mask_center=True):
        super().__init__()
        
        self.model_type = model_type
        
        if model_type == "FIVarNet":
            self.model = FIVarNet(
                num_cascades=num_cascades,
                chans=chans,
                sens_chans=sens_chans,
                pools=pools,
                sens_pools=sens_pools,
                acceleration=acceleration,
                mask_center=mask_center
            )
        elif model_type == "IFVarNet":
            self.model = IFVarNet(
                num_cascades=num_cascades,
                chans=chans,
                sens_chans=sens_chans,
                pools=pools,
                sens_pools=sens_pools,
                acceleration=acceleration,
                mask_center=mask_center
            )
        elif model_type == "FeatureVarNet_sh_w":
            self.model = FeatureVarNet_sh_w(
                num_cascades=num_cascades,
                chans=chans,
                sens_chans=sens_chans,
                pools=pools,
                sens_pools=sens_pools,
                mask_center=mask_center
            )
        elif model_type == "FeatureVarNet_n_sh_w":
            self.model = FeatureVarNet_n_sh_w(
                num_cascades=num_cascades,
                chans=chans,
                sens_chans=sens_chans,
                pools=pools,
                sens_pools=sens_pools,
                mask_center=mask_center
            )
        elif model_type == "AttentionFeatureVarNet_n_sh_w":
            self.model = AttentionFeatureVarNet_n_sh_w(
                num_cascades=num_cascades,
                chans=chans,
                sens_chans=sens_chans,
                pools=pools,
                sens_pools=sens_pools,
                acceleration=acceleration,
                mask_center=mask_center
            )
        elif model_type == "E2EVarNet":
            self.model = E2EVarNet(
                num_cascades=num_cascades,
                chans=chans,
                sens_chans=sens_chans,
                pools=pools,
                sens_pools=sens_pools,
                mask_center=mask_center
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, masked_kspace, mask, num_low_frequencies=None, crop_size=None):
        """
        Forward pass that matches VarNet interface
        
        Args:
            masked_kspace: Masked k-space data
            mask: Sampling mask
            num_low_frequencies: Number of low frequencies (optional)
            crop_size: Crop size (optional)
        
        Returns:
            Reconstructed image
        """
        # 기본 crop_size를 384x384로 설정 (FastMRI challenge 표준)
        if crop_size is None:
            crop_size = (384, 384)
            
        return self.model(
            masked_kspace=masked_kspace,
            mask=mask,
            num_low_frequencies=num_low_frequencies,
            crop_size=crop_size
        )


# 기존 VarNet과 같은 인터페이스로 사용할 수 있는 클래스들
class FeatureVarNet(FeatureVarNetWrapper):
    """Default Feature VarNet (shared weight version)"""
    def __init__(self, num_cascades=12, chans=18, sens_chans=8):
        super().__init__(
            model_type="FeatureVarNet_sh_w",
            num_cascades=num_cascades,
            chans=chans,
            sens_chans=sens_chans
        )


class FIFeatureVarNet(FeatureVarNetWrapper):
    """Feature-Image VarNet"""
    def __init__(self, num_cascades=12, chans=18, sens_chans=8, acceleration=4):
        super().__init__(
            model_type="FIVarNet",
            num_cascades=num_cascades,
            chans=chans,
            sens_chans=sens_chans,
            acceleration=acceleration
        )


class AttentionFeatureVarNet(FeatureVarNetWrapper):
    """Attention Feature VarNet"""
    def __init__(self, num_cascades=12, chans=18, sens_chans=8, acceleration=4):
        super().__init__(
            model_type="AttentionFeatureVarNet_n_sh_w",
            num_cascades=num_cascades,
            chans=chans,
            sens_chans=sens_chans,
            acceleration=acceleration
        ) 