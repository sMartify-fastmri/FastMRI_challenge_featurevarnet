import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    """
    Data Transformer with optional MRAugment data augmentation support.
    """
    def __init__(self, isforward, max_key, augmentor=None):
        self.isforward = isforward
        self.max_key = max_key
        self.augmentor = augmentor
        self.use_augment = augmentor is not None
    
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        # Convert input to complex tensor
        if isinstance(input, np.ndarray) and input.dtype == np.complex64:
            # Input is already complex
            kspace = to_tensor(input * mask)
        else:
            # Input might be real/imag separated - convert to complex
            if len(input.shape) >= 2 and input.shape[-1] == 2:
                # Last dimension is real/imag
                input_complex = input[..., 0] + 1j * input[..., 1]
                kspace = to_tensor(input_complex * mask)
            else:
                # Assume input is already complex or handle as is
                        kspace = to_tensor(input * mask)
        
        # Apply MRAugment if available and configured
        if self.use_augment and not self.isforward and target is not None:
            # Convert kspace to the format expected by MRAugment (complex tensor with real/imag as last dim)
            if kspace.dtype.is_complex:
                kspace_for_aug = torch.stack([kspace.real, kspace.imag], dim=-1)
            else:
                kspace_for_aug = kspace
            
            # Get current augmentation probability
            if hasattr(self.augmentor, 'schedule_p') and self.augmentor.schedule_p() > 0.0:
                try:
                    # Apply augmentation
                    augmented_kspace, augmented_target = self.augmentor(kspace_for_aug, target.shape)
                    
                    # Convert back to the format expected by the model
                    if augmented_kspace.shape[-1] == 2:  # real/imag format
                        kspace = augmented_kspace
                    target = augmented_target
                    
                except Exception as e:
                    print(f"Warning: MRAugment failed, using original data: {e}")
                    # Fall back to original data processing
                    pass
        
        # Ensure kspace is in the correct format for the model
        if hasattr(kspace, 'dtype') and kspace.dtype.is_complex:
            kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        elif len(kspace.shape) >= 2 and kspace.shape[-1] != 2:
            # If kspace is not in real/imag format, convert it
            if kspace.dtype.is_complex:
                kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        # Ensure mask is in the correct format
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
        return mask, kspace, target, maximum, fname, slice
