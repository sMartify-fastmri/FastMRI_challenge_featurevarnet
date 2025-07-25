import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

# fastMRI feature_varnet 경로 추가
fastmri_path = os.path.join(os.path.dirname(os.getcwd()), 'fastMRI')
if fastmri_path not in sys.path:
    sys.path.insert(1, fastmri_path)

feature_varnet_path = os.path.join(fastmri_path, 'fastmri_examples/feature_varnet')
if feature_varnet_path not in sys.path:
    sys.path.insert(1, feature_varnet_path)

from utils.learning.test_part import forward
import time

    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    
    # 모델 타입 선택 옵션 추가
    parser.add_argument('--model-type', type=str, default='e2e_varnet', 
                       choices=['e2e_varnet', 'feature_varnet'], 
                       help='Type of VarNet model to use')
    
    # feature_varnet 관련 옵션 추가
    parser.add_argument('--varnet-type', type=str, default='fi_varnet',
                       choices=['fi_varnet', 'if_varnet', 'feature_varnet_sh_w', 
                               'feature_varnet_n_sh_w', 'attention_feature_varnet_sh_w', 'e2e_varnet'],
                       help='Specific type of feature varnet (only used when model-type is feature_varnet)')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    
    # feature_varnet에서 추가로 필요한 파라미터들
    parser.add_argument('--pools', type=int, default=4, help='Number of pooling layers for U-Net (for feature_varnet)')
    parser.add_argument('--sens_pools', type=int, default=4, help='Number of pooling layers for sensitivity estimation U-Net (for feature_varnet)')
    
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    args = parser.parse_args()
    
    # net_name에서 model_type과 varnet_type 자동 추론
    net_name_str = str(args.net_name)
    if 'fi_varnet' in net_name_str:
        args.model_type = 'feature_varnet'
        if 'fi_varnet_light' in net_name_str:
            args.varnet_type = 'fi_varnet'
        elif 'if_varnet' in net_name_str:
            args.varnet_type = 'if_varnet'
        else:
            args.varnet_type = 'fi_varnet'
    
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'

    start_time = time.time()
    
    # acc4
    args.data_path = args.path_data / "acc4"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!') if reconstructions_time < 3600 else print('Fail!')