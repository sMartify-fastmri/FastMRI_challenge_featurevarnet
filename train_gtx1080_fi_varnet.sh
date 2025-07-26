#!/bin/bash

# GTX 1080 최적화된 FI VarNet 학습 스크립트
# - 첫 20 에폭: MRAugment 없이
# - 이후 50 에폭: MRAugment 포함
# - FI VarNet: 3개 feature layer (2개 attention + 1개 일반), 2개 image layer (12 channels)
# - 메모리 최적화: cascade=5, chans=8, sens_chans=4

echo "🚀 Starting GTX 1080 Optimized FI VarNet Training"
echo "================================================"
echo "Configuration:"
echo "  - GPU: GTX 1080"
echo "  - Model: FI VarNet (Feature + Image VarNet)"
echo "  - Cascades: 5 (GTX 1080 optimized)"
echo "  - Channels: 8"
echo "  - Sensitivity Channels: 4"
echo "  - Feature Layers: 3 (2 attention + 1 regular)"
echo "  - Image Layers: 2 (12 channels)"
echo "  - Total Epochs: 70 (20 without aug + 50 with MRAugment)"
echo "  - Learning Rate: 1e-3"
echo "  - Batch Size: 1"
echo ""

python train.py \
    --GPU-NUM 0 \
    --batch-size 1 \
    --num-epochs 70 \
    --lr 1e-3 \
    --report-interval 10 \
    --net-name "gtx1080_fi_varnet" \
    --data-path-train "/Data/train/" \
    --data-path-val "/Data/val/" \
    --model-type "feature_varnet" \
    --varnet-type "fi_varnet" \
    --cascade 5 \
    --chans 8 \
    --sens_chans 4 \
    --feature_model_type "FI" \
    --pools 4 \
    --sens_pools 4 \
    --image_layers 2 \
    --image_chans 12 \
    --feature_layers 3 \
    --attention_layers 2 \
    --input-key "kspace" \
    --target-key "image_label" \
    --max-key "max" \
    --seed 430 \
    --aug-start-epoch 20 \
    --aug_strength 0.5 \
    --aug_schedule "constant" \
    --aug_delay 0

echo ""
echo "✅ Training completed!"
echo "📊 Results will be saved in: ../result/gtx1080_fi_varnet_FI_c5_ch8_sch4_fl3_att2_il2_ich12_mraugment_epoch20/"
echo ""
echo "📈 Training Summary:"
echo "  - Epochs 1-20: No augmentation (clean training)"
echo "  - Epochs 21-70: MRAugment enabled (data augmentation)"
echo "  - Model: GTX 1080 optimized FI VarNet"
echo "  - Memory usage: Optimized for 8GB GTX 1080" 