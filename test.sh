CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
    --data-root /data/pos+mot/Datadir/ \
    --exp-name rfmotip_dancetrack \
    --config-path ./configs/rf_detr_motip_dancetrack.yaml
