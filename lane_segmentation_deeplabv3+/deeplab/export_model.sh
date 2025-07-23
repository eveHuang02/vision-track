#!/bin/bash

# python3 train.py \
# --logtostderr \
# --training_number_of_steps=30000 \
# --train_split="train" \
# --model_variant="mobilenet_v2" \
# --train_crop_size="512,512" \
# --train_batch_size=4 \
# --dataset="custom" \
# --fine_tune_batch_norm=False \
# --tf_initial_checkpoint="/home/data/models/research/deeplab/datasets/custom/init_models/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000" \
# --train_logdir="datasets/custom/exp/train_on_trainval_set/train" \
# --dataset_dir="datasets/PascalVOC2012/VOC2012/tfrecord" \
# --initialize_last_layer=False \
# --last_layers_contain_logits_only=False

# python3 train.py \
# --logtostderr \
# --training_number_of_steps=30000 \
# --train_split="train" \
# --model_variant="mobilenet_v2" \
# --train_crop_size="512,512" \
# --train_batch_size=4 \
# --dataset="custom" \
# --fine_tune_batch_norm=False \
# --tf_initial_checkpoint="datasets/custom/exp/train_on_trainval_set/train_0/model.ckpt-30000" \
# --train_logdir="datasets/custom/exp/train_on_trainval_set/train_1" \
# --dataset_dir="datasets/custom/tfrecord" \
# --initialize_last_layer=True \
# --last_layers_contain_logits_only=False

# python3 vis.py --logtostderr \
# --vis_split="val" \
# --model_variant="mobilenet_v2" \
# --vis_crop_size="512,512" \
# --dataset="custom" \
# --checkpoint_dir="datasets/custom/exp/train_on_trainval_set/train_1" \
# --vis_logdir="datasets/custom/exp/train_on_trainval_set/vis" \
# --dataset_dir="datasets/custom/tfrecord" \
# --max_number_of_iterations=1 \
# --eval_interval_secs=0

python3 export_model.py \
--logtostderr \
--checkpoint_path="datasets/custom/exp/train_on_trainval_set/train_1/model.ckpt-30000" \
--export_path="datasets/custom/exp/train_on_trainval_set/export/frozen_inference_graph.pb" \
--model_variant="mobilenet_v2" \
--num_classes=2 \
--crop_size=512 \
--crop_size=512 \
--inference_scales=1.0

# python3 eval.py \
# --logtostderr \
# --eval_split="val" \
# --model_variant="mobilenet_v2" \
# --eval_crop_size="512,512" \
# --checkpoint_dir="datasets/custom/exp/train_on_trainval_set/train_1" \
# --eval_logdir="datasets/custom/exp/train_on_trainval_set/eval" \
# --dataset_dir="datasets/lane_data/test/tfrecord" \
# --dataset="test" \
# --max_number_of_evaluations=1