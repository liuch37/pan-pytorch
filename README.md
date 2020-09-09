# Pixel Aggregation Network

This is a PyTorch implementation of paper "Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network" published in ICCV 2019, with PyTorch >= v1.4.0.

## Task

- [x] Backbone model
- [x] FPEM model
- [x] FFM model
- [x] Integrated model
- [x] Loss Function
- [x] Data preprocessing
- [x] Data postprocessing
- [ ] Training pipeline
- [ ] Inference pipeline

## Command

### Training

``
python train.py --batch 32 --epoch 5000 --dataset_type ctw --gpu True --bbox_type poly
``

### Inference

``
``

## Results

## Supported Dataset

- [ ] SynthText: https://www.robots.ox.ac.uk/~vgg/data/scenetext/
- [ ] Total-Text: https://github.com/cs-chan/Total-Text-Dataset
- [x] CTW1500: https://github.com/Yuliang-Liu/Curve-Text-Detector

## Source

[1] Original paper: https://arxiv.org/abs/1908.05900

[2] Official PyTorch code: https://github.com/whai362/pan_pp.pytorch
