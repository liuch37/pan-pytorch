# Pixel Aggregation Network

This is an unofficial PyTorch re-implementation of paper "Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network" published in ICCV 2019, with PyTorch >= v1.4.0.

## Task

- [x] Backbone model
- [x] FPEM model
- [x] FFM model
- [x] Integrated model
- [x] Loss Function
- [x] Data preprocessing
- [x] Data postprocessing
- [x] Training pipeline
- [x] Inference pipeline
- [x] Evaluation pipeline

## Command

### Training

``
python train.py --batch 32 --epoch 5000 --dataset_type ctw --gpu True
``

### Inference

``
python inference.py --input ./data/CTW1500/test/text_image --model ./outputs/model_epoch_0.pth --bbox_type poly
``

## Results

### CTW1500
![Statstics for CTW training](https://github.com/liuch37/pan-pytorch/blob/master/misc/ctw_statistics.png)

## Supported Dataset

- [ ] SynthText: https://www.robots.ox.ac.uk/~vgg/data/scenetext/
- [ ] Total-Text: https://github.com/cs-chan/Total-Text-Dataset
- [x] CTW1500: https://github.com/Yuliang-Liu/Curve-Text-Detector

## Source

[1] Original paper: https://arxiv.org/abs/1908.05900

[2] Official PyTorch code: https://github.com/whai362/pan_pp.pytorch
