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

Model   | Precision | Recall | F score | FPS (CPU) + pa.py   | FPS (1 GPU) + pa.py | FPS (1 GPU) + pa.pyx |
------- | --------- | ------ | ------- | ------------------- | ------------------- | -------------------- |
PAN-640 | 0.8509    | 0.7927 | 0.8208  | 0.3493              | 4.6347              | 21.167               |

### TotalText
![Statstics for TT training](https://github.com/liuch37/pan-pytorch/blob/master/misc/tt_statistics.png)

Model   | Precision | Recall | F score | FPS (CPU) + pa.py   | FPS (1 GPU) + pa.py | FPS (1 GPU) + pa.pyx |
------- | --------- | ------ | ------- | ------------------- | ------------------- | -------------------- |
PAN-640 | 0.9011    | 0.8040 | 0.8498  | 0.2883              | 7.6481              | 20.390               |

## Supported Dataset

- [x] CTW1500: https://github.com/Yuliang-Liu/Curve-Text-Detector
- [x] Total-Text: https://github.com/cs-chan/Total-Text-Dataset
- [x] SynthText: https://www.robots.ox.ac.uk/~vgg/data/scenetext/

## Source

[1] Original paper: https://arxiv.org/abs/1908.05900

[2] Official PyTorch code: https://github.com/whai362/pan_pp.pytorch
