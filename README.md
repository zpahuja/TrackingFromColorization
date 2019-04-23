# Tracking from Colorization

Implements and improves upon [Tracking Emerges by Colorizing Videos](https://arxiv.org/abs/1806.09594) by Vondrick et al.

[![tensorflow](https://img.shields.io/badge/tensorflow-1.10-ed6c20.svg)](https://www.tensorflow.org/)
[![CircleCI](https://circleci.com/gh/wbaek/tracking_via_colorization.svg?style=svg)](https://circleci.com/gh/wbaek/tracking_via_colorization)

## Examples

### Colorization

![tracking via colorization sample2](./out/examples/sample2.gif)

### Segmentation

![tracking via colorization sample0](./out/examples/sample0.gif)
![tracking via colorization sample1](./out/examples/sample1.gif)

## Usage

### Cluster
```
python3 scripts/cluster.py -k 16 -n 1000 -o out/centroids/centroids_16k_cifar10_1000samples.npy
```

### Train

* Colorization
```
python3 scripts/colorization/train.py --model-dir out/models/colorization --gpus 0 1 --num_process 16

tensorboard --host 127.0.0.1 --port 6016 --logdir out/models
```

* CIFAR-10
```
python3 bin/train_estimator_cifar10.py --model-dir models/test
```

### Test

* Colorization
```
python3 bin/test_colorizer.py --checkpoint models/test/model.ckpt-100000 --scale 1 --name davis -o results/davis/
```

## Requirements

- Tensorflow >= 1.10
- opencv >= 3.0

Install below dependencies:

```bash
apt install -y libsm6 libxext-dev libxrender-dev libcap-dev
apt install -y ffmpeg
pip install -r requirements.txt
```

## Software Design

-  **colorizer**: package containing classes and models specific to colorization task
-  **results**: reports and Jupyter notebooks for demos, experiments and ablation studies
- **scripts**: python scripts for setup, training & evaluating models, and reusable functions to make elegant Jupyter notebooks
