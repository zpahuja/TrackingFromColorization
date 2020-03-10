# Tracking from Colorization

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

### Test

* Colorization
```
python3 bin/test_colorizer.py --checkpoint models/test/model.ckpt-100000 --scale 1 --name davis -o results/davis/
```

## Requirements

- Tensorflow >= 1.10
- opencv >= 3.0

## Software Design

-  **colorizer**: package containing classes and models specific to colorization task
-  **results**: reports and Jupyter notebooks for demos, experiments and ablation studies
- **scripts**: python scripts for setup, training & evaluating models, and reusable functions to make elegant Jupyter notebooks
