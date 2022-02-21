# SAITS

Code repository for paper *[SAITS: Self-Attention-based Imputation for Time Series](https://arxiv.org/abs/2202.08516)*.

The implementation of SAITS is in dir `modeling`. We give configurations of our models in dir `configs`, provide dataset links and preprocessing scripts in dir `dataset_generating_scripts`. Dir `NNI_tuning` contains hyper-parameter searching configurations.

## Development Environment
All development environment dependencies are listed in file `conda_env_dependencies.yml`. You can quickly create a
usable python environment with an anaconda command `conda env create -f conda_env_dependencies.yml`. Note that this file is for Linux platform.

## Datasets
Please check out sub-directory `dataset_generating_scripts`.

## Quick Run
For example,

```r
# for training
CUDA_VISIBLE_DEVICES=2 nohup python run_models.py \
    --config_path configs/PhysioNet2012_SAITS_best.ini \
    > NIPS_results/PhysioNet2012_SAITS_best.out &

# for testing
CUDA_VISIBLE_DEVICES=3 python run_models.py \
    --config_path configs/PhysioNet2012_SAITS_best.ini \
    --test_mode
```

Note that paths of datasets and saving dirs may be different on personal computers, please check. 
