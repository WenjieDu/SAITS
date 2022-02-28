# SAITS

The official code repository for paper *[SAITS: Self-Attention-based Imputation for Time Series](https://arxiv.org/abs/2202.08516)*. 

The implementation of SAITS is in dir `modeling`. We give configurations of our models in dir `configs`, provide dataset links and preprocessing scripts in dir `dataset_generating_scripts`. Dir `NNI_tuning` contains hyper-parameter searching configurations.


## Stars⭐️, forks, issues, and PRs are all welcome!
<details open>
<summary><b><i>Click to View Stargazers and Forkers</i></b></summary>

[![Stargazers repo roster for @WenjieDu/SAITS](https://reporoster.com/stars/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/stargazers)
    
[![Forkers repo roster for @WenjieDu/SAITS](https://reporoster.com/forks/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/network/members)
</details>


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


## Reference
If you use this model and love it, please cite our paper 🤗

```
@article{Du2022SAITS,
      title={{SAITS: Self-Attention-based Imputation for Time Series}}, 
      author={Wenjie Du and David Côté and Yan Liu},
      year={2022},
      eprint={2202.08516},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```