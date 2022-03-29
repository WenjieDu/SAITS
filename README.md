# SAITS
![](https://img.shields.io/badge/Python-3.8-yellowgreen)
![](https://img.shields.io/badge/PyTorch-1.8.1-green)
![](https://img.shields.io/badge/Conda-Supported-lightgreen?style=social&logo=anaconda)
![](https://img.shields.io/badge/License-GPL3-lightgreen)
[![](https://img.shields.io/badge/Paper-arXiv_preprint-success)](https://arxiv.org/abs/2202.08516)
![](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FSAITS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false)

The official code repository for paper *[SAITS: Self-Attention-based Imputation for Time Series](https://arxiv.org/abs/2202.08516)*. 

⦿ **`Motivation`**: SAITS is developed primarily to help overcome the drawbacks (slow speed, memory constraints, and compounding error) of RNN-based imputation models and to obtain the state-of-the-art (SOTA) imputation accuracy on partially-observed time series.

⦿ **`Performance`**: SAITS outperforms [BRITS](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html) by **12% ∼ 38%** in MAE (mean absolute error) and achieves **2.0 ∼ 2.6** times faster training speed. Furthermore, SAITS outperforms Transformer (trained by our joint-optimization approach) by **2% ∼ 13%** in MAE with a more efficient model structure (to obtain comparable performance, SAITS needs only **15% ∼ 30%** parameters of Transformer). Compared to another SOTA self-attention imputation model [NRTSI](https://github.com/lupalab/NRTSI), SAITS achieves **7% ∼ 39%** smaller mean squared error (<ins>*above 20% in nine out of sixteen cases*</ins>), meanwhile, needs much fewer parameters and less imputation time in practice. Please refer to our [full paper](https://arxiv.org/pdf/2202.08516.pdf) for more details about SAITS' performance.

## ❖ Repository Structure
The implementation of SAITS is in dir [`modeling`](https://github.com/WenjieDu/SAITS/blob/master/modeling/SA_models.py). We give configurations of our models in dir [`configs`](https://github.com/WenjieDu/SAITS/tree/master/configs), provide the dataset links and preprocessing scripts in dir [`dataset_generating_scripts`](https://github.com/WenjieDu/SAITS/tree/master/dataset_generating_scripts). Dir [`NNI_tuning`](https://github.com/WenjieDu/SAITS/tree/master/NNI_tuning) contains the hyper-parameter searching configurations.

## ❖ Implemented Models
The implemented models in dir [`modeling`](https://github.com/WenjieDu/SAITS/blob/master/modeling) are listed below:

* [MRNN](https://ieeexplore.ieee.org/document/8485748) (in [`modeling/mrnn.py`](https://github.com/WenjieDu/SAITS/blob/master/modeling/mrnn.py#L44))
* [BRITS](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html) (in [`modeling/brits.py`](https://github.com/WenjieDu/SAITS/blob/master/modeling/brits.py#L151))
* [Transformer](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) (in [`modeling/SA_models.py#L28`](https://github.com/WenjieDu/SAITS/blob/master/modeling/SA_models.py#L28))
* [SAITS](https://arxiv.org/abs/2202.08516) (in [`modeling/SA_models.py#L93`](https://github.com/WenjieDu/SAITS/blob/master/modeling/SA_models.py#L93))

For other baseline models used in the paper, please refer to their GitHub open-source repositories given in their original papers (the links also available in our paper).

## ❖ Development Environment
All dependencies of our development environment are listed in file [`conda_env_dependencies.yml`](https://github.com/WenjieDu/SAITS/blob/master/conda_env_dependencies.yml). You can quickly create a
usable python environment with an anaconda command `conda env create -f conda_env_dependencies.yml`. **❗️Note that this file is for Linux platform,** but you still can use it for reference of dependency libraries.

## ❖ Datasets
For datasets downloading and generating, please check out the scripts in dir [`dataset_generating_scripts`](https://github.com/WenjieDu/SAITS/tree/master/dataset_generating_scripts).

## ❖ Quick Run
For example,

```bash
# for training
CUDA_VISIBLE_DEVICES=2 nohup python run_models.py \
    --config_path configs/PhysioNet2012_SAITS_best.ini \
    > NIPS_results/PhysioNet2012_SAITS_best.out &

# for testing
CUDA_VISIBLE_DEVICES=3 python run_models.py \
    --config_path configs/PhysioNet2012_SAITS_best.ini \
    --test_mode
```

Note that paths of datasets and saving dirs may be different on personal computers, please check them in the configuration files.

## ❖ Reference
If you use this model or the code in this repository, please cite our paper 🤗

```bibtex
@article{Du2022SAITS,
      title={{SAITS: Self-Attention-based Imputation for Time Series}}, 
      author={Wenjie Du and David Côté and Yan Liu},
      year={2022},
      eprint={2202.08516},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

or

`Wenjie Du, David Côté, and Yan Liu. "SAITS: Self-Attention-based Imputation for Time Series." ArXiv abs/2202.08516`

## ❖ Acknowledgments
Thanks to Mitacs and NSERC (Natural Sciences and Engineering Research Council of Canada) for funding support. Thanks to Ciena for providing computing resources.

<details open>
<summary><b><i>👏 Click to View Stargazers and Forkers: </i></b></summary>

[![Stargazers repo roster for @WenjieDu/SAITS](https://reporoster.com/stars/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/stargazers)
    
[![Forkers repo roster for @WenjieDu/SAITS](https://reporoster.com/forks/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/network/members)
</details>

✨Stars, forks, issues, and PRs are all welcome! If you have any other questions, please [drop me an email](mailto:wenjay.du@gmail.com) at any time.