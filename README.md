<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="https://raw.githubusercontent.com/WenjieDu/SAITS/master/figs/SAITS full title.svg?sanitize=true" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-v3.8-yellowgreen" />
  <img src="https://img.shields.io/badge/PyTorch-1.8.1-green" />
  <img src="https://img.shields.io/badge/Conda-Supported-lightgreen?style=social&logo=anaconda" />
  <img src="https://img.shields.io/badge/License-GPL--v3-lightgreen" />
  <a href="https://arxiv.org/abs/2202.08516">
    <img src="https://img.shields.io/badge/Paper-arXiv_preprint-success" />
  </a>
  <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FSAITS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false" />
</p>

The official code repository for paper *[SAITS: Self-Attention-based Imputation for Time Series](https://arxiv.org/abs/2202.08516)*. 

> 📣 Attention please‼️ <br>
> SAITS now is available in [PyPOTS](https://github.com/WenjieDu/PyPOTS), a Python toolbox born for data mining on partially-observed time series (POTS). An example of training SAITS for imputing dataset PhysioNet-2012 is shown below. With [PyPOTS](https://github.com/WenjieDu/PyPOTS), easy peasy! 😉

``` python
# Install PyPOTS first: pip install pypots 

from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, fill_nan_with_mask
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae
# Data preprocessing. Tedious, but PyPOTS can help. 🤓
X = load_specific_dataset('physionet_2012')['X']  # For datasets in PyPOTS database, PyPOTS will automatically download and extract it.
num_samples=len(X['RecordID'].unique())
X = X.drop('RecordID', axis=1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
X = fill_nan_with_mask(X, missing_mask)
# Model training. This is PyPOTS showtime. 💪
saits_base = SAITS(seq_len=48, n_features=37, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
saits_base.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
imputation = saits_base.impute(X)  # impute the originally-missing values and artificially-missing values
mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
```

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
Generate the dataset you need first. To do so, please check out the generating scripts in dir [`dataset_generating_scripts`](https://github.com/WenjieDu/SAITS/tree/master/dataset_generating_scripts).

After data generation, train and test your model, for example,

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

❗️Note that paths of datasets and saving dirs may be different on personal computers, please check them in the configuration files.

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
Thanks to Mitacs and NSERC (Natural Sciences and Engineering Research Council of Canada) for funding support. Thanks to Ciena for providing computing resources. Thanks to all reviewers for helping improve the quality of this paper. And thank you all for your attention to this work!

<details open>
<summary><b><i>👏 Click to View Stargazers and Forkers: </i></b></summary>

[![Stargazers repo roster for @WenjieDu/SAITS](https://reporoster.com/stars/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/stargazers)
    
[![Forkers repo roster for @WenjieDu/SAITS](https://reporoster.com/forks/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/network/members)
</details>

✨Stars, forks, issues, and PRs are all welcome! If you have any other questions, please [drop me an email](mailto:wenjay.du@gmail.com) at any time.