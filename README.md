<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="https://raw.githubusercontent.com/WenjieDu/SAITS/main/figs/SAITS full title.jpg" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-v3-yellowgreen" />
    <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-❤️-bbcdc5?logo=pytorch&logoColor=white">
    <img src="https://img.shields.io/badge/Conda-Supported-lightgreen?style=social&logo=anaconda" />
    <img src="https://img.shields.io/badge/License-GPL--v3-lightgreen" />
    <a href="https://arxiv.org/abs/2202.08516">
        <img src="https://img.shields.io/badge/arXiv-preprint-success" />
    </a>
    <a href="https://doi.org/10.1016/j.eswa.2023.119619">
        <img src="https://img.shields.io/badge/ESWA-accepted-255674" />
    </a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FSAITS&count_bg=%23009A0A&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits&edge_flat=false" />
</p>

The official code repository for the paper [SAITS: Self-Attention-based Imputation for Time Series](https://doi.org/10.1016/j.eswa.2023.119619) 
(preprint on arXiv is [here](https://arxiv.org/abs/2202.08516)), which has been accepted by the journal
*[Expert Systems With Applications (ESWA)](https://www.sciencedirect.com/journal/expert-systems-with-applications)*
[2022 IF 8.665, CiteScore 12.2, JCR-Q1, CAS-Q1 (中科院-1区), CCF-C]. You may never hear of ESWA, 
while this journal was ranked 1st in Google Scholar under the top publications of Artificial Intelligence in 2016 
([info source](https://www.sciencedirect.com/journal/expert-systems-with-applications/about/news#expert-systems-with-applications-is-currently-ranked-no1-in)), 
and [here](https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence) is the current ranking list for your information. 

Please star🌟 this repo to help others notice SAITS if you think it is useful. It really means a lot to my open-source work. Thank you! 

**‼️Kind reminder: This document can <ins>help you solve many common questions</ins>, please read it before you run the code 😊**

> <a href='https://github.com/WenjieDu/PyPOTS'><img src='https://raw.githubusercontent.com/WenjieDu/PyPOTS/main/docs/_static/figs/PyPOTS_logo.svg?sanitize=true' width='120' align='right' /></a> 📣 Attention please: <br>
> SAITS now is available in [PyPOTS](https://github.com/WenjieDu/PyPOTS), a Python toolbox for data mining on POTS (Partially-Observed Time Series). An example of training SAITS for imputing dataset PhysioNet-2012 is shown below. With [PyPOTS](https://github.com/WenjieDu/PyPOTS), easy peasy! 😉 

<details>
  <summary><b>👉 Click here to see the example 👀</b></summary>

``` python
# Install PyPOTS first: pip install pypots>=0.0.10
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae
# Data preprocessing. Tedious, but PyPOTS can help. 🤓
data = load_specific_dataset('physionet_2012')  # For datasets in PyPOTS database, PyPOTS will automatically download and extract it.
X = data['X']
num_samples = len(X['RecordID'].unique())
X = X.drop('RecordID', axis = 1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
X = masked_fill(X, 1 - missing_mask, np.nan)
# Model training. This is PyPOTS showtime. 💪
saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=10)
dataset = {"X": X}
saits.fit(dataset)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
imputation = saits.impute(dataset)  # impute the originally-missing values and artificially-missing values
mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
```

</details>

⦿ **`Motivation`**: SAITS is developed primarily to help overcome the drawbacks (slow speed, memory constraints, and compounding error)
of RNN-based imputation models and to obtain the state-of-the-art (SOTA) imputation accuracy on partially-observed time series.

⦿ **`Performance`**: SAITS outperforms [BRITS](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html)
by **12% ∼ 38%** in MAE (mean absolute error) and achieves **2.0 ∼ 2.6** times faster training speed.
Furthermore, SAITS outperforms Transformer (trained by our joint-optimization approach) by **2% ∼ 19%** in MAE with a
more efficient model structure (to obtain comparable performance, SAITS needs only **15% ∼ 30%** parameters of Transformer).
Compared to another SOTA self-attention imputation model [NRTSI](https://github.com/lupalab/NRTSI), SAITS achieves
**7% ∼ 39%** smaller mean squared error (<ins>above 20% in nine out of sixteen cases</ins>), meanwhile, needs much
fewer parameters and less imputation time in practice. 
Please refer to our [full paper](https://arxiv.org/pdf/2202.08516.pdf) for more details about SAITS' performance.

## ❖ Brief Graphical Illustration of Our Methodology
Here we only show the two main components of our method: the joint-optimization training approach and SAITS structure.
For the detailed description and explanation, please read our [full paper](https://arxiv.org/pdf/2202.08516.pdf).

<!-- <details>
  <summary><b>👉 Click here to see the figures of our model training approach and SAITS structure 👀</b></summary> -->

<div align="center">
<img src="https://raw.githubusercontent.com/WenjieDu/SAITS/main/figs/Training approach.svg?sanitize=true" alt="Training approach" title="Training approach" width="800"/>

<b>Fig. 1: Training approach</b>

<img src="https://raw.githubusercontent.com/WenjieDu/SAITS/main/figs/SAITS arch.svg?sanitize=true" alt="SAITS architecture" title="SAITS architecture" width="600"/>

<b>Fig. 2: SAITS structure</b>

</div>
<!-- </details> -->

## ❖ Repository Structure
The implementation of SAITS is in dir [`modeling`](https://github.com/WenjieDu/SAITS/blob/main/modeling/SA_models.py).
We give configurations of our models in dir [`configs`](https://github.com/WenjieDu/SAITS/tree/main/configs), provide
the dataset links and preprocessing scripts in dir [`dataset_generating_scripts`](https://github.com/WenjieDu/SAITS/tree/main/dataset_generating_scripts).
Dir [`NNI_tuning`](https://github.com/WenjieDu/SAITS/tree/main/NNI_tuning) contains the hyper-parameter searching configurations.

## ❖ Development Environment
All dependencies of our development environment are listed in file [`conda_env_dependencies.yml`](https://github.com/WenjieDu/SAITS/blob/main/conda_env_dependencies.yml).
You can quickly create a usable python environment with an anaconda command `conda env create -f conda_env_dependencies.yml`.

## ❖ Datasets
For datasets downloading and generating, please check out the scripts in 
dir [`dataset_generating_scripts`](https://github.com/WenjieDu/SAITS/tree/main/dataset_generating_scripts).

## ❖ Quick Run
Generate the dataset you need first. To do so, please check out the generating scripts in 
dir [`dataset_generating_scripts`](https://github.com/WenjieDu/SAITS/tree/main/dataset_generating_scripts).

After data generation, train and test your model, for example,

```bash
# create a dir to save logs and results
mkdir NIPS_results

# train a model
nohup python run_models.py \
    --config_path configs/PhysioNet2012_SAITS_best.ini \
    > NIPS_results/PhysioNet2012_SAITS_best.out &

# during training, you can run the blow command to read the training log
less NIPS_results/PhysioNet2012_SAITS_best.out

# after training, pick the best model and modify the path of the model for testing in the config file, then run the below command to test the model
python run_models.py \
    --config_path configs/PhysioNet2012_SAITS_best.ini \
    --test_mode
```

❗️Note that paths of datasets and saving dirs may be different on personal computers, please check them in the configuration files.

## ❖ Reference
If you find SAITS is helpful to your research, please cite our paper as below, 
⭐️star this repository, and recommend it to others who you think may need it. 🤗

```bibtex
@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}
```

or

`Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023.`

## ❖ Acknowledgments
Thanks to Mitacs and NSERC (Natural Sciences and Engineering Research Council of Canada) for funding support.
Thanks to Ciena for providing computing resources.
Thanks to all our reviewers for helping improve the quality of this paper.
And thank you all for your attention to this work! ❤️

### ✨Stars/forks/issues/PRs are all welcome!

<details open>
<summary><b><i>👏 Click to View Stargazers and Forkers: </i></b></summary>

[![Stargazers repo roster for @WenjieDu/SAITS](https://reporoster.com/stars/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/stargazers)

[![Forkers repo roster for @WenjieDu/SAITS](https://reporoster.com/forks/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/network/members)
</details>

## ❖ Last but Not Least
If you have any additional questions or have interests in collaboration, 
please take a look at [my GitHub profile](https://github.com/WenjieDu) and feel free to contact me😃.