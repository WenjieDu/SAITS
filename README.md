
> [!TIP]
> **[Updates in May 2025]** 🎉 Our survey paper [Deep Learning for Multivariate Time Series Imputation: A Survey](https://arxiv.org/abs/2402.04059) gets accepted by IJCAI 2025!
We comprehensively review the literature of the state-of-the-art deep-learning imputation methods for time series, provide a new taxonomy based on uncertainty and model architecture for them, systematically compare multiple toolkits, and discuss the challenges and future directions in this field.
> 
> **[Updates in Jun 2024]** 😎 The 1st comprehensive time-seres imputation benchmark paper
[TSI-Bench: Benchmarking Time Series Imputation](https://arxiv.org/abs/2406.12747) is now publicly available.
The code is open source in the repo [Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation).
With nearly 35,000 experiments, we provide a comprehensive benchmarking study on 28 imputation methods, 3 missing patterns (points, sequences, blocks),
various missing rates, and 8 real-world datasets.
> 
> **[Updates in May 2024]** 🔥 We applied SAITS embedding and training strategies to **iTransformer, FiLM, FreTS, Crossformer, PatchTST, DLinear, ETSformer, FEDformer, 
> Informer, Autoformer, Non-stationary Transformer, Pyraformer, Reformer, SCINet, RevIN, Koopa, MICN, TiDE, and StemGNN** in <a href="https://github.com/WenjieDu/PyPOTS"><img src="https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg" width="26px" align="center"/> PyPOTS</a>
> to enable them to be applicable to the time-series imputation task.


<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="https://pypots.com/figs/pypots_logos/SAITS/banner.jpg" alt="SAITS Title" title="SAITS Title" width="80%"/>
    </a>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-v3-E97040?logo=python&logoColor=white" />
    <img alt="powered by PyTorch" src="https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white">
    <a href="https://github.com/WenjieDu/SAITS/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-3C7699?logo=opensourceinitiative&logoColor=white" />
    </a>
    <a href="https://doi.org/10.1016/j.eswa.2023.119619">
        <img src="https://img.shields.io/badge/ESWA-published-75C1C4?logo=elsevier&color=FF6C00" />
    </a>
    <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=j9qvUg0AAAAJ&citation_for_view=j9qvUg0AAAAJ:Y0pCki6q_DkC" title="Paper citation number from Google Scholar">
        <img src="https://pypots.com/figs/citation_badges/saits.svg" />
    </a>
    <a href="https://webofscience.clarivate.cn/wos/woscc/full-record/WOS:000943170100001?SID=USW2EC0D82x89d30RifxLVxJpho5Y" title="This is a Highly Cited Paper recognized by ESI">
        <img src="https://pypots.com/figs/citation_badges/ESI_highly_cited_paper.svg" />
    </a>
</p>


**‼️Kind reminder: This document can <ins>help you solve many common questions</ins>, please read it before you run the code.**

The official code repository is for the paper [SAITS: Self-Attention-based Imputation for Time Series](https://doi.org/10.1016/j.eswa.2023.119619) 
(preprint on arXiv is [here](https://arxiv.org/abs/2202.08516)), which has been accepted by the journal
*[Expert Systems with Applications (ESWA)](https://www.sciencedirect.com/journal/expert-systems-with-applications)*
[2022 IF 8.665, CiteScore 12.2, JCR-Q1, CAS-Q1, CCF-C]. You may never have heard of ESWA, 
while it was ranked 1st in Google Scholar under the top publications of Artificial Intelligence in 2016 
([info source](https://www.sciencedirect.com/journal/expert-systems-with-applications/about/news#expert-systems-with-applications-is-currently-ranked-no1-in)), and is still the top 1 AI journal according to Google Scholar metrics 
([here is the current ranking list](https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence) FYI).

SAITS is the first work applying pure self-attention without any recursive design in the algorithm for general time series imputation.
Basically you can take it as a validated framework for time series imputation, like we've integrated more than 2️⃣0️⃣ forecasting models into PyPOTS by adapting SAITS framework.
More generally, you can use it for sequence imputation. Besides, the code here is open source under the MIT license. 
Therefore, you're welcome to modify the SAITS code for your own research purpose and domain applications.
Of course, it probably needs a bit of modification in the model structure or loss functions for specific scenarios or data input.
And this is [an incomplete list](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&as_ylo=2022&q=%E2%80%9CSAITS%E2%80%9D+%22time+series%22) of scientific research referencing SAITS in their papers. 

🤗 Please [cite SAITS](https://github.com/WenjieDu/SAITS#-citing-saits) in your publications if it helps with your work.
Please star🌟 this repo to help others notice SAITS if you think it is useful. 
It really means a lot to our open-source research. Thank you! 
BTW, you may also like 
<a href="https://github.com/WenjieDu/PyPOTS">
PyPOTS <img align="center" src="https://img.shields.io/github/stars/WenjieDu/PyPOTS?style=social">
</a>
for easily modeling your partially-observed time-series datasets.

> [!IMPORTANT]
> <a href='https://github.com/WenjieDu/PyPOTS'><img src='https://pypots.com/figs/pypots_logos/PyPOTS/logo_FFBG.svg' width='130' align='right' /></a>
> **📣 Attention please:**
> 
> SAITS now is available in [PyPOTS](https://github.com/WenjieDu/PyPOTS), a Python toolbox for data mining on POTS (Partially-Observed Time Series). 
> An example of training SAITS for imputing dataset PhysioNet-2012 is shown below. With [PyPOTS](https://github.com/WenjieDu/PyPOTS), easy peasy! 😉 

<details open>
  <summary><b>👉 Click here to see the example 👀</b></summary>

``` python
import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, calc_missing_rate
from benchpots.datasets import preprocess_physionet2012
data = preprocess_physionet2012(subset='set-a',rate=0.1) # Our ecosystem libs will automatically download and extract it
train_X, val_X, test_X = data["train_X"], data["val_X"], data["test_X"]
print(train_X.shape)  # (n_samples, n_steps, n_features)
print(val_X.shape)  # samples (n_samples) in train set and val set are different, but they have the same sequence len (n_steps) and feature dim (n_features)
print(f"We have {calc_missing_rate(train_X):.1%} values missing in train_X")  
train_set = {"X": train_X}  # in training set, simply put the incomplete time series into it
val_set = {
    "X": val_X,
    "X_ori": data["val_X_ori"],  # in validation set, we need ground truth for evaluation and picking the best model checkpoint
}
test_set = {"X": test_X}  # in test set, only give the testing incomplete time series for model to impute
test_X_ori = data["test_X_ori"]  # test_X_ori bears ground truth for evaluation
indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)  # mask indicates the values that are missing in X but not in X_ori, i.e. where the gt values are 

from pypots.imputation import SAITS  # import the model you want to use
from pypots.nn.functional import calc_mae
saits = SAITS(n_steps=train_X.shape[1], n_features=train_X.shape[2], n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1, epochs=5)
saits.fit(train_set, val_set)  # train the model on the dataset
imputation = saits.impute(test_set)  # impute the originally-missing values and artificially-missing values
mae = calc_mae(imputation, np.nan_to_num(test_X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
saits.save("save_it_here/saits_physionet2012.pypots")  # save the model for future use
saits.load("save_it_here/saits_physionet2012.pypots")  # reload the serialized model file for following imputation or training
```

<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%"/>
</a>
<br>
<b> ☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!</b>
</p>

</details>


## ❖ Motivation and Performance
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
For the detailed description and explanation, please read our full paper `Paper_SAITS.pdf` in this repo 
or [on arXiv](https://arxiv.org/pdf/2202.08516.pdf).

<div align="center">
<img src="https://raw.githubusercontent.com/WenjieDu/SAITS/main/figs/Training approach.svg?sanitize=true" alt="Training approach" title="Training approach" width="800"/>

<b>Fig. 1: Training approach</b>

<img src="https://raw.githubusercontent.com/WenjieDu/SAITS/main/figs/SAITS arch.svg?sanitize=true" alt="SAITS architecture" title="SAITS architecture" width="600"/>

<b>Fig. 2: SAITS structure</b>

</div>


## ❖ Citing SAITS
If you find SAITS is helpful to your work, please cite our paper as below, 
⭐️star this repository, and recommend it to others who you think may need it. 🤗 Thank you!

```bibtex
@article{du2023saits,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {10.1016/j.eswa.2023.119619},
url = {https://arxiv.org/abs/2202.08516},
author = {Wenjie Du and David Cote and Yan Liu},
}
```
or
> Wenjie Du, David Cote, and Yan Liu.
> SAITS: Self-Attention-based Imputation for Time Series.
> Expert Systems with Applications, 219:119619, 2023.

### 😎 Our latest survey and benchmarking research on time-series imputation may also be useful to your work:
```bibtex
@article{wang2025survey,
title={Deep Learning for Multivariate Time Series Imputation: A Survey},
author={Jun Wang and Wenjie Du and Yiyuan Yang and Linglong Qian and Wei Cao and Keli Zhang and Wenjia Wang and Yuxuan Liang and Qingsong Wen},
booktitle = {Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI)},
year={2025}
}
```

```bibtex
@article{du2024tsibench,
title={TSI-Bench: Benchmarking Time Series Imputation},
author={Wenjie Du and Jun Wang and Linglong Qian and Yiyuan Yang and Fanxing Liu and Zepu Wang and Zina Ibrahim and Haoxin Liu and Zhiyuan Zhao and Yingjie Zhou and Wenjia Wang and Kaize Ding and Yuxuan Liang and B. Aditya Prakash and Qingsong Wen},
journal={arXiv preprint arXiv:2406.12747},
year={2024}
}
```

### 🔥 In case you use PyPOTS in your research, please also cite the following paper:

``` bibtex
@article{du2023pypots,
title={{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
author={Wenjie Du},
journal={arXiv preprint arXiv:2305.18811},
year={2023},
}
```
or
> Wenjie Du.
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> arXiv, abs/2305.18811, 2023.


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

```shell
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


## ❖ Acknowledgments
Thanks to Ciena, Mitacs, and NSERC (Natural Sciences and Engineering Research Council of Canada) for funding support.  
Thanks to all our reviewers for helping improve the quality of this paper.  
Thanks to Ciena for providing computing resources.  
And thank you all for your attention to this work.


### ✨Stars/forks/issues/PRs are all welcome!

<details open>
<summary><b><i>👏 Click to View Stargazers and Forkers: </i></b></summary>

[![Stargazers repo roster for @WenjieDu/SAITS](http://reporoster.com/stars/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/stargazers)

[![Forkers repo roster for @WenjieDu/SAITS](http://reporoster.com/forks/dark/WenjieDu/SAITS)](https://github.com/WenjieDu/SAITS/network/members)
</details>


## ❖ Last but Not Least
If you have any additional questions or have interests in collaboration, 
please take a look at [my GitHub profile](https://github.com/WenjieDu) and feel free to contact me 😃.