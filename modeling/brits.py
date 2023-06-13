"""
Our implementation of BRITS model for time-series imputation.

If you use code in this repository, please cite our paper as below. Many thanks.

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

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from modeling.utils import masked_mae_cal


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(RITS, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # other hyper parameters
        self.device = kwargs["device"]
        self.MIT = kwargs["MIT"]

        # create models
        self.rnn_cell = nn.LSTMCell(self.feature_num * 2, self.rnn_hidden_size)
        # # Temporal Decay here is used to decay the hidden state
        self.temp_decay_h = TemporalDecay(
            input_size=self.feature_num, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.feature_num, output_size=self.feature_num, diag=True
        )
        # # History regression and feature regression layer
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.feature_num)
        self.feat_reg = FeatureRegression(self.feature_num)
        # # weight-combine is used to combine history regression and feature regression
        self.weight_combine = nn.Linear(self.feature_num * 2, self.feature_num)

    def impute(self, data, direction):
        values = data[direction]["X"]
        masks = data[direction]["missing_mask"]
        deltas = data[direction]["deltas"]

        # use device of input values
        hidden_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )
        cell_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        estimations = []
        reconstruction_loss = 0.0
        reconstruction_MAE = 0.0

        # imputation period
        for t in range(self.seq_len):
            # for data, [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += masked_mae_cal(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += masked_mae_cal(z_h, x, m)

            alpha = F.sigmoid(self.weight_combine(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_MAE += masked_mae_cal(c_h, x, m)
            reconstruction_loss += reconstruction_MAE

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(
                inputs, (hidden_states, cell_states)
            )

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, [reconstruction_MAE, reconstruction_loss]

    def forward(self, data, direction="forward"):
        imputed_data, [reconstruction_MAE, reconstruction_loss] = self.impute(
            data, direction
        )
        reconstruction_MAE /= self.seq_len
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.seq_len * 3

        ret_dict = {
            "consistency_loss": torch.tensor(
                0.0, device=self.device
            ),  # single direction, has no consistency loss
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_MAE": reconstruction_MAE,
            "imputed_data": imputed_data,
        }
        if "X_holdout" in data:
            ret_dict["X_holdout"] = data["X_holdout"]
            ret_dict["indicating_mask"] = data["indicating_mask"]
        return ret_dict


class BRITS(nn.Module):
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(BRITS, self).__init__()
        self.MIT = kwargs["MIT"]
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(seq_len, feature_num, rnn_hidden_size, **kwargs)
        self.rits_b = RITS(seq_len, feature_num, rnn_hidden_size, **kwargs)

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(
                indices, dtype=torch.long, device=tensor_.device, requires_grad=False
            )
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def merge_ret(self, ret_f, ret_b, stage):
        consistency_loss = self.get_consistency_loss(
            ret_f["imputed_data"], ret_b["imputed_data"]
        )
        imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2
        reconstruction_loss = (
            ret_f["reconstruction_loss"] + ret_b["reconstruction_loss"]
        ) / 2
        reconstruction_MAE = (
            ret_f["reconstruction_MAE"] + ret_b["reconstruction_MAE"]
        ) / 2
        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                imputed_data, ret_f["X_holdout"], ret_f["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)
        imputation_loss = imputation_MAE

        ret_f["imputed_data"] = imputed_data
        ret_f["consistency_loss"] = consistency_loss
        ret_f["reconstruction_loss"] = reconstruction_loss
        ret_f["reconstruction_MAE"] = reconstruction_MAE
        ret_f["imputation_MAE"] = imputation_MAE
        ret_f["imputation_loss"] = imputation_loss
        return ret_f

    def impute(self, data):
        imputed_data_f, _ = self.rits_f.impute(data, "forward")
        imputed_data_b, _ = self.rits_b.impute(data, "backward")
        imputed_data_b = {"imputed_data_b": imputed_data_b}
        imputed_data_b = self.reverse(imputed_data_b)["imputed_data_b"]
        imputed_data = (imputed_data_f + imputed_data_b) / 2
        return imputed_data, [imputed_data_f, imputed_data_b]

    def forward(self, data, stage):
        ret_f = self.rits_f(data, "forward")
        ret_b = self.reverse(self.rits_b(data, "backward"))
        ret = self.merge_ret(ret_f, ret_b, stage)
        return ret
