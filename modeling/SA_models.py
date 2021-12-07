import numpy as np

from modeling.layers import *
from modeling.utils import masked_mae_cal


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 **kwargs):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs['input_with_mask']
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs['param_sharing_strategy']
        self.MIT = kwargs['MIT']
        self.device = kwargs['device']

        if kwargs['param_sharing_strategy'] == 'between_group':
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
                for _ in range(n_groups)
            ])

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        input_X = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = masks * X + (1 - masks) * learned_presentation  # replace non-missing part with original data
        return imputed_data, learned_presentation

    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(learned_presentation, X, masks)
        if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(learned_presentation, inputs['X_holdout'], inputs['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)

        return {'imputed_data': imputed_data,
                'reconstruction_loss': reconstruction_MAE, 'imputation_loss': imputation_MAE,
                'reconstruction_MAE': reconstruction_MAE, 'imputation_MAE': imputation_MAE}


class SAITS(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 **kwargs):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs['input_with_mask']
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs['param_sharing_strategy']
        self.MIT = kwargs['MIT']
        self.device = kwargs['device']

        if kwargs['param_sharing_strategy'] == 'between_group':
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_groups)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_groups)
            ])

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(self.position_enc(input_X_for_first))  # namely term e in math algo
        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze()  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(self.weight_combine(torch.cat([masks, attn_weights], dim=2)))  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)
        reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(X_tilde_3, inputs['X_holdout'], inputs['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)

        return {'imputed_data': imputed_data,
                'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_MAE,
                'reconstruction_MAE': final_reconstruction_MAE, 'imputation_MAE': imputation_MAE}
