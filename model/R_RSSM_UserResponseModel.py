from model.general import BaseModel
from model.components import DNN
import torch
import torch.nn as nn

import torch.nn.functional as F

from model.retnet import RetNet

class NormalLinear(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features) -> None:
        super().__init__()
        self.act = F.relu

        self.fc = nn.Linear(in_features, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, out_features)
        self.var = nn.Linear(hidden_dim, out_features)

    def forward(self, input) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.fc(input))
        h = self.act(self.fc1(x))
        mu = self.mu(h)
        var = F.softplus(self.var(h)) + 1e-1
        return mu, var
    
# class ClickPrediction(nn.Module):
#     def __init__(self, input_dim):
#         super(ClickPrediction, self).__init__()
#         self.fc = nn.Linear(input_dim, 1)  # 线性层
#         self.sigmoid = nn.Sigmoid()  # sigmoid激活函数
        
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.sigmoid(x)
#         return x

class ClickPrediction(nn.Module):
    def __init__(self, input_dim):
        super(ClickPrediction, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)  # 第一个线性层
        self.fc2 = nn.Linear(128, 64)        # 第二个线性层
        self.fc3 = nn.Linear(64, 1)          # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 使用sigmoid函数输出概率
        return x

class Restruction(nn.Module):
    def __init__(self, input_dim):
        super(Restruction, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)  # 第一个线性层
        self.fc2 = nn.Linear(128, 64)        # 第二个线性层
        self.fc3 = nn.Linear(64, 32)          # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 使用sigmoid函数输出概率
        return x


class R_RSSM_UserResponseModel(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - feature_dim
        - attn_n_head
        - hidden_dims
        - dropout_rate
        - batch_norm
        - from BaseModel:
            - model_path
            - loss
            - l2_coef
        '''
        parser = BaseModel.parse_model_args(parser)
        
        parser.add_argument('--feature_dim', type=int, default=32, 
                            help='dimension size for all features')
        parser.add_argument('--attn_n_head', type=int, default=4, 
                            help='dimension size for all features')
        parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--dropout_rate', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        return parser
    
    def log(self):
        super().log()
        print("\tencoding_dim = " + str(self.feature_dim))
        print("\titem_input_dim = " + str(self.item_dim))
        print("\tuser_input_dim = " + str(self.portrait_len))
        
    def __init__(self, args, device):
        super().__init__(args, device)
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        
    def _define_params(self, args):
        # stats = reader.get_statistics()
        self.portrait_len = 42#stats['user_portrait_len']
        self.item_dim = 40#stats['item_vec_size']
        self.feature_dim = args.feature_dim
        # print("hidden_dims = " ,args.hidden_dims)
        # portrait embedding
        self.portrait_encoding_layer = DNN(self.portrait_len, args.hidden_dims, args.feature_dim, 
                                           dropout_rate = args.dropout_rate, do_batch_norm = False)
        # item_emb
        self.item_emb_layer = nn.Linear(self.item_dim, args.feature_dim)
        # user history encoder
        self.seq_self_attn_layer = nn.MultiheadAttention(args.feature_dim, args.attn_n_head, batch_first = True)
        self.seq_user_attn_layer = nn.MultiheadAttention(args.feature_dim, args.attn_n_head, batch_first = True)

        # self.seq_self_attn_layer = RetNet(4 ,args.feature_dim, , args.attn_n_head, batch_first = True)
        # self.seq_user_attn_layer = RetNet(4 ,args.feature_dim, , args.attn_n_head, batch_first = True)

        self.layer = 3
        self.ret_seq = RetNet(self.layer ,args.feature_dim*2, args.hidden_dims[0], args.attn_n_head, double_v_dim= False)
        self.ret_nn = nn.Linear(args.feature_dim*3, args.feature_dim)
        self.posterior_latent = NormalLinear(args.feature_dim*2, args.hidden_dims[0], args.feature_dim)
        self.click = ClickPrediction(args.feature_dim*3)
    
    def get_forward(self, feed_dict: dict) -> dict:
        # user embedding (B,1,f_dim)
        user_emb = self.portrait_encoding_layer(feed_dict['user_profile']).view(-1,1,self.feature_dim) 
        # print({'user_emb': user_emb.shape})

        # history embedding (B,H,f_dim)
        history_item_emb = self.item_emb_layer(feed_dict['history_features'])
        # print({'history_item_emb': history_item_emb.shape})

        # sequence self attention, encoded sequence is (B,H,f_dim)
        seq_encoding, attn_weight = self.seq_self_attn_layer(history_item_emb, history_item_emb, history_item_emb)
        # print({'seq_encoding': seq_encoding.shape, 'attn_weight': attn_weight.shape})

        # rec item embedding (B,L,f_dim)
        exposure_item_emb = self.item_emb_layer(feed_dict['exposure_features'])
        # print({'exposure_item_emb': exposure_item_emb.shape})        

        seq_encoding = torch.cat([seq_encoding, exposure_item_emb], dim = 1)

        # cross attention, encoded history is (B,1,f_dim)

        user_and_seq = torch.cat([seq_encoding, user_emb.repeat(1,59,1)], dim = 2)

        y_n= self.ret_seq.forward(user_and_seq)
        d_state = y_n[:, -9:, :]
        click_list = []
        for i in range(0,9):
            tmp_d_state = d_state[:, i:i+1, :]
            # tmp_d_state = tmp_d_state[:,-1]
            tmp_d_state = tmp_d_state.squeeze(1)
            mu, var = self.posterior_latent(tmp_d_state)
            normal_dist = torch.distributions.Normal(mu, var)
            s_state = normal_dist.sample()
            # print({'tmp_d_state': tmp_d_state.shape,'s_state': s_state.shape})
            tmp_click = self.click(torch.cat([tmp_d_state, s_state],dim=1))
            click_list.append(tmp_click)

        score = torch.stack(click_list, dim=1).squeeze(-1)

        # score = torch.sum(exposure_item_emb * user_interest, dim = -1)
        # print({'score': score.shape})
        
        # regularization terms
        reg = self.get_regularization(self.portrait_encoding_layer, self.item_emb_layer, 
                                      self.posterior_latent, self.click,
                                      self.ret_seq)
        # print({'reg': reg.shape})

        # print({'preds': score.shape, 'reg': reg.shape})
        return {'preds': score, 'reg': reg}
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        """
        @input:
        - feed_dict: {...}
        - out_dict: {"preds":, "reg":}
        
        Loss terms implemented:
        - BCE
        """
        
        preds, reg = out_dict["preds"].view(-1), out_dict["reg"] # (B,L), scalar
        target = feed_dict['feedback'].view(-1).to(torch.float) # (B,L)
        # loss
        loss = torch.mean(self.bce_loss(self.sigmoid(preds), target))
        loss = loss + self.l2_coef * reg
        return loss
    
