import math
import os
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tbsim.models.context_encoders import MapEncoderPtsMA
from tbsim.models.diffuser_helpers import SinusoidalPosEmb

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class SceneTransformer(nn.Module):
    '''
    SceneTransformer Class.
    '''
    def __init__(self, d_k=128, d_edge=32, L_enc=1, dropout=0.0, k_attr=7, neighbor_inds=[0,1,2,3], edge_attr_separation=[], map_attr=2, num_heads=16, L_dec=1, tx_hidden_size=384, use_map_lanes=False, transition_dim=6, output_dim=2, agent_hist_embed_method='transformer', neigh_hist_embed_method=None, map_embed_method='transformer', interaction_edge_speed_repr='rel_vel', single_cond_feat=False, mask_social_interaction=False, mask_edge=False, social_attn_radius=50, all_interactive_social=False, mask_time=True, layer_num_per_edge_decoder=1, attn_combination='sum'):
        '''
        d_edge: applies only when neigh_hist_embed_method is 'edge_interaction'

        k_attr: number of attributes of history trajectories
        neighbor_inds: neighbor indices to be used for edge modeling (default correspondence: x, y, cos, sin)
        edge_attr_separation: if not empty, the edge attributes will be separated into different edge attrs before encoding.
        map_attr: number of points per road segment
        output_dim: output final dimension of the prediction

        tx_hidden_size: hidden size of FNN for each transformer encoder
        use_map_lanes: if map is used
        data_centric: 'agent' or 'scene'. If 'agent', an agent-level mask applies to social attention layer
        
        single_cond_feat: if True, use a single conditional feature consists of map, agent_hist, and neigh_hist.

        all_interactive_social: if True, all social attention layer will involve interactions. If False, only the first social attention layer will involve interactions.
        mask_time: if True, mask future time steps in the social attention layer
        layer_num_per_edge_decoder: number of layers per edge decoder

        attn_combination: how to combine attention and x.
        '''
        super(SceneTransformer, self).__init__()

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.k_attr = k_attr
        self.map_attr = map_attr
        self.neighbor_inds = neighbor_inds
        self.edge_attr_separation = edge_attr_separation
        self.d_k = d_k
        self.d_edge = d_edge
        self.L_enc = L_enc
        self.dropout = dropout
        self.num_heads = num_heads
        self.L_dec = L_dec
        self.tx_hidden_size = tx_hidden_size
        self.use_map_lanes = use_map_lanes

        self.transition_dim = transition_dim
        self.output_dim = output_dim
        self.agent_hist_embed_method = agent_hist_embed_method
        self.neigh_hist_embed_method = neigh_hist_embed_method
        self.map_embed_method = map_embed_method
        self.interaction_edge_speed_repr = interaction_edge_speed_repr
        self.single_cond_feat = single_cond_feat
        self.mask_social_interaction = mask_social_interaction
        self.mask_edge = mask_edge
        self.social_attn_radius = social_attn_radius
        self.all_interactive_social = all_interactive_social
        self.mask_time = mask_time
        self.layer_num_per_edge_decoder = layer_num_per_edge_decoder
        self.attn_combination = attn_combination

        # not encode map when it is already included in cond feat
        if self.single_cond_feat or self.map_embed_method == 'cnn_local_patch':
            self.use_map_lanes = False



        # ============================== Neighbor ENCODERS ==============================
        if len(self.edge_attr_separation) == 0:
            self.neighbor_hist_feat_encoder = nn.Sequential(
                init_(nn.Linear(len(neighbor_inds), self.d_edge)),
                nn.Mish(),
                init_(nn.Linear(self.d_edge, self.d_edge)),
            )
            self.neighbor_fut_feat_encoder = nn.Sequential(
                init_(nn.Linear(len(neighbor_inds), self.d_edge)),
                nn.Mish(),
                init_(nn.Linear(self.d_edge, self.d_edge)),
            )
        else:
            self.neighbor_hist_feat_encoder = MultipleInputEmbedding(in_channels=[len(attrs) for attrs in self.edge_attr_separation], out_channel=self.d_edge)
            self.neighbor_fut_feat_encoder = self.neighbor_hist_feat_encoder

        # ============================== INPUT ENCODERS ==============================
        # agent history input encoder
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        # agent noise input encoder
        input_dim = self.transition_dim
        if self.neigh_hist_embed_method == 'interaction_edge_and_input':
            input_dim += 32
        if self.map_embed_method == 'cnn_local_patch':
            input_dim += 32
        self.agents_noisy_input_encoder = nn.Sequential(init_(nn.Linear(input_dim, self.d_k)))
        # ============================== Time ENCODER ==============================
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.d_k),
            nn.Linear(self.d_k, self.d_k * 4),
            nn.Mish(),
            nn.Linear(self.d_k * 4, self.d_k),
        )
        # ============================== AutoBot-Joint ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=2))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # ============================== MAP ENCODER ==========================
        if self.use_map_lanes:
            self.map_encoder = MapEncoderPtsMA(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
            self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)

        # ============================== AutoBot-Joint DECODER ==============================

        self.self_temporal_attn_decoder_layers = []
        self.social_attn_decoder_layers = []
        self.temporal_attn_decoder_layers = []
        for i in range(self.L_dec):
            tx_self_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
            self.self_temporal_attn_decoder_layers.append(nn.TransformerEncoder(tx_self_encoder_layer, num_layers=2))

            tx_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
            self.temporal_attn_decoder_layers.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))

            # agent-agent coord curstomized social attention
            if self.neigh_hist_embed_method == 'interaction_edge':
                if i == 0 or self.all_interactive_social:
                    tx_social_decoder_layer = SocialDecoderLayer(embed_dim=self.d_k, edge_dim=self.d_edge, num_heads=self.num_heads, dropout=self.dropout, attn_combination=self.attn_combination)
                    self.social_attn_decoder_layers.append(SocialTransformerDecoder(decoder_layer=tx_social_decoder_layer, num_layers=self.layer_num_per_edge_decoder, norm=nn.LayerNorm(self.d_k)))
                else:
                    tx_social_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
                    self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_social_encoder_layer, num_layers=1))
            else: # agent-coord naive self social attention
                tx_social_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout, dim_feedforward=self.tx_hidden_size, norm_first=False)
                self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_social_encoder_layer, num_layers=1))

        # self.temporal_attn_decoder_map_layers = nn.ModuleList(self.temporal_attn_decoder_map_layers)
        self.temporal_attn_decoder_layers = nn.ModuleList(self.temporal_attn_decoder_layers)
        self.social_attn_decoder_layers = nn.ModuleList(self.social_attn_decoder_layers)
        self.self_temporal_attn_decoder_layers = nn.ModuleList(self.self_temporal_attn_decoder_layers)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = nn.Sequential(init_(nn.Linear(self.d_k, self.output_dim)))

        self.train()

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T_obs, B, M, d_k)
        :param agent_masks: (B, T_obs, M)
        :return: (T_obs, B, M, d_k)
        '''        
        T_obs, B, M, _ = agents_emb.shape
        # (B, T_obs, M) -> (B*M, T_obs)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure agent's that don't exist don't throw NaNs.

        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * M, -1)),
                                src_key_padding_mask=agent_masks)
        return agents_temp_emb.view(T_obs, B, M, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T_obs, B, M, d_k)
        :param agent_masks: (B, T_obs, M)
        :return: (T_obs, B, M, d_k)
        '''
        if self.mask_social_interaction:
            M = agents_emb.shape[2]
            agent_interaction_mask = (1-torch.eye(M, device=agents_emb.device)).bool()
        else:
            agent_interaction_mask = None
        T_obs, B, M, _ = agents_emb.shape
        # (T_obs, B, M, d_k) -> (M, B*T_obs, d_k)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(M, B * T_obs, -1)
        
        # Original: agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, M))
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.reshape(-1, M), mask=agent_interaction_mask)
        agents_soc_emb = agents_soc_emb.view(M, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb
    
    def self_temporal_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T_all, B, M, d_k)
        :param agent_masks: (B, T_obs, M)
        :return: (T_all, B, M, d_k)
        '''        
        T_all, B, M, _ = agents_emb.shape
        T_obs = agent_masks.shape[1]
        T_fut = T_all - T_obs
        # (B, T_obs, M) -> (B*M, T_obs)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        # padding with current availability for future steps
        # (B*M, T_obs) -> (B*M) -> (B*M, 1) -> (B*M, T_fut)
        agent_masks_fut = agent_masks[:,-1].unsqueeze(-1).repeat(1,T_fut)
        # (B*M, T_obs), (B*M, T_fut) -> (B*M, T_all)
        agent_masks = torch.cat([agent_masks, agent_masks_fut], dim=-1)

        # Ensure agent's that don't exist don't throw NaNs.
        agent_masks[:, T_obs-1][agent_masks.sum(-1) == T_all] = False  
        
        if self.mask_time:
            agent_masks[:, 0] = False  # Ensure agent's that don't exist don't throw NaNs.
            # [T_all, T_all], time mask for only later steps consider earlier steps.
            time_masks = self.generate_decoder_mask(seq_len=T_all, device=agents_emb.device)
        else:
            time_masks = None

        # (T_all, B, M, d_k) -> (T_all, B*M, d_k)
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_all, B * M, -1)),
                                src_key_padding_mask=agent_masks, mask=time_masks)
        # (T_all, B*M, d_k) -> (T_all, B, M, d_k)
        return agents_temp_emb.view(T_all, B, M, -1)

    def temporal_attn_decoder_fn(self, agents_emb, context, agent_masks, layer):
        '''
        :param agents_emb: (T_fut, B, M, d_k)
        :param context: (T_obs, B, M, d_k)
        :param agent_masks: (B, T_obs, M)
        :return: (T_fut, B, M, d_k)
        '''
        T_fut = agents_emb.shape[0]
        T_obs, B, M, _ = context.shape
        if self.mask_time:
            # [T_fut, T_fut], time mask for only later steps consider earlier steps.
            time_masks = self.generate_decoder_mask(seq_len=T_fut, device=agents_emb.device)
        else:
            time_masks = None
        # (B, T_obs, N) -> (BxM, T_obs)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_emb = self.pos_encoder(agents_emb.reshape(T_fut, -1, self.d_k))  # [T_fut, BxM, H]

        context = context.reshape(-1, B*(M), self.d_k) # [T_obs, BxM, H]

        agents_temp_emb = layer(agents_emb, context, tgt_mask=time_masks, memory_key_padding_mask=agent_masks)
        agents_temp_emb = agents_temp_emb.view(T_fut, B, M, -1)

        return agents_temp_emb

    def social_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T_all, B, M, d_k)
        :param agent_masks: (B, T_hist, M), T_all = 1 when mlp is used
        :return: (T_all, B, M, d_k)
        '''
        T_all, B, M, _ = agents_emb.shape
        T_hist = agent_masks.shape[1]
        T_fut = T_all - T_hist
        if self.mask_social_interaction:
            M = agents_emb.shape[2]
            agent_interaction_mask = (1-torch.eye(M, device=agents_emb.device)).bool()
        else:
            agent_interaction_mask = None

        # ------------------- Agent Mask -------------------
        # (B, T_hist, M) -> (B, 1, M) -> (B, T_fut, M) 
        agent_fut_masks = agent_masks[:, -1:, :].repeat(1, T_fut, 1) # take last timestep of all agents.
        # (B, T_all, M) -> (B*T_all, M)
        agent_masks = torch.cat([agent_masks, agent_fut_masks], dim=-2).view(B*T_all, M)


        # (T_all, B, M, d_k) -> (M, BxT_all, d_k)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(M, B * T_all, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks, mask=agent_interaction_mask)
        # (M, BxT_all, d_k) -> (M, B, T_all, d_k) -> (T_all, B, M, d_k)
        agents_soc_emb = agents_soc_emb.view(M, B, T_all, -1).permute(2, 1, 0, 3)
        return agents_soc_emb
    
    def social_attn_with_edge_decoder_fn(self, agents_emb, agent_masks, layer, aux_info, attn_name="", T_total=83):
        '''
        :param agents_emb: (T_all, B, M, d_k)
        :param agent_masks: (B, T_hist, M), T_all = 1 when mlp is used
        :param aux_info: dict
        :return: (T_all, B, M, d_k)
        '''
        T_all, B, M, _ = agents_emb.shape
        # T_hist = agent_masks.shape[1]
        # T_fut = T_all - T_hist

        # ------------------- Attention Mask -------------------
        # (M, M) -> (M, M, 1) -> (M, M, M) -> (M, M*M)
        attn_block_mask = torch.eye(M, device=agents_emb.device).unsqueeze(-1).repeat(1,1,M).reshape(M, M*M)
        attn_block_mask = (1 - attn_block_mask).to(torch.bool)
        # (M, M*M) -> (1, M, M*M) -> (B*T_all*num_heads, M, M*M)
        attn_block_mask = attn_block_mask.unsqueeze(0).repeat(B*T_all*self.num_heads, 1, 1)
        
        # (M, M) -> (M, 1, M) -> (M, M, M) -> (M, M*M)
        attn_self_unmask = torch.eye(M, device=agents_emb.device).unsqueeze(1).repeat(1,M,1).view(M,M*M)
        attn_self_unmask = (1 - attn_self_unmask).type(torch.bool)
        # (M, M*M) -> (B*T_all*self.num_heads, M, M*M)
        attn_self_unmask = attn_self_unmask.unsqueeze(0).repeat(B*T_all*self.num_heads, 1, 1)
        attn_self_unmask = attn_self_unmask | attn_block_mask
        
        if self.mask_social_interaction:
            attn_mask = attn_self_unmask
        else:
            # (B, M1, M2, T_all)
            neighbor_mask = (1.0 - aux_info['neighbor_feat'][...,-1]).type(torch.bool).to(agents_emb.device)
            # (B, M1, M2, T_all) -> (B, T_all, M1, M2) -> (B, T_all, M1, 1, M2) -> (B, T_all, M1, M1, M2) -> (B*T_all, M1, M1*M2)
            attn_mask = neighbor_mask.permute(0, 3, 1, 2).unsqueeze(-2).repeat(1, 1, 1, M, 1).view(B*T_all, M, M*M)
            # (B*T_all, M1, M1*M2) -> (B*T_all, 1, M1, M1*M2) -> (B*T_all, num_heads, M1, M1*M2) -> (B*T_all*num_heads, M1, M1*M2)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_heads,1,1).view(B*T_all*self.num_heads, M, M*M)
            
            # (B, M1, M2, T_all, 2)
            if self.interaction_edge_speed_repr in ['rel_vel_per_step']:
                # print('self.interaction_edge_speed_repr', self.interaction_edge_speed_repr)
                # print("aux_info['neighbor_feat'].shape", aux_info['neighbor_feat'].shape)
                # print("aux_info['neighbor_feat']", aux_info['neighbor_feat'].shape, aux_info['neighbor_feat'][0, 4, 0, 30:50, [0,1,2,3,4,5,-3,-2]])
                neighbor_pos = aux_info['neighbor_feat'][...,(-3,-2)]
                # (B, M1, M2, T_all, 2) -> (B, M1, M2, T_all) -> (B, T_all, M1, M2)
                neighbor_dist = torch.norm(neighbor_pos, dim=-1).permute(0,3,1,2)
            elif self.interaction_edge_speed_repr in ['rel_vel_new']:
                neighbor_pos = aux_info['neighbor_feat'][...,(-3,-2)]
                # (B, M1, M2, T_all, 2) -> (B, T_all, 2, M1) -> (B, M1, T_all, 2) -> (B, M1, 1, T_all, 2) -> (B, M1, M2, T_all, 2)
                self_pos = torch.diagonal(neighbor_pos, dim1=1, dim2=2).permute(0,3,1,2).unsqueeze(2).repeat(1,1,M,1,1)
                # (B, M1, M2, T_all, 2) -> (B, M1, M2, T_all) -> (B, T_all, M1, M2)
                neighbor_dist = torch.norm(neighbor_pos-self_pos, dim=-1).permute(0,3,1,2)
            else: # backward compatibility for earlier models
                neighbor_pos = aux_info['neighbor_feat'][...,(0,1)]
                # (B, M1, M2, T_all, 2) -> (B, M1, M2, T_all) -> (B, T_all, M1, M2)
                neighbor_dist = torch.norm(neighbor_pos, dim=-1).permute(0,3,1,2)

            neighbor_dist_mask = neighbor_dist > self.social_attn_radius
            # (B, T_all, M1, M2) -> (B, T_all, M1, 1, M2) -> (B, T_all, M1, M1, M2) -> (B*T_all, M1, M1*M2)
            neighbor_dist_mask = neighbor_dist_mask.unsqueeze(-2).repeat(1,1,1,M,1).view(B*T_all, M, M*M)
            # (B*T_all, M1, M1*M2) -> (B*T_all, 1, M1, M1*M2) -> (B*T_all, num_heads, M1, M1*M2) -> (B*T_all*num_heads, M1, M1*M2)
            neighbor_dist_mask = neighbor_dist_mask.unsqueeze(1).repeat(1,self.num_heads,1,1).view(B*T_all*self.num_heads, M, M*M)

            # (B*T_all*num_heads, M1, M1*M2)
            attn_mask = attn_mask | attn_block_mask | neighbor_dist_mask
            # print('0 attn_mask.to(torch.float32).mean()', attn_mask.to(torch.float32).mean())
            # unmask self-attention to avoid invalid values
            attn_mask = attn_mask & attn_self_unmask
            # print('1 attn_mask.to(torch.float32).mean()', attn_mask.to(torch.float32).mean())

            if self.attn_combination == 'gate':
                # mask all self-attentions as we use gate to combine self and neighbors
                attn_self_mask =  ~attn_self_unmask
                attn_mask = attn_mask | attn_self_mask
                # print('3 attn_mask.to(torch.float32).mean()', attn_mask.to(torch.float32).mean())
                # de-mask some to avoid nan values
                if_nan = torch.sum(attn_mask, dim=-1) == M*M
                attn_mask[if_nan] = attn_mask[if_nan] & attn_self_unmask[if_nan]
                # print('4 attn_mask.to(torch.float32).mean()', attn_mask.to(torch.float32).mean())
                # print('0 if_nan.to(torch.float32).mean()', if_nan.to(torch.float32).mean())

        # # ------------------- Agent Mask -------------------
        # # (M, M) -> (1, M, M) -> (B*T_all, M*M)
        # agent_self_unmask = (1.0 - torch.eye(M, device=agents_emb.device).unsqueeze(0).repeat(B*T_all,1,1).view(B*T_all,M*M)).type(torch.BoolTensor).to(agents_emb.device)
        # if self.mask_social_interaction:
        #     agent_masks = agent_self_unmask
        # else:
        #     # (B, T_hist, M) -> (B, 1, M) -> (B, T_fut, M) 
        #     agent_fut_masks = agent_masks[:, -1:, :].repeat(1, T_fut, 1) # take last timestep of all agents.
        #     # (B, T_all, M) -> (B, T_all, 1, M) -> (B, T_all, M, M) -> (B*T_all, M*M)
        #     agent_masks = torch.cat([agent_masks, agent_fut_masks], dim=-2).unsqueeze(-2).repeat(1,1,M,1).view(B*T_all, M*M)
        #     agent_masks = agent_masks & agent_self_unmask

        #     # if self.attn_combination == 'gate':
        #     #     # mask all self-attentions
        #     #     agent_self_mask = ~agent_self_unmask
        #     #     agent_masks = agent_masks | agent_self_mask
        #     #     # to avoid nan values
        #     #     nan_row = torch.sum(agent_masks, dim=-1) == M*M
        #     #     # print('torch.mean(nan_row)', torch.mean(nan_row.to(torch.float32)))
        #     #     # print('0 torch.any(torch.sum(agent_masks, dim=-1) == M*M)', torch.any(torch.sum(agent_masks, dim=-1) == M*M))
        #     #     agent_masks[nan_row] = agent_masks[nan_row] & agent_self_unmask[nan_row]
        #     #     # if torch.any(torch.sum(agent_masks, dim=-1) == M*M):
        #     #     #     print('1 torch.any(torch.sum(agent_masks, dim=-1) == M*M)', torch.any(torch.sum(agent_masks, dim=-1) == M*M))

        # print('attn_mask', attn_mask.shape, attn_mask[0, 1])
        # print('agent_masks', agent_masks.shape, agent_masks[0])
        # ------------------- Query -------------------
        aux_info_neighbor_hist_feat = aux_info['neighbor_hist_feat']
        aux_info_neighbor_fut_feat = aux_info['neighbor_fut_feat']

        edge_hist_feat = aux_info_neighbor_hist_feat[..., self.neighbor_inds]
        edge_fut_feat = aux_info_neighbor_fut_feat[..., self.neighbor_inds]
        if self.mask_edge:
            edge_hist_feat = torch.zeros_like(edge_hist_feat, device=edge_hist_feat.device)
            edge_fut_feat = torch.zeros_like(edge_fut_feat, device=edge_fut_feat.device)

        # (B, M, M, T_hist, edge_attr) -> (B*M*M*T_hist, edge_attr)
        edge_hist_feat = edge_hist_feat.view(-1, len(self.neighbor_inds))
        # (B, M, M, T_fut, edge_attr) -> (B*M*M*T_fut, edge_attr)
        edge_fut_feat = edge_fut_feat.view(-1, len(self.neighbor_inds))

        if len(self.edge_attr_separation) > 0:
            edge_hist_feat = [edge_hist_feat[...,attrs] for attrs in self.edge_attr_separation]
            edge_fut_feat = [edge_fut_feat[...,attrs] for attrs in self.edge_attr_separation]

        # (B*M*M*T_hist, edge_attr) -> (B*M*M*T_hist, d_edge)
        edge_hist_feat = self.neighbor_hist_feat_encoder(edge_hist_feat)
        # (B*M*M*T_fut, edge_attr) -> (B*M*M*T_fut, d_edge)
        edge_fut_feat = self.neighbor_fut_feat_encoder(edge_fut_feat)

        # (B*M*M*T_hist, d_edge) -> (B, M, M, T_hist, d_edge)
        edge_hist_feat = edge_hist_feat.view(B, M, M, -1, self.d_edge)
        # (B*M*M*T_fut, d_edge) -> (B, M, M, T_fut, d_edge)
        edge_fut_feat = edge_fut_feat.view(B, M, M, -1, self.d_edge)

        # (B, M, M, T_all, d_edge)
        edge_feat = torch.cat([edge_hist_feat, edge_fut_feat], dim=-2)
        # (B, M, M, T_all, d_edge) -> (M, M, B, T_all, d_edge) -> (M*M, BxT_all, d_edge)
        edge = edge_feat.permute(1, 2, 0, 3, 4).reshape(M*M, B * T_all, self.d_edge)

        # (T_all, B, M, K_neigh) -> (M, B, T_all, K_neigh) -> (M, B*T_all, K_neigh)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(M, B * T_all, -1)
        # (M, B*T_all, d_k) -> (1, M, B*T_all, d_k) -> (M, M, B*T_all, d_k) -> (M*M, B*T_all, d_k)
        agents_emb_expand = agents_emb.unsqueeze(0).repeat(M, 1, 1, 1).view(M*M, B*T_all, self.d_k)
        # (M*M, B*T_all, d_k+d_edge)
        agents_emb_expand_and_edge = torch.cat([agents_emb_expand, edge], dim=-1)

        # agents_soc_emb = layer(agents_emb, agents_emb_expand_and_edge, memory_mask=attn_mask, memory_key_padding_mask=agent_masks, attn_name=attn_name)
        agents_soc_emb, attn_weights = layer(agents_emb, agents_emb_expand_and_edge, memory_mask=attn_mask, attn_name=attn_name, T_total=T_total)
        # (M, BxT_all, d_k) -> (M, B, T_all, d_k) -> (T_all, B, M, d_k)
        agents_soc_emb = agents_soc_emb.view(M, B, T_all, -1).permute(2, 1, 0, 3)
        
        return agents_soc_emb, attn_weights

    def forward(self, x_noisy, aux_info, t_inp, use_cond=True):
        '''
        : param x_noisy: (B, M, T_fut, k_output)
        : param aux_info: 
            agent_hist_feat mlp:(B, M, 1, d_k), transformer:(B, M, T_hist, d_k)
            map_feat cnn:(B, M, 1, d_k), transformer:(B, M, S_seg, d_k)

        : param t_inp: (B)
        : param use_cond: bool, if condition is used. When False, dummy values are provided.

        :return:
            out: [B, M, (T_hist+)T_fut, output_dim]
        '''
        B, M, T_fut, _ = x_noisy.shape
        attn_weights = None
        # if we put everything (subset of {map, agent_hist, neighbor_hist}) into cond_feat
        if self.single_cond_feat:
            if use_cond:
                agent_hist_feat = aux_info['cond_feat'].unsqueeze(-2)
            else:
                agent_hist_feat = aux_info['cond_feat_non_cond'].unsqueeze(-2)
        else:
            if use_cond:
                agent_hist_feat = aux_info['agent_hist_feat']
                map_feat = aux_info['map_feat']
            else:
                agent_hist_feat = aux_info['agent_hist_feat_non_cond']
                map_feat = aux_info['map_feat_non_cond']
        # Process agent history information
        if self.agent_hist_embed_method == 'transformer':
            # extract agent history vectors
            # Encode all input observations
            # (B, M, T, k_attr+1) -> (B, T, M, k_attr+1)
            agent_hist_feat = agent_hist_feat.permute(0, 2, 1, 3)
            # (B, M, T, k_attr+1) -> (B, T, M, k_attr), (B, T, M)
            agents_tensor, opps_masks = agent_hist_feat[..., :-1], (1.0 - agent_hist_feat[..., -1]).type(torch.BoolTensor).to(agent_hist_feat.device) 

            # (B, T, M, k_attr) -> (T, B, M, d_k)
            agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

            # Process through AutoBot's encoder
            for i in range(self.L_enc):
                agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
                agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])
            # (T, B, M, d_k)
        elif self.agent_hist_embed_method == 'mlp':
            # (B, M, 1, d_k) -> (1, B, M, d_k)
            agents_emb = agent_hist_feat.permute(2, 0, 1, 3)
            # dummy mask (B, 1, M)
            opps_masks = torch.zeros((B, 1, M)).type(torch.BoolTensor).to(agents_emb.device)
        elif self.agent_hist_embed_method == 'concat':
            # extract agent history vectors
            # Encode all input observations
            # (B, M, T_hist, k_attr+1) -> (B, T_hist, M, k_attr+1)
            agent_hist_feat = agent_hist_feat.permute(0, 2, 1, 3)
            # (B, T_hist, M, k_attr+1) -> (B, T_hist, M, k_attr), (B, T_hist, M)
            agents_tensor, opps_masks = agent_hist_feat[..., :-1], (1.0 - agent_hist_feat[..., -1]).type(torch.BoolTensor).to(agent_hist_feat.device)

            # (B, T_hist, M, k_attr) -> (T_hist, B, M, d_k)
            agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)
            # TBD: think about how to deal with availability field properly
        else:
            raise NotImplementedError
        
        # Process map information
        if self.use_map_lanes:
            if self.map_embed_method == 'transformer':
                # (S_seg, B, M, d_k) -> (S_seg, B, M, d_k+1) add availability dimension
                nan_inds = torch.sum(torch.isnan(map_feat).float(), dim=-1) > 0
                avail = torch.ones_like(map_feat[...,0])
                avail[nan_inds] = 0
                # TBD: currently convert nan values to 0.0, maybe try other values?
                map_feat = torch.nan_to_num(map_feat, nan=0.0)
                map_feat = torch.cat([map_feat, avail.unsqueeze(-1)], dim=-1)

                # extract map vectors
                # (B, M, S_seg, P_per_seg, map_attr) -> (S_seg, B, M, d_k)
                orig_map_features, orig_road_segs_masks = self.map_encoder(map_feat)
                # (S_seg, B, M, d_k) -> (S_seg, BxM, d_k)
                map_features = orig_map_features.view(-1, B * M, self.d_k)
                # (B, M, S_seg) -> (BxM, S_seg)
                road_segs_masks = orig_road_segs_masks.view(B * M, -1)
            elif self.map_embed_method == 'cnn':
                # (B, M, 1, d_k) -> (1, BxM, d_k)
                map_features = map_feat.permute(2, 0, 1, 3).reshape(1, B*M, -1)
                # dummy mask (B*M, 1)
                road_segs_masks = torch.zeros((B*M, 1)).type(torch.BoolTensor).to(map_features.device)
            else:
                raise NotImplementedError

        # AutoBot-Joint Decoding
        # (B, M, T_fut, output_dim) -> (B, T_fut, M, output_dim)
        dec_parameters = x_noisy.permute(0, 2, 1, 3)
        # (B, T_fut, M, output_dim) -> (T_fut, B, M, d_k)
        agents_dec_emb = self.agents_noisy_input_encoder(dec_parameters).permute(1, 0, 2, 3)
        if self.agent_hist_embed_method == 'concat':
            # (T_fut, B, M, d_k) -> (T_hist+T_fut, B, M, d_k)
            agents_dec_emb = torch.cat([agents_emb, agents_dec_emb], dim=0)
            T_hist = agents_emb.shape[0]
            T_total = T_hist + T_fut
        else:
            T_total = T_fut
        # Add time embedding for the denoising step k
        # (B) -> (B, d_k) -> (1, B, 1, d_k)
        time_embed = self.time_mlp(t_inp).unsqueeze(0).unsqueeze(-2)
        agents_dec_emb = agents_dec_emb + time_embed
        for d in range(self.L_dec):
            # Cross-Attention with Map
            if self.use_map_lanes and d == 1:
                # (T_total, B, M, d_k) -> (T_total, B*M, d_k)
                agents_dec_emb = agents_dec_emb.reshape(T_total, -1, self.d_k)
                agents_dec_emb_map = self.map_attn_layers(query=agents_dec_emb, key=map_features, value=map_features, key_padding_mask=road_segs_masks)[0]
                agents_dec_emb = agents_dec_emb + agents_dec_emb_map
                # (T_total, B*M, d_k) -> (T_total, B, M, d_k)
                agents_dec_emb = agents_dec_emb.reshape(T_total, B, M, -1)
            # agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, map_features, opps_masks, layer=self.self.temporal_attn_decoder_map_layers[d])
            if self.agent_hist_embed_method in ['mlp', 'transformer']:
                agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, agents_emb, opps_masks, layer=self.temporal_attn_decoder_layers[d])
            # add this extra self temporal layer to capture the temporal dynamics of the agents
            agents_dec_emb = self.self_temporal_attn_decoder_fn(agents_dec_emb, opps_masks, layer=self.self_temporal_attn_decoder_layers[d])

            if self.neigh_hist_embed_method == 'interaction_edge':
                if d == 0 or self.all_interactive_social:
                    if 'attn_name' not in aux_info or aux_info['attn_name']=='':
                        attn_name_str = ""
                    else:
                        attn_name_str = aux_info['attn_name']+"_"+str(d)
                    agents_dec_emb, attn_weights = self.social_attn_with_edge_decoder_fn(agents_dec_emb, opps_masks, layer=self.social_attn_decoder_layers[d], aux_info=aux_info, attn_name=attn_name_str, T_total=T_total)
                else:
                    agents_dec_emb = self.social_attn_decoder_fn(agents_dec_emb, opps_masks, layer=self.social_attn_decoder_layers[d])
            else:
                agents_dec_emb = self.social_attn_decoder_fn(agents_dec_emb, opps_masks, layer=self.social_attn_decoder_layers[d])
        # (T_total, B, M, d_k) -> (T_total*B*M, output_dim)
        out = self.output_model(agents_dec_emb.reshape(-1, self.d_k))
        # (T_total*B*M, output_dim) -> (T_total, B, M, output_dim) -> (B, M, T_total, output_dim)
        out = out.reshape(T_total, B, M, -1).permute(1, 2, 0, 3)
        pred_info = {'attn_weights': attn_weights}

        return out, pred_info

class SocialTransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, attn_name="", T_total=83):
        output = tgt

        for i, mod in enumerate(self.layers):
            if attn_name != "":
                attn_name_str = attn_name+"_"+str(i)
            else:
                attn_name_str = ""
            output, attn_weights_i = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask, attn_name=attn_name_str, T_total=T_total)
            # pick the first layer's attn_weights
            if i == 0:
                attn_weights = attn_weights_i

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights

class SocialDecoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 edge_dim: int,
                 num_heads: int = 16,
                 dropout: float = 0.1,
                 attn_combination: str = 'sum') -> None:
        super(SocialDecoderLayer, self).__init__()
        self.self_attn_with_edge = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, kdim=embed_dim+edge_dim, vdim=embed_dim+edge_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.attn_combination = attn_combination
        if self.attn_combination == 'gate':
            self.lin_self = nn.Linear(embed_dim, embed_dim)
            self.lin_ih = nn.Linear(embed_dim, embed_dim)
            self.lin_hh = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                attn_name: str = "",
                T_total: int = 83) -> torch.Tensor:
        '''
        Input:
            src: (M, B*T_all, d_k)
            memory: (M, M, B*T_all, d_edge)
            memory_mask: (B*T_all*num_heads, M, M*M)
            memory_key_padding_mask: (B*T_all, M*M)
            attn_name: str, the name of attention weights to be visualized. If "", no visualization.
        Output:
            x: (M, B*T_all, d_k)
            attn_weights: (M, M)
        '''
        x, x_expand_and_edge = tgt, memory

        # always return weights but not necessarily visualize them
        attn, attn_weights = self._sa_block(self.norm1(x), x_expand_and_edge, key_padding_mask=memory_key_padding_mask, attn_mask=memory_mask, need_weights=True)
        if attn_name != '':
            visualize_attn(attn_weights, attn_name, T_total)

        BT, M, MM = attn_weights.size()
        B = BT // T_total
        attn_weights = attn_weights.reshape(B, T_total, M, MM)
        row_indices = torch.arange(M).view(-1, 1).expand(-1, M)
        col_indices = (torch.arange(M).view(-1, 1) * M + torch.arange(M).view(1, -1)).expand(M, -1)
        # (B, S, M, MM) -> (B, S, M, M)
        attn_weights_selected = attn_weights[:, :, row_indices, col_indices]

        if self.attn_combination == 'gate':
            gate = torch.sigmoid(self.lin_ih(attn) + self.lin_hh(x))
            x = attn + gate * (self.lin_self(x) - attn)
        else:
            x = x + attn
        x = x + self._ff_block(self.norm2(x))

        return x, attn_weights_selected

    def _sa_block(self,
                  x: torch.Tensor,
                  x_expand_and_edge: torch.Tensor,
                  key_padding_mask: Optional[torch.Tensor],
                  attn_mask: Optional[torch.Tensor],
                  need_weights: bool=False) -> torch.Tensor:

        x, attn_weights = self.self_attn_with_edge(x, x_expand_and_edge, x_expand_and_edge, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=need_weights)

        return self.dropout1(x), attn_weights

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)

def visualize_attn(attn_weights, attn_name='', T_total=83):
    from matplotlib import pyplot as plt
    T_hist = 31
    ts = [31, 51]
    S = len(ts)
    BT, M, MM = attn_weights.size()
    B = BT // T
    # (BT, M, MM) -> (B, T, M, MM) -> (T, M, MM) -> (S, M, MM)
    attn_weights = attn_weights.reshape(B, T_total, M, MM).mean(0)[ts, ...]
    # (S, M, MM)
    # attn_weights = attn_weights.view(S, M, M, M)

    row_indices = torch.arange(M).view(-1, 1).expand(-1, M)
    col_indices = (torch.arange(M).view(-1, 1) * M + torch.arange(M).view(1, -1)).expand(M, -1)
    # (S, M, MM) -> (S, M, M)
    attn_weights = attn_weights[:, row_indices, col_indices]

    fig_size = (int(MM//3), int(M*S//3))
    fig_size_2 = (M, M*S)
    
    # Plot the attention weights for each timstep
    _, axs = plt.subplots(nrows=S, ncols=1, figsize=fig_size_2)
    for i, ax in enumerate(axs):
        attn_weights_i = attn_weights[i].cpu().detach().numpy()
        ax.imshow(attn_weights_i, cmap='hot', interpolation='nearest', origin='lower')
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.set_xticklabels([i for i in range(M)], fontsize=40)
        ax.set_yticklabels([i for i in range(M)], fontsize=40)
        ax.set_title(f'Planning Timestep {ts[i]-T_hist}', fontsize=50)

        # add annotations
        for k in range(attn_weights_i.shape[0]):
            for j in range(attn_weights_i.shape[1]):
                ax.annotate(str(round(attn_weights_i[k, j], 2)), xy=(j, k), 
                            horizontalalignment='center',
                            verticalalignment='center')
    plt.tight_layout()
    dir = 'nusc_results/attn_weights/'
    os.makedirs(dir, exist_ok=True)
    plt.savefig(dir+'attn_weights_'+attn_name+'.png')
    plt.close()

class MultipleInputEmbedding(nn.Module):
    def __init__(self,
                    in_channels: List[int],
                    out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                            nn.LayerNorm(out_channel),
                            nn.ReLU(inplace=True),
                            nn.Linear(out_channel, out_channel))
                for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)