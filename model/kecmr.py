import torch
import torch.nn as nn
# from torch.nn import MultiheadAttention
# from torch.nn import LayerNorm
from .net_utils import FC, FFN, AttFlat, LayerNorm, MHAtt


class KeyEntityEtract(nn.Module):
    def __init__(self, __C):
        super(KeyEntityEtract, self).__init__()
        self.__C = __C
        self.v_top_num = __C.IMG_KEY_NUM
        self.t_top_num = __C.TEXT_KEY_NUM

        self.mhatt_v_1 = MHAtt(__C)
        self.mhatt_v_2 = MHAtt(__C)
        self.ffn_v = FFN(__C)

        self.dropout_v_1 = nn.Dropout(__C.DROPOUT_R)
        self.norm_v_1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_v_2 = nn.Dropout(__C.DROPOUT_R)
        self.norm_v_2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_v_3 = nn.Dropout(__C.DROPOUT_R)
        self.norm_v_3 = LayerNorm(__C.HIDDEN_SIZE)

        self.score_linear_v = FC(
            __C.HIDDEN_SIZE, 1, dropout_r=__C.DROPOUT_R, use_relu=False
        )

        self.mhatt_t_1 = MHAtt(__C)
        self.mhatt_t_2 = MHAtt(__C)
        self.ffn_t = FFN(__C)

        self.dropout_t_1 = nn.Dropout(__C.DROPOUT_R)
        self.norm_t_1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_t_2 = nn.Dropout(__C.DROPOUT_R)
        self.norm_t_2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_t_3 = nn.Dropout(__C.DROPOUT_R)
        self.norm_t_3 = LayerNorm(__C.HIDDEN_SIZE)

        self.score_linear_t = FC(
            __C.HIDDEN_SIZE, 1, dropout_r=__C.DROPOUT_R, use_relu=False
        )

        self.mhatt_q_1 = MHAtt(__C)
        self.ffn_q = FFN(__C)

        self.dropout_q_1 = nn.Dropout(__C.DROPOUT_R)
        self.norm_q_1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout_q_2 = nn.Dropout(__C.DROPOUT_R)
        self.norm_q_2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(
        self, v_feat, t_feat, ques_feat, v_feat_mask, t_feat_mask, ques_feat_mask
    ):
        # self attn
        v_feat = self.selfattn(
            v_feat, v_feat_mask, self.mhatt_v_1, self.dropout_v_1, self.norm_v_1
        )

        t_feat = self.selfattn(
            t_feat, t_feat_mask, self.mhatt_t_1, self.dropout_t_1, self.norm_t_1
        )

        ques_feat = self.selfattn(
            ques_feat, ques_feat_mask, self.mhatt_q_1, self.dropout_q_1, self.norm_q_1
        )
        ques_feat = self.norm_q_2(ques_feat + self.dropout_q_2(self.ffn_q(ques_feat)))
        # ques attn
        v_feat = self.crossattn(
            v_feat,
            ques_feat,
            ques_feat_mask,
            self.mhatt_v_2,
            self.dropout_v_2,
            self.norm_v_2,
        )

        t_feat = self.crossattn(
            t_feat,
            ques_feat,
            ques_feat_mask,
            self.mhatt_t_2,
            self.dropout_t_2,
            self.norm_t_2,
        )

        # key entity extract
        v_feat, v_score = self.score(
            v_feat,
            v_feat_mask,
            self.ffn_v,
            self.dropout_v_3,
            self.norm_v_3,
            self.score_linear_v,
        )
        v_key_index = self.extract(v_score, v_feat_mask, self.v_top_num)

        t_feat, t_score = self.score(
            t_feat,
            t_feat_mask,
            self.ffn_t,
            self.dropout_t_3,
            self.norm_t_3,
            self.score_linear_t,
        )
        t_key_index = self.extract(t_score, t_feat_mask, self.t_top_num)

        return v_feat, t_feat, ques_feat, v_key_index, t_key_index

    def selfattn(self, feat, feat_mask, mhatt, dropout, norm):
        return norm(feat + dropout(mhatt(feat, feat, feat, feat_mask)))

    def crossattn(self, feat, m_feat, m_mask, mhatt, dropout, norm):
        feat = norm(feat + dropout(mhatt(m_feat, m_feat, feat, m_mask)))
        return feat

    def score(self, feat, feat_mask, ffn, dropout, norm, score_linear):
        feat = norm(feat + dropout(ffn(feat)))
        return feat, score_linear(feat)

    def extract(self, score, mask, top_num):
        score = score.masked_fill(mask.squeeze(1).squeeze(1).unsqueeze(-1), -1e9)
        _, key_index = torch.topk(score, top_num, dim=1, largest=True)
        # key_index, _ = torch.sort(key_index, stable=True, dim=1)
        return key_index.repeat(1, 1, self.__C.HIDDEN_SIZE)


class CrossMediaReason(nn.Module):
    def __init__(self, __C):
        super(CrossMediaReason, self).__init__()
        self.__C = __C

        self.v_top_num = __C.IMG_KEY_NUM
        self.t_top_num = __C.TEXT_KEY_NUM

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(
        self,
        v_feat,
        t_feat,
        ques_feat,
        v_feat_mask,
        t_feat_mask,
        ques_feat_mask,
        v_key_index,
        t_key_index,
    ):
        memory = torch.cat([v_feat, t_feat, ques_feat], dim=1)
        memory_mask = torch.cat([v_feat_mask, t_feat_mask, ques_feat_mask], dim=-1)

        v_key = torch.gather(v_feat, dim=1, index=v_key_index)
        t_key = torch.gather(t_feat, dim=1, index=t_key_index)
        ques_key = ques_feat
        query = torch.cat([v_key, t_key, ques_key], dim=1)
        query_mask = self.make_mask(query)

        query = self.norm1(
            query + self.dropout1(self.mhatt1(query, query, query, query_mask))
        )
        query = self.norm2(
            query + self.dropout2(self.mhatt2(memory, memory, query, memory_mask))
        )
        query = self.norm3(query + self.dropout3(self.ffn(query)))

        v_key, t_key, ques_key = query.split(
            [self.v_top_num, self.t_top_num, ques_key.size(1)], dim=1
        )

        v_feat = v_feat.scatter(dim=1, index=v_key_index, src=v_key)
        t_feat = t_feat.scatter(dim=1, index=t_key_index, src=t_key)
        ques_feat = ques_key

        return v_feat, t_feat, ques_feat, v_key_index, t_key_index

    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)


class KECMRModule(nn.Module):
    def __init__(self, __C):
        super(KECMRModule, self).__init__()
        self.__C = __C
        self.kee = KeyEntityEtract(__C)
        self.cmr_list = nn.ModuleList(
            [CrossMediaReason(__C) for i in range(__C.CMR_NUM_PER_KECMR)]
        )

    def forward(
        self, v_feat, t_feat, ques_feat, v_feat_mask, t_feat_mask, ques_feat_mask
    ):
        v_feat, t_feat, ques_feat, v_key_index, t_key_index = self.kee(
            v_feat, t_feat, ques_feat, v_feat_mask, t_feat_mask, ques_feat_mask
        )
        for cmr in self.cmr_list:
            v_feat, t_feat, ques_feat, v_key_index, t_key_index = cmr(
                v_feat,
                t_feat,
                ques_feat,
                v_feat_mask,
                t_feat_mask,
                ques_feat_mask,
                v_key_index,
                t_key_index,
            )
        return v_feat, t_feat, ques_feat, v_key_index, t_key_index


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size, embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )

        self.img_feat_linear = nn.Linear(__C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE)

        self.kecmr_list = nn.ModuleList(
            [KECMRModule(__C) for i in range(__C.KECMR_NUM)]
        )

        self.attflat_ques = AttFlat(__C)
        self.attflat_v = AttFlat(__C)
        self.attflat_t = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, v_feat, t_feat, ques_feat):
        # Make mask
        ques_feat_mask = self.make_mask(ques_feat.unsqueeze(2))
        v_feat_mask = self.make_mask(v_feat)
        t_feat_mask = self.make_mask(t_feat.unsqueeze(2))

        # Pre-process Language Feature
        ques_feat = self.embedding(ques_feat)
        ques_feat, _ = self.lstm(ques_feat)

        t_feat = self.embedding(t_feat)
        t_feat, _ = self.lstm(t_feat)

        # Pre-process Image Feature
        v_feat = self.img_feat_linear(v_feat)

        for kecmr in self.kecmr_list:
            v_feat, t_feat, ques_feat, v_key_index, t_key_index = kecmr(
                v_feat, t_feat, ques_feat, v_feat_mask, t_feat_mask, ques_feat_mask
            )

        ques_feat = self.attflat_ques(ques_feat, ques_feat_mask)

        v_key = torch.gather(v_feat, dim=1, index=v_key_index)
        t_key = torch.gather(t_feat, dim=1, index=t_key_index)
        v_feat = self.attflat_v(v_key, self.make_mask(v_key))
        t_feat = self.attflat_t(t_key, self.make_mask(t_key))

        # v_feat = self.attflat_v(v_feat, v_feat_mask)
        # t_feat = self.attflat_t(t_feat, t_feat_mask)

        proj_feat = ques_feat + v_feat + t_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat

    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)
