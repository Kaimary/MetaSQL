import torch
from torch.nn.modules import Module
from torch import nn
import torch.nn.functional as F
from overrides import overrides
from typing import Dict, List

from allennlp.nn import util
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import Average

from allenmodels.modules.mask_transformer import MaskTransformerEncoder
from ..metrics import NDCG, MRR
from allenmodels.models.losses.neuralNDCG import neuralNDCG

@Model.register('listwise_ranker')
class ListwiseRanker(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            embedder: TextFieldEmbedder,
            embedder1: TextFieldEmbedder,
            loss_weight: List = (0.2, 0.8, 1.0),
            # triplet_loss_weight: List = (1, 1, 1, 1),
            # triplet_score_weight: List = (1, 1, 1),
            dropout: float = None,
            ff_dim: int = 256,
    ):
        super().__init__(vocab)
        self._embedder = embedder
        self._embedder1 = embedder1

        self._dropout = dropout and torch.nn.Dropout(dropout)
        self._ff_dim = ff_dim
        # self.lw_0, self.lw_1, self.lw_2, self.lw_3 = triplet_loss_weight
        # self.sw_1, self.sw_2, self.sw_3 = triplet_score_weight
        self.pair_lw, self.list_lw_coarse, self.list_lw_fine = loss_weight

        self._pooler = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self._embedder.get_output_dim(),
                out_features=self._ff_dim,
            ),
            torch.nn.Tanh()
        )
    
        self._pooler2 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self._embedder.get_output_dim(),
                out_features=self._ff_dim,
            ),
            torch.nn.Tanh()
        )

        # self._sumLayer = nn.Linear(5, 1, bias=False)
        self._denseLayer = nn.Linear(self._ff_dim, 1, bias=False)
        self._denseLayer1 = nn.Linear(self._ff_dim, 1, bias=False)

        self._sim = BlockCosineSimilarity()
        self._loss = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=lambda a, b: 1.0 - self._sim(a, b), 
            reduction="none", 
            margin=0.4
        )

        self.acc = Average()
        self._mrr = MRR(padding_value=-1)
        self._ndcg = NDCG(padding_value=-1)

    def pooling(self, seg_list, pooler):
        return pooler(seg_list[:, 0])
    # def pooling(self, seg_list):
    #     return self._pooler1(seg_list[:, 0])

    def global_seg_vec(self, sent: Dict, segs: Dict = None):
        C, C_mask, E, E_mask = None, None, None, None
        if sent:
            C = self._embedder1(sent)
            C_mask = util.get_text_field_mask(sent)
        if segs:
            E = self._embedder1(segs, num_wrapping_dims = 1)
            E_mask = util.get_text_field_mask(segs, num_wrapping_dims = 1)
        return C, C_mask, E, E_mask


    def score(self, h_N, H_C, E_C_mask):
        # s_0 = self._sim(a_N, a_S)
        # s_1 = self._sim(h_N, h_S)
        s_2 = self._sim(h_N, H_C, E_C_mask)
        # s_3 = self._sim(h_S, H_P)
        # score = self.sw_3 * s_0 + self.sw_1 * s_1 + self.sw_2 * s_2 #+ self.sw_3 * s_3
        score = s_2
        return score

    def forward(
            self,
            metadata,
            nl, # phrases,
            clauses0, clauses1, clauses2, clauses3, clauses4, clauses5, clauses6, clauses7, clauses8, clauses9,
            # query_clauses,
            query_dialect_pairs,
            # gold_sql=None,
            gold_clauses=None,
            # neg_sql=None,
            neg_clauses=None,
            # pos_clause_mask: torch.LongTensor = None,
            neg_clause_mask: torch.LongTensor = None,
            labels: torch.IntTensor = None 
    ) -> Dict[str, torch.Tensor]:
        output_dict = {}
        embedded_query_dialect_pairs = self._embedder(query_dialect_pairs, num_wrapping_dims=1)
        batch_size, num_pairs, num_tokens, _ = embedded_query_dialect_pairs.size()
        if self._dropout: embedded_query_dialect_pairs = self._dropout(embedded_query_dialect_pairs)
        encoder_outputs = self._pooler(
            embedded_query_dialect_pairs.view(batch_size * num_pairs, num_tokens, -1)[:, 0]
        )
        encoder_outputs = encoder_outputs.view(batch_size, num_pairs, -1)
        coarse_gained_scores = self._denseLayer(encoder_outputs).squeeze(-1)
        # coarse_gained_scores = torch.sigmoid(self._denseLayer(encoder_outputs).squeeze(-1))


        # embedded_query_clauses = self._embedder(query_clauses, num_wrapping_dims=1)
        # _, _, num_tokens, _ = embedded_query_clauses.size()
        # if self._dropout: embedded_query_dialect_pairs = self._dropout(embedded_query_clauses)
        # encoder_outputs1 = self._pooler(
        #     embedded_query_clauses.view(batch_size * num_pairs, num_tokens, -1)[:, 0]
        # )
        # encoder_outputs1 = encoder_outputs1.view(batch_size, num_pairs, -1)
        # fine_grained_scores = self._denseLayer(encoder_outputs1).squeeze(-1)


        fine_grained_scores = []
        for clauses in [clauses0, clauses1, clauses2, clauses3, clauses4, clauses5, clauses6, clauses7, clauses8, clauses9]:
            # print(clauses['bert']['token_ids'].size())
            _, _, E_C, E_C_mask = self.global_seg_vec(None, clauses)
            if self._dropout: E_C = self._dropout(E_C)
            batch_size, num_cls, cls_toks = E_C_mask.shape
            H_C = torch.reshape(E_C, (batch_size*num_cls, cls_toks, -1))
            H_C = self.pooling(H_C,  self._pooler2)
            H_C = torch.reshape(H_C, (batch_size, num_cls, -1))
            fine_grained_output = self._denseLayer1(H_C).squeeze(-1)
            fine_grained_output = fine_grained_output * E_C_mask[:, :, 0]
            fine_grained_output = fine_grained_output.sum(-1)
            nums = E_C_mask[:, :, 0].sum(-1)
            fine_grained_output = torch.div(fine_grained_output, nums)
            fine_grained_scores.append(fine_grained_output)
            # h_N, h_S, H_C, a_N, a_S = self.encoding(C_N, C_N_mask, C_S, C_S_mask, E_C, E_C_mask)
            # fine_grained_scores.append(self.score(h_N, H_C, E_C_mask[:, :, 0]))
        fine_grained_scores = torch.stack(fine_grained_scores, dim=1)
        # fine_grained_scores = torch.sigmoid(torch.stack(fine_grained_scores, dim=1))

        # if any(meta['index'] % 1000 == 0 for meta in  metadata):
        #     print(f"\ncoarse gained scores:\n{coarse_gained_scores}")
        #     print(f"\nfine grained scores:\n {fine_grained_scores}")

        # scores = fine_grained_scores
        scores = coarse_gained_scores + fine_grained_scores
        # scores = coarse_gained_scores + 2 * fine_grained_scores - abs(coarse_gained_scores - fine_grained_scores)
        output_dict['logits'] = scores
        # probs = torch.sigmoid(scores)

        if labels is not None:
            # Metrics calcuation
            label_mask = (labels != -1)
            self._mrr(scores, labels, label_mask)
            self._ndcg(scores, labels, label_mask)
            # p_score = self.score(h_N, h_S, H_C, a_N, a_S)
            # n_score = self.score(n_h_N, n_h_S, n_H_C, n_a_N, n_a_S)
            # acc_list = p_score > n_score
            # for acc in acc_list: self.acc(acc)

        if self.training:
            listwise_loss = neuralNDCG(coarse_gained_scores.double(), labels)
            listwise_loss2 = neuralNDCG(fine_grained_scores.double(), labels)
            # C_S, _, _, _ = self.global_seg_vec(gold_sql)
            # _, _, E_C, E_C_mask = self.global_seg_vec(None, gold_clauses)
            # if self._dropout: E_C = self._dropout(E_C)
            # _, num_cls, cls_toks = E_C_mask.shape
            # H_C = torch.reshape(E_C, (batch_size*num_cls, cls_toks, -1))
            # H_C = self.pooling(H_C)
            # H_C = torch.reshape(H_C, (batch_size, num_cls, -1))
            
            # n_C_S, _, _, _ = self.global_seg_vec(neg_sql, None)
            # n_h_N, n_h_S, n_H_C, n_a_N, n_a_S = self.encoding(
            #     C_N, C_N_mask, n_C_S, n_C_S_mask, n_E_C, n_E_C_mask
            # )

            # C_N, _, _, _ = self.global_seg_vec(nl)
            # if self._dropout: C_N = self._dropout(C_N)
            # h_N = self.pooling(C_N, self._pooler2)

            # _, _, E_C, E_C_mask = self.global_seg_vec(None, gold_clauses)
            # _, _, n_E_C, n_E_C_mask = self.global_seg_vec(None, neg_clauses)
            # if self._dropout: 
            #     E_C = self._dropout(E_C)
            #     n_E_C = self._dropout(n_E_C)
            # _, num_cls1, cls_toks1 = E_C_mask.shape
            # H_C = torch.reshape(E_C, (batch_size*num_cls1, cls_toks1, -1))
            # H_C = self.pooling(H_C,  self._pooler2)
            # H_C = torch.reshape(H_C, (batch_size, num_cls1, -1))
            # n_H_C_p = H_C * E_C_mask[:, :, 0].unsqueeze(-1)

            # _, num_cls, cls_toks = n_E_C_mask.shape
            # n_H_C = torch.reshape(n_E_C, (batch_size*num_cls, cls_toks, -1))
            # n_H_C = self.pooling(n_H_C,  self._pooler2)
            # n_H_C = torch.reshape(n_H_C, (batch_size, num_cls, -1))
            # # n_H_C = n_H_C * n_E_C_mask[:, :, 0].unsqueeze(-1)
            # # pos_clause_mask = pos_clause_mask.unsqueeze(-1)
            # neg_clause_mask = neg_clause_mask.unsqueeze(-1)
            # # n_H_C_p = n_H_C * pos_clause_mask
            # n_H_C_n = n_H_C * neg_clause_mask
            # # # Listwise loss
            output_dict['loss'] = self.list_lw * listwise_loss + self.list_lw * listwise_loss2
            # # # Triplet loss
            # # # L_0 = self._loss(a_N, a_S, n_a_S) #+ self._loss(a_S, a_N, n_a_N)
            # # # L_1 = self._loss(h_N, h_S, n_h_S) #+ self._loss(h_S, h_N, n_h_N)
            # # # L_2 = self._loss(h_N, H_C, n_H_C) #+ self._loss(h_S, H_P, n_H_P)
            # L_3 = self._loss(h_N, n_H_C_p, n_H_C_n) #+ self._loss(n_h_N, p_H_C_p, p_H_C_n)
            # triplet_loss = self.lw_3 * L_3
            # # # triplet_loss = self.lw_0 * L_0 + self.lw_1 * L_1 + self.lw_2 * L_2 + self.lw_3 * L_3
            # output_dict['loss'] = self.pair_lw * triplet_loss.sum() + self.list_lw * listwise_loss + self.list_lw * listwise_loss2

        return output_dict
    #
    # @overrides
    # def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """ Does a simple argmax over the probabilities, converts index to string label,
    #         and add label key to the dictionary with the result """
    #     classes = []
    #     predictions = output_dict["probabilities"].cpu().data.numpy()
    #     # print(f"predictions:{predictions[0]}")
    #     # Its a multilabel classification so, need to iterate through all of the labels
    #     for i, p in enumerate(predictions[0]):
    #         if p > 0.5:
    #             label = self.vocab.get_token_from_index(i, namespace="labels")
    #             classes.append(label)
    #
    #     output_dict['labels'] = [', '.join(classes)]
    #     # print(output_dict['labels'])
    #     return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            # "accuracy": self.acc.get_metric(reset),
            "mrr": self._mrr.get_metric(reset),
            "ndcg": self._ndcg.get_metric(reset),
        }
        return metrics


class BlockCosineSimilarity(Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(BlockCosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if len(x1.shape) == 2 and len(x2.shape) == 3:
            bts, num, emb_dim = x2.shape
            r_x1 = x1.repeat((1, num)).view(-1, emb_dim)
            r_x2 = x2.reshape(bts * num, emb_dim)
            sim = F.cosine_similarity(r_x1, r_x2, self.dim, self.eps) # / num
            res = sim.view(bts, -1).sum(-1)
            if torch.is_tensor(mask):
                nums = mask.sum(-1)
            else:
                nonZeroRows = torch.abs(x2).sum(dim=2) > 0
                nums = nonZeroRows.sum(-1)
            res = torch.div(res, nums)
            return res
        return F.cosine_similarity(x1, x2, self.dim, self.eps)