# -*- coding: utf-8 -*-
# @Time    : 2022/7/4 11:01
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : red.py
# @Software: PyCharm
import random
import torch
from allennlp.nn.util import masked_mean
from torch import nn
from typing import Dict
import torch.nn.functional
from overrides import overrides
from allennlp.models.model import Model
from allennlp.nn import util as annu

from ..metrics import NDCG, MRR
from allennlp.nn import InitializerApplicator
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allenmodels.models.losses.neuralNDCG import neuralNDCG
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

from utils import vec_utils


@Model.register("listwise_pair_ranker")
class RED(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            schema_encoder: Seq2SeqEncoder,
            encoder: Seq2VecEncoder = None,
            dropout: float = 0.1,
            level_trans: float = 1,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._max_embed_num = 20  # keep cuda memory
        self._max_embed_size = 3000  # keep cuda memory
        self._max_candidate_num = 60  # padding num
        self._level_num = 10
        self._device = None
        self._level_trans = 1
        if 0 < level_trans < 1:
            self._level_trans = level_trans

        self._ff_dim = 512
        self._text_field_embedder = text_field_embedder
        vocab_size = self._text_field_embedder.token_embedder_bert.config.vocab_size
        self._text_field_embedder.token_embedder_bert.transformer_model.resize_token_embeddings(vocab_size + 17)
        self._dropout = dropout and torch.nn.Dropout(dropout)
        self._encoder = encoder
        self._emb_to_ff_dim = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self._text_field_embedder.get_output_dim(),
                out_features=self._ff_dim,
            ),
            torch.nn.Tanh()
        )
        self._pooler = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self._ff_dim,
                out_features=self._ff_dim,
            ),
            torch.nn.Tanh()
        )
        self._denseLayer = nn.Linear(self._ff_dim, 1, bias=False)
        self._schema_encoder = schema_encoder

        # self._auc = Auc()
        self._mrr = MRR(padding_value=-1)
        # self._acc = Average()
        self._ndcg = NDCG(padding_value=-1)
        # self._ndcg_level = [NDCG(padding_value=-1) for _ in range(self._level_num)]
        self.output_detail = False

        initializer(self)

    def forward(  # type: ignore
            self,
            metadata: Dict,
            candidates: TextFieldTensors,  # batch * unique_nums * words
            offsets: torch.IntTensor,  # batch * token_num * 2
            relations: torch.IntTensor,  # batch * token_num * token_num
            invert_index: torch.IntTensor = None,  # batch * actual_nums
            gold: torch.IntTensor = None  # batch * num_gold
    ) -> Dict:
        # print(f"gold:{gold.size()}")
        # Memory limit, sample training instance here
        self._device = candidates['bert']['token_ids'].device
        batch_size, num, length = candidates['bert']['token_ids'].shape
        # print(f"num:{num}")
        # data_size = batch_size * num * length
        # levels = [data['level_num'] for data in metadata]
        # discount = torch.tensor(
        #     [self._level_trans ** (9 - level) for level in levels], device=self._device
        # ).unsqueeze(-1)

        # if invert_index is None:
        #     invert_index = torch.range(0, batch_size * num - 1, dtype=torch.int, device=self._device)
        #     invert_index = (invert_index % num).reshape(batch_size, num)
        # if self.training and data_size > self._max_embed_size:
        #     # sample candidate
        #     sample_num = int(self._max_embed_size / length / batch_size)
        #     candidates, gold, offsets, relations = candidate_sample(
        #         candidates, gold, offsets, relations, sample_num
        #     )

        # embedded_q_a_pairs = self._text_field_embedder(
        #     candidates, num_wrapping_dims=1
        # )
        embedded_q_a_pairs = self._encode_q_a_pairs(candidates, offsets, relations)

        # batch_pairs, num_tokens,  = embedded_q_a_pairs.size()

        if self._dropout:
            embedded_q_a_pairs = self._dropout(embedded_q_a_pairs)

        # q_a_pairs_mask = get_text_field_mask(candidates).long()
        # encoder_outputs = self._encoder(
        #     embedded_q_a_pairs.view(batch_size * num_pairs, num_tokens, -1),
        #     q_a_pairs_mask.view(batch_size * num_pairs, num_tokens)
        # )
        # encoder_outputs = self._text_field_embedder.token_embedder_tokens.transformer_model.pooler(
        #     embedded_q_a_pairs
        # )
        encoder_outputs = embedded_q_a_pairs.view(
            batch_size, int(embedded_q_a_pairs.shape[0] / batch_size), -1
        )
        scores = self._denseLayer(encoder_outputs).squeeze(-1)
        # print(f"score:{scores.size()}")
        probs = torch.sigmoid(scores)
        output_dict = {"logits": scores, "probs": probs}
        # if self.training or invert_index is None:
            # output_dict = {
            #     "logits": scores,
            #     "probs": probs
            # }
        # else:
        #     output_dict = {
        #         "logits": self.invert(scores, invert_index),
        #         "probs": self.invert(probs, invert_index)
        #     }
        # output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        # if gold is not None:
        #     # Get loss
        #     loss = neuralNDCG(scores.double(), gold.double())
        #     # loss = binary_list_net_loss(scores.double(), gold.double()) * discount
        #     output_dict["loss"] = loss  # loss.masked_fill(~label_mask, 0).sum() / label_mask.sum()

        #     # Calculate metrics data format
        #     scores = output_dict["logits"]
        #     probs = output_dict["probs"]
        #     # if not self.training and invert_index is not None:
        #     #     gold = self.invert(gold, invert_index)

        #     output_dict.update(self.compute_metric(scores, probs, gold, metadata))

        if gold is not None:
            label_mask = (gold != -1)
            self._mrr(probs, gold, label_mask)
            self._ndcg(scores.float(), gold.float(), label_mask)

            # loss = listNet_loss(scores, labels)
            loss = neuralNDCG(scores.double(), gold.double())
            # print(f"scores:{scores}")
            # print(f"loss:{loss}")
            # probs = probs.view(-1)
            # gold = gold.view(-1)
            # label_mask = label_mask.view(-1)
            # self._auc(probs, labels.ge(0.5).long(), label_mask)

            output_dict["loss"] = loss  # loss.masked_fill(~label_mask, 0).sum() / label_mask.sum()

        return output_dict

    def split_embedding(self, candidates: TextFieldTensors):
        embedded_query_algebra_pairs = []
        batch_size, num, length = candidates['tokens']['token_ids'].shape
        step_num = num / self._max_embed_num
        if step_num - int(step_num) > 1e-3:
            step_num += 1
        for i in range(int(step_num)):
            s_id = i * self._max_embed_num
            e_id = (i + 1) * self._max_embed_num
            embedded_query_algebra_pairs.append(
                self._text_field_embedder(
                    candidate_split(candidates, s_id, e_id), num_wrapping_dims=1
                )
            )
        embedded_query_algebra_pairs = torch.cat(embedded_query_algebra_pairs, dim=1)
        return embedded_query_algebra_pairs

    def _encode_q_a_pairs(self, candidates, offsets, relations):
        embedded_q_a_pairs = self._text_field_embedder(
            candidates, num_wrapping_dims=1
        )
        batch_size, can_num, tok_num, dim = embedded_q_a_pairs.shape
        embedded_q_a_pairs = embedded_q_a_pairs.view(
            batch_size * can_num, tok_num, dim
        )
        offsets = offsets.view(
            batch_size * can_num, -1, 2
        )

        (
            embedded_q_a_pairs,
            embedded_q_a_mask,
        ) = vec_utils.batched_span_select(embedded_q_a_pairs, offsets)
        embedded_q_a_pairs = masked_mean(
            embedded_q_a_pairs,
            embedded_q_a_mask.unsqueeze(-1),
            dim=-2,
        )

        # batch_size, num, tok_num, emb_dim = embedded_q_a_pairs.shape
        # embedded_q_a_pairs = embedded_q_a_pairs.reshape(-1, tok_num, emb_dim)

        # Prepare relation
        tok_num = embedded_q_a_pairs.shape[1]
        assert tok_num == relations.shape[-1]
        relations = relations.reshape(-1, tok_num, tok_num)
        relation_mask = (relations >= 0).float()  # TODO: fixme
        torch.abs(relations, out=relations)

        # RAT-layers
        embedded_q_a_pairs = self._emb_to_ff_dim(embedded_q_a_pairs)
        embedded_q_a_pairs = self._schema_encoder(
            embedded_q_a_pairs, relations.long(), relation_mask
        )
        embedded_q_a_pairs = self._pooler(embedded_q_a_pairs[:, 0])

        return embedded_q_a_pairs

    @staticmethod
    def invert(score, invert_index):
        return annu.batched_index_select(
            score.detach().unsqueeze(-1), invert_index.detach()
        ).squeeze(-1)

    def compute_metric(self, scores, probs, gold, metadata):
        batch_size, num_pairs = scores.shape
        padding = (0, self._max_candidate_num - num_pairs)
        probs = torch.nn.functional.pad(probs, padding, value=-1)
        gold = torch.nn.functional.pad(gold.int(), padding, value=-1).int()
        scores = torch.nn.functional.pad(scores, padding, value=-1)
        # Metrics calculation
        label_mask = (gold != -1)
        # levels = [data['level_num'] for data in metadata]
        # self._mrr(probs, gold, label_mask)
        self._ndcg(scores, gold, label_mask)
        # for i, level in enumerate(levels):
        #     # self._ndcg_level[level](
        #     #     scores[i].unsqueeze(0), gold[i].unsqueeze(0), label_mask[i].unsqueeze(0)
        #     # )
        #     if level == self._level_num - 1:
        #         acc_mask = torch.eq(scores[i], scores[i].max())
        #         acc = (acc_mask & gold.bool()).sum()
        #         self._acc(acc > 0)
        probs = probs.view(-1)
        gold = gold.view(-1)
        label_mask = label_mask.view(-1)
        # self._auc(probs, gold.ge(0.5).long(), label_mask)

        # Detail output for analysis
        res = {}
        if self.output_detail:
            if gold is None:
                return res
            gold = gold.view(batch_size, -1)
            res['ndcg'] = NDCG(scores.float(), gold.float())
            res['pred_target'] = []
            if not self.training:
                with torch.no_grad():
                    indices = torch.argsort(scores, -1, True)
                for b, index in enumerate(indices):
                    print(metadata[b]['candidates'])
                    print(index)
                    candidate_list = []
                    for i in index:
                        candidate_list.append(

                            ' '.join(metadata[b]['candidates'][i])
                        )
                        if gold[b][i]:
                            candidate_list[-1] = candidate_list[-1]
                            break
                    res['pred_target'].append(candidate_list)
        return res

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        # for i in range(0, self._level_num, 1 if reset else 3):
        #     ndcg_level = self._ndcg_level[i].get_metric(reset)
        #     if ndcg_level > 0.0001:
        #         metrics[f"ndcg_{i}"] = ndcg_level

        # metrics["auc"] = self._auc.get_metric(reset)
        # metrics["acc"] = self._acc.get_metric(reset)
        metrics["mrr"] = self._mrr.get_metric(reset)
        metrics["ndcg"] = self._ndcg.get_metric(reset)
        return metrics

    default_predictor = "document_ranker"


def candidate_split(candidates: TextFieldTensors, s_id, e_id):
    candidate_slice = {'tokens': {}}
    tokens = candidate_slice['tokens']
    tokens['token_ids'] = candidates['tokens']['token_ids'][:, s_id: e_id, :]
    tokens['mask'] = candidates['tokens']['mask'][:, s_id: e_id, :]
    tokens['typer_ids'] = candidates['tokens']['type_ids'][:, s_id: e_id, :]
    return candidate_slice


def candidate_sample(
        candidates: TextFieldTensors,
        gold: torch.IntTensor,
        offsets: torch.IntTensor,
        relations: torch.IntTensor,
        num
):
    candidates_num = candidates['tokens']['token_ids'].shape[1]
    sample = random.sample(list(range(candidates_num)), num)
    with torch.no_grad():
        # assert gold.sum() > 0, "No gold label available"
        if gold.sum() <= 0:
            gold[0][0] = 1
        while gold is not None and gold[:, sample].sum() == 0:
            sample = random.sample(list(range(candidates_num)), num)

    candidates['tokens']['token_ids'] = candidates['tokens']['token_ids'][:, sample, :]
    candidates['tokens']['mask'] = candidates['tokens']['mask'][:, sample, :]
    candidates['tokens']['type_ids'] = candidates['tokens']['type_ids'][:, sample, :]

    if gold is not None:
        gold = gold[:, sample]

    offsets = offsets[:, sample, :]
    relations = relations[:, sample, :]
    return candidates, gold, offsets, relations
