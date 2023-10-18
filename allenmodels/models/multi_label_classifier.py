import torch
from typing import Optional, Dict
from overrides import overrides
import allennlp.nn.util as util
from allennlp.nn.util import masked_mean
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator

from allenmodels.models.model_util import vec_utils
from allenmodels.models.metrics.multilabel_f1 import MultiLabelF1Measure
eps = 1e-8

@Model.register('metadata_classifier')
class metadataClassifierModel(Model):
    def __init__(self, 
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder : Seq2VecEncoder,
            schema_encoder: Seq2SeqEncoder,
            classifier_feedforward: FeedForward,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = RegularizerApplicator())->None:

        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.q_emb_dim = self.text_field_embedder.get_output_dim()
        self._action_dim = 512
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.encoder = encoder
        self._pooler = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self._action_dim,
                out_features=self._action_dim,
            ),
            torch.nn.Tanh()
        )
        self._schema_encoder = schema_encoder
        self._emb_to_action_dim = torch.nn.Linear(
            in_features=self.q_emb_dim,
            out_features=self._action_dim,
        )

        self.classifier_feedforward = classifier_feedforward
        pos_weight = torch.ones([24])
        pos_weight[1] = 10
        pos_weight[6] = 10
        pos_weight[10] = 10
        pos_weight[12] = 10
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.loss = torch.nn.MultiLabelMarginLoss(reduction='sum')
        self.f1 = MultiLabelF1Measure()
        initializer(self)
        
    def forward(self, 
            enc,
            lengths,
            offsets,
            relation,
            labels=None)->Dict[str, torch.Tensor]:

        (
            _,
            _,
            embedded_utterance,
            utterance_mask,
        ) = self._encode_utt_schema(enc, offsets, relation, lengths)

        # batch_size, utterance_length, _ = embedded_utterance.shape
        # embedded_text = self.text_field_embedder(tokens)
        # embedded_type = self.text_field_embedder(types)
        # encoder_input = embedded_text.add(embedded_type)
        # torch.mean(encoder_input)
        # mask = util.get_text_field_mask(tokens)
        encoder_input = self._pooler(embedded_utterance[:, 0])

        # encoder_input = torch.cat([encoded_text, encoded_type], 1)
        logits = self.classifier_feedforward(encoder_input)
        probabilities = torch.nn.functional.sigmoid(logits)

        output_dict = {'logits': logits, 'probabilities': probabilities}
        if labels is not None:
            loss = self.loss(logits + eps, labels.float())
            #loss = self.loss(logits, labels.squeeze(-1).long())
            output_dict['loss'] = loss

            predictions = (logits.data > 0.0).long()
            label_data = labels.squeeze(-1).data.long()
            self.f1(predictions, label_data)         

        return output_dict

    def emb_q(self, enc):
        pad_dim = enc["bert"]["mask"].size(-1)
        if pad_dim > 512:
            for key in enc["bert"].keys():
                enc["bert"][key] = enc["bert"][key][:, :512]

            embedded_utterance_schema = self.text_field_embedder(enc)
        else:
            embedded_utterance_schema = self.text_field_embedder(enc)

        return embedded_utterance_schema

    def _encode_utt_schema(self, enc, offsets, relation, lengths):
        embedded_utterance_schema = self.emb_q(enc)

        (
            embedded_utterance_schema,
            embedded_utterance_schema_mask,
        ) = vec_utils.batched_span_select(embedded_utterance_schema, offsets)
        embedded_utterance_schema = masked_mean(
            embedded_utterance_schema,
            embedded_utterance_schema_mask.unsqueeze(-1),
            dim=-2,
        )

        relation_mask = (relation >= 0).float()  # TODO: fixme
        torch.abs(relation, out=relation)
        embedded_utterance_schema = self._emb_to_action_dim(embedded_utterance_schema)
        enriched_utterance_schema = self._schema_encoder(
            embedded_utterance_schema, relation.long(), relation_mask
        )

        utterance_schema, utterance_schema_mask = vec_utils.batched_span_select(
            enriched_utterance_schema, lengths
        )
        utterance, schema = torch.split(utterance_schema, 1, dim=1)
        utterance_mask, schema_mask = torch.split(utterance_schema_mask, 1, dim=1)
        utterance_mask = torch.squeeze(utterance_mask, 1)
        schema_mask = torch.squeeze(schema_mask, 1)
        embedded_utterance = torch.squeeze(utterance, 1)
        schema = torch.squeeze(schema, 1)
        return schema, schema_mask, embedded_utterance, utterance_mask

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Does a simple argmax over the probabilities, converts index to string label,
            and add label key to the dictionary with the result """
        classes = []
        probs = []
        predictions = output_dict["probabilities"].cpu().data.numpy()
        # print(f"predictions:{predictions[0]}")
        # Its a multilabel classification so, need to iterate through all of the labels
        for i, p in enumerate(predictions[0]):
            # if p > 0.5:
            label = self.vocab.get_token_from_index(i, namespace="labels")
            classes.append(label)
            probs.append(f"{p:.2f}")
        output_dict['labels'] = [', '.join(classes)]
        output_dict['probs'] = [', '.join(probs)]
        # print(output_dict['labels'])
        return output_dict
    
    @overrides
    def get_metrics(self, reset: bool = False)->Dict[str, float]:
        precision, recall , f1 = self.f1.get_metric(reset)
        return {'precision': precision, 'recall': recall, 'f1': f1}
