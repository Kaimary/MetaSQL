#coding=utf8
import torch
import torch.nn as nn
from model.model_utils import Registrable, PoolingFunction
from model.encoder.graph_encoder import *
from model.decoder.sql_parser import *
from model.decoder.multi_label_cls import *

@Registrable.register('text2sql')
class Text2SQL(nn.Module):
    def __init__(self, args, transition_system):
        super(Text2SQL, self).__init__()
        self.encoder = Registrable.by_name('encoder_text2sql')(args)
        self.encoder2decoder = PoolingFunction(args.gnn_hidden_size, args.lstm_hidden_size, method='attentive-pooling')
        self.decoder = Registrable.by_name('decoder_tranx')(args, transition_system)

    def forward(self, batch):
        """ This function is used during training, which returns the entire training loss
        """
        encodings, mask, gp_loss = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        loss = self.decoder.score(encodings, mask, h0, batch)
        return loss, gp_loss

    def parse(self, batch, beam_size):
        """ This function is used for decoding, which returns a batch of [DecodeHypothesis()] * beam_size
        """
        encodings, mask = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        hyps = []
        for i in range(len(batch)):
            """ table_mappings and column_mappings are used to map original database ids to local ids,
            while reverse_mappings perform the opposite function, mapping local ids to database ids
            """
            hyps.append(self.decoder.parse(encodings[i:i+1], mask[i:i+1], h0[i:i+1], batch, beam_size))
        return hyps

    def pad_embedding_grad_zero(self, index=None):
        """ For glove.42B.300d word vectors, gradients for <pad> symbol is always 0;
        Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.encoder.input_layer.pad_embedding_grad_zero(index)

@Registrable.register('text2sql_origin')
class Text2SQL1(nn.Module):
    def __init__(self, args, transition_system):
        super(Text2SQL1, self).__init__()
        self.encoder = Registrable.by_name('encoder_text2sql_origin')(args)
        self.encoder2decoder = PoolingFunction(args.gnn_hidden_size, args.lstm_hidden_size, method='attentive-pooling')
        self.decoder = Registrable.by_name('decoder_tranx_origin')(args, transition_system)

    def forward(self, batch):
        """ This function is used during training, which returns the entire training loss
        """
        encodings, mask, gp_loss = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        loss = self.decoder.score(encodings, mask, h0, batch)
        return loss, gp_loss

    def parse(self, batch, beam_size):
        """ This function is used for decoding, which returns a batch of [DecodeHypothesis()] * beam_size
        """
        encodings, mask = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        hyps = []
        for i in range(len(batch)):
            """ table_mappings and column_mappings are used to map original database ids to local ids,
            while reverse_mappings perform the opposite function, mapping local ids to database ids
            """
            hyps.append(self.decoder.parse(encodings[i:i+1], mask[i:i+1], h0[i:i+1], batch, beam_size))
        return hyps

    def pad_embedding_grad_zero(self, index=None):
        """ For glove.42B.300d word vectors, gradients for <pad> symbol is always 0;
        Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.encoder.input_layer.pad_embedding_grad_zero(index)

@Registrable.register('text2label')
class Text2Label(nn.Module):
    def __init__(self, args):
        super(Text2Label, self).__init__()
        self.encoder = Registrable.by_name('encoder_text2sql')(args)
        self.encoder2decoder = PoolingFunction(args.gnn_hidden_size, args.lstm_hidden_size, method='attentive-pooling')
        self.decoder = Registrable.by_name('multi_label_cls')(args)

    def forward(self, batch):
        """ This function is used during training, which returns the entire training loss
        """
        encodings, mask, gp_loss = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        loss = self.decoder(h0, batch)
        return loss, gp_loss

    def parse(self, batch):
        """ This function is used for decoding, which returns a batch of [DecodeHypothesis()] * beam_size
        """
        encodings, mask = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        tp, fp, tn, fn = self.decoder.inference(h0, batch)
        return tp, fp, tn, fn

    def test(self, batch):
        """ This function is used for label prediction.
        """
        encodings, mask = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=mask)
        logits = self.decoder.test(h0)
        return logits
    
    def pad_embedding_grad_zero(self, index=None):
        """ For glove.42B.300d word vectors, gradients for <pad> symbol is always 0;
        Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.encoder.input_layer.pad_embedding_grad_zero(index)
