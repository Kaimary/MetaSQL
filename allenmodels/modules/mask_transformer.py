# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 16:41
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : mask_transformer.py
# @Software: PyCharm
from typing import Optional, Union, Callable
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
import torch
from overrides import overrides
from torch import Tensor
from torch.nn import functional as F, MultiheadAttention, ModuleList
from torch.nn.modules.transformer import TransformerEncoderLayer


@Seq2SeqEncoder.register("mask_transformer")
class MaskTransformerEncoder(Seq2SeqEncoder):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
    """

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def is_bidirectional(self):
        return False

    def __init__(self, num_layers, norm=None, batch_first=True, **kwargs):
        super(MaskTransformerEncoder, self).__init__()
        self.layers = ModuleList([
            TransformerEncoderLayer(batch_first=batch_first, **kwargs) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

# class MaskTransformer(Seq2SeqEncoder):
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.
#
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of the intermediate layer, can be a string
#             ("relu" or "gelu") or a unary callable. Default: relu
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
#         norm_first: if ``True``, layer norm is done prior to attention and feedforward
#             operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
#
#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = encoder_layer(src)
#
#     Alternatively, when ``batch_first`` is ``True``:
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
#         >>> src = torch.rand(32, 10, 512)
#         >>> out = encoder_layer(src)
#     """
#     __constants__ = ['batch_first', 'norm_first']
#
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(MaskTransformer, self).__init__()
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
#
#         self.norm_first = norm_first
#         self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#
#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation
#
#     @overrides
#     def get_input_dim(self) -> int:
#         return self._input_dim
#
#     @overrides
#     def get_output_dim(self) -> int:
#         return self._input_dim
#
#     @overrides
#     def is_bidirectional(self):
#         return False
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(MaskTransformer, self).__setstate__(state)
#
#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
#
#         x = src
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
#             x = x + self._ff_block(self.norm2(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
#             x = self.norm2(x + self._ff_block(x))
#
#         return x
#
#     # self-attention block
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)
#
#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)
#
#
# def _get_activation_fn(activation):
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return F.gelu
#
#     raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# from typing import Tuple
# from torch.nn.init import constant_, xavier_normal_
# from torch.nn.init import xavier_uniform_
# from torch.nn.parameter import Parameter
# from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
# from torch.nn.modules.module import Module
# class MultiheadAttention(Module):
#     r"""Allows the model to jointly attend to information
#     from different representation subspaces as described in the paper:
#     `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
#
#     Multi-Head Attention is defined as:
#
#     .. math::
#         \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
#
#     where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
#
#     Args:
#         embed_dim: Total dimension of the model.
#         num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
#             across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
#         dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
#         bias: If specified, adds bias to input / output projection layers. Default: ``True``.
#         add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
#         add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
#             Default: ``False``.
#         kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
#         vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
#
#     Examples::
#
#         >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
#         >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
#     """
#     __constants__ = ['batch_first']
#     bias_k: Optional[torch.Tensor]
#     bias_v: Optional[torch.Tensor]
#
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(MultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
#
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#
#         if self._qkv_same_embed_dim is False:
#             self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
#             self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
#             self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
#             self.register_parameter('in_proj_weight', None)
#         else:
#             self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)
#
#         if bias:
#             self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
#
#         if add_bias_kv:
#             self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#             self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         else:
#             self.bias_k = self.bias_v = None
#
#         self.add_zero_attn = add_zero_attn
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         if self._qkv_same_embed_dim:
#             xavier_uniform_(self.in_proj_weight)
#         else:
#             xavier_uniform_(self.q_proj_weight)
#             xavier_uniform_(self.k_proj_weight)
#             xavier_uniform_(self.v_proj_weight)
#
#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             xavier_normal_(self.bias_v)
#
#     def __setstate__(self, state):
#         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#         if '_qkv_same_embed_dim' not in state:
#             state['_qkv_same_embed_dim'] = True
#
#         super(MultiheadAttention, self).__setstate__(state)
#
#     def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
#                 need_weights: bool = True, attn_mask: Optional[Tensor] = None,
#                 average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
#         r"""
#     Args:
#         query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
#             or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
#             :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
#             Queries are compared against key-value pairs to produce the output.
#             See "Attention Is All You Need" for more details.
#         key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
#             or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
#             :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
#             See "Attention Is All You Need" for more details.
#         value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
#             ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
#             sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
#             See "Attention Is All You Need" for more details.
#         key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
#             to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
#             Binary and byte masks are supported.
#             For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
#             the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
#             value will be ignored.
#         need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
#             Default: ``True``.
#         attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
#             :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
#             :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
#             broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
#             Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
#             corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
#             corresponding position is not allowed to attend. For a float mask, the mask values will be added to
#             the attention weight.
#         average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
#             heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
#             effect when ``need_weights=True.``. Default: True (i.e. average weights across heads)
#
#     Outputs:
#         - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
#           :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
#           where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
#           embedding dimension ``embed_dim``.
#         - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
#           returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
#           :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
#           :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
#           head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
#
#         .. note::
#             `batch_first` argument is ignored for unbatched inputs.
#         """
#         is_batched = query.dim() == 3
#         if self.batch_first and is_batched:
#             query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
#
#         if not self._qkv_same_embed_dim:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, use_separate_proj_weight=True,
#                 q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
#                 v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
#         else:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, average_attn_weights=average_attn_weights)
#         if self.batch_first and is_batched:
#             return attn_output.transpose(1, 0), attn_output_weights
#         else:
#             return attn_output, attn_output_weights
