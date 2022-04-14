import inspect
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from typing import Callable, List, Set, Tuple
from paddle.nn import Layer
from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.fnet.modeling import ACT2FN

__all__ = [
    "LongformerEmbeddings",
    "LongformerSelfAttention",
    "LongformerSelfOutput",
    "LongformerAttention",
    "LongformerIntermediate",
    "LongformerOutput",
    "LongformerLayer",
    "LongformerEncoder",
    "LongformerPooler",
    "LongformerLMHead",
    "LongformerPreTrainedModel",
    "LongformerModel",
    "LongformerClassificationHead",
    "LongformerForSequenceClassification",
    "LongformerForTokenClassification",
    "LongformerForMaskedLM",
    "LongformerForQuestionAnswering",
    "LongformerForMultipleChoice"
]


class LongformerEmbeddings(Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    
    def __init__(
            self,
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            pad_token_id,
    ):
        super(LongformerEmbeddings, self).__init__()
        # self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(max_position_embeddings).expand((1, -1)))
        
        self.padding_idx = pad_token_id
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size, padding_idx=self.padding_idx
        )
    
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx)
        
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]
        
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: paddle.Tensor x:

        Returns: paddle.Tensor
        """
        mask = paddle.cast(input_ids != padding_idx, dtype=paddle.int64)
        incremental_indices = paddle.cast(paddle.cumsum(mask, axis=1), dtype=mask.dtype) * mask
        return paddle.cast(incremental_indices, dtype=paddle.int64) + padding_idx


class LongformerSelfAttention(Layer):
    def __init__(
            self,
            layer_id,
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window,
    ):
        super(LongformerSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        self.num_heads = num_attention_heads
        self.head_dim = int(hidden_size / num_attention_heads)
        self.embed_dim = hidden_size
        
        self.query = nn.Linear(hidden_size, self.embed_dim)
        self.key = nn.Linear(hidden_size, self.embed_dim)
        self.value = nn.Linear(hidden_size, self.embed_dim)
        
        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(hidden_size, self.embed_dim)
        self.key_global = nn.Linear(hidden_size, self.embed_dim)
        self.value_global = nn.Linear(hidden_size, self.embed_dim)
        
        self.dropout = attention_probs_dropout_prob
        
        self.layer_id = layer_id
        attention_window = attention_window[self.layer_id]
        
        assert (
                attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
                attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"
        
        self.one_sided_attn_window_size = attention_window // 2
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            is_index_masked=None,
            is_index_global_attn=None,
            is_global_attn=None,
            output_attentions=False,
    ):
        """
                [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
                *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

                The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

                    - -10000: no attention
                    - 0: local attention
                    - +10000: global attention
        """
        hidden_states = paddle.transpose(hidden_states, perm=[1, 0, 2])
        
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        
        seq_len, batch_size, embed_dim = hidden_states.shape
        assert (
                embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"
        
        # normalize query
        query_vectors /= math.sqrt(self.head_dim)
        
        query_vectors = query_vectors.reshape(shape=[seq_len, batch_size, self.num_heads, self.head_dim]).transpose(
            perm=[1, 0, 2, 3])
        key_vectors = key_vectors.reshape(shape=[seq_len, batch_size, self.num_heads, self.head_dim]).transpose(
            perm=[1, 0, 2, 3])
        
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )
        
        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        
        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = paddle.cast(remove_from_windowed_attention_mask, dtype=query_vectors.dtype)
        float_mask = masked_fill(float_mask, remove_from_windowed_attention_mask, -10000.0)
        
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            paddle.ones(shape=float_mask.shape, dtype=float_mask.dtype), float_mask, self.one_sided_attn_window_size
        )
        
        # pad local attention probs
        attn_scores += diagonal_mask
        
        assert list(attn_scores.shape) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.shape}"
        
        # compute local attention probs from global attention keys and contact over window axis
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key
            
            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = paddle.concat(x=[global_key_attn_scores, attn_scores], axis=-1)
            
            # free memory
            del global_key_attn_scores
        
        # use fp32 for numerical stability
        attn_probs = F.softmax(attn_scores, axis=-1, dtype=paddle.float32)
        
        if layer_head_mask is not None:
            assert layer_head_mask.shape == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.shape}"
            attn_probs = layer_head_mask.reshape([1, 1, -1, 1]) * attn_probs

        attn_probs = masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = paddle.cast(attn_probs, dtype=attn_scores.dtype)
        
        # free memory
        del attn_scores
        
        # apply dropout
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        value_vectors = value_vectors.reshape(shape=[seq_len, batch_size, self.num_heads, self.head_dim]).transpose(
            perm=[1, 0, 2, 3])
        
        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )
        
        assert attn_output.shape == [batch_size, seq_len, self.num_heads, self.head_dim], "Unexpected size"
        attn_output = attn_output.transpose(perm=[1, 0, 2, 3]).reshape(shape=[seq_len, batch_size, embed_dim])
        
        # compute value for global attention and overwrite to attention output
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )
            
            # get only non zero global attn output
            nonzero_global_attn_output = paddle.to_tensor(
                global_attn_output.numpy()[is_local_index_global_attn_nonzero[0], :,
                is_local_index_global_attn_nonzero[1]],
                stop_gradient=global_attn_output.stop_gradient,
                dtype=global_attn_output.dtype
            )
            
            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.reshape(
                shape=[len(is_local_index_global_attn_nonzero[0]), -1]
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0
        
        outputs = (attn_output.transpose(perm=[1, 0, 2]),)
        
        if output_attentions:
            outputs += (attn_probs,)
        
        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs
    
    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = F.pad(hidden_states_padded, padding)
        # padding value is not important because it will be overwritten
        hidden_states_padded = paddle.reshape(hidden_states_padded,
                                              shape=[*hidden_states_padded.shape[:-2], hidden_states_padded.shape[-1],
                                                     hidden_states_padded.shape[-2]])
        return hidden_states_padded
    
    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.shape
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = F.pad(chunked_hidden_states.transpose([0, 1, 3, 2]),
                                      (0, 0, 0, window_overlap + 1)).transpose([0, 1, 3, 2])
        chunked_hidden_states = chunked_hidden_states.reshape(
            shape=[total_num_heads, num_chunks, -1]
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
                                :, :, :-window_overlap
                                ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.reshape(
            shape=[total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim]
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states
    
    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # transpose [batch_size,seq_length,hidden_dim] -> unsqueeze [batch_size,hidden_dim,seqlen,1] -> unfold [batch_size, hidden_dim*2 * window_overlap, num_output_chunks]
        # transpose [batch_size,num_output_chunks,hidden_dim*2*window_overlap] -> reshape [batch_size,num_output_chunks,hidden_dim, 2*window_overlap] -> transpose [batch_size,num_output_chunks,2*window_overlap,hidden_dim]
        batch_size, seq_length, hidden_dim = hidden_states.shape
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1
        
        hidden_states = (
            F.unfold(
                hidden_states.transpose(perm=[0, 2, 1]).unsqueeze(-1),
                kernel_sizes=[window_overlap * 2, 1],
                strides=[window_overlap, 1],
            )
                .transpose(perm=[0, 2, 1])
                .reshape(shape=[batch_size, num_output_chunks, hidden_dim, window_overlap * 2])
                .transpose(perm=[0, 1, 3, 2])
        )
        
        return hidden_states
    
    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> paddle.Tensor:
        beginning_mask_2d = paddle.flip(
            paddle.tril(paddle.ones(shape=[affected_seq_len, affected_seq_len + 1], dtype=input_tensor.dtype)), axis=0)
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = paddle.flip(beginning_mask, axis=[1, 3])
        
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = paddle.expand(beginning_mask, shape=beginning_input.shape)
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = masked_fill(
            beginning_input, beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = paddle.expand(ending_mask, shape=ending_input.shape)
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):] = masked_fill(
            ending_input, ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
    
    def _sliding_chunks_query_key_matmul(self, query: paddle.Tensor, key: paddle.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        assert (
                seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.shape == key.shape
        
        chunks_count = seq_len // window_overlap - 1
        
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(perm=[0, 2, 1, 3]).reshape(shape=[batch_size * num_heads, seq_len, head_dim])
        key = key.transpose(perm=[0, 2, 1, 3]).reshape(shape=[batch_size * num_heads, seq_len, head_dim])
        
        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)
        
        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = paddle.einsum("bcxd,bcyd->bcxy", query, key)  # multiply
        
        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )
        
        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        
        diagonal_attention_scores = paddle.empty(
            shape=(batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1),
            dtype=diagonal_chunked_attention_scores.dtype
        )
        
        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
                                                                ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
                                                               ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                               ]
        
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
                                                                              ]
        
        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.reshape(
            shape=[batch_size, num_heads, seq_len, 2 * window_overlap + 1]
        ).transpose(perm=[0, 2, 1, 3])
        
        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores
    
    def _sliding_chunks_matmul_attn_probs_value(
            self, attn_probs: paddle.Tensor, value: paddle.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.shape
        
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.shape[:3] == value.shape[:3]
        assert attn_probs.shape[3] == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap
        
        chunked_attn_probs = attn_probs.transpose(perm=[0, 2, 1, 3]).reshape(
            shape=[batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1]
        )
        
        # group batch_size and num_heads dimensions into one
        value = value.transpose(perm=[0, 2, 1, 3]).reshape(shape=[batch_size * num_heads, seq_len, head_dim])
        
        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = paddle.squeeze(F.pad(paddle.unsqueeze(value, axis=0),
                                            (0, 0, window_overlap, window_overlap),
                                            value=-1), axis=0)
        
        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value = paddle.nn.functional.unfold(padded_value.transpose(perm=[0, 2, 1]).unsqueeze(-1),
                                                    kernel_sizes=[window_overlap * 3, 1],
                                                    strides=[window_overlap, 1]).transpose(perm=[0, 2, 1]).reshape(
            shape=[batch_size * num_heads, chunks_count, head_dim, window_overlap * 3]).transpose(perm=[0, 1, 3, 2])
        
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        
        context = paddle.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        return context.reshape(shape=[batch_size, num_heads, seq_len, head_dim]).transpose(perm=[0, 2, 1, 3])
    
    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = paddle.cast(is_index_global_attn, paddle.int64).sum(axis=1)
        
        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()
        
        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        is_index_global_attn_nonzero = tuple([item.squeeze() for item in is_index_global_attn_nonzero])
        
        # helper variable
        is_local_index_global_attn = paddle.arange(max_num_global_attn_indices) < num_global_attn_indices.unsqueeze(
            axis=-1)
        
        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)
        is_local_index_global_attn_nonzero = tuple([item.squeeze() for item in is_local_index_global_attn_nonzero])
        
        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        is_local_index_no_global_attn_nonzero = tuple(
            [item.squeeze() for item in is_local_index_no_global_attn_nonzero])
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )
    
    def _concat_with_global_key_attn_probs(
            self,
            key_vectors,
            query_vectors,
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]
        
        # create only global key vectors
        key_vectors_only_global = paddle.zeros(
            shape=[batch_size, max_num_global_attn_indices[0], self.num_heads, self.head_dim], dtype=key_vectors.dtype
        )
        
        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]
        
        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = paddle.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)
        
        attn_probs_from_global_key[
        is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]
        ] = -10000.0
        
        return attn_probs_from_global_key
    
    def _compute_attn_output_with_global_indices(
            self,
            value_vectors,
            attn_probs,
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]
        
        # cut local attn probs to global only
        max_num_global_attn_indices = max_num_global_attn_indices.tolist()
        attn_probs_only_global = paddle.slice(attn_probs, axes=[-1], starts=[0], ends=max_num_global_attn_indices)
        
        # get value vectors for global only
        value_vectors_only_global = paddle.zeros(
            shape=[batch_size, max_num_global_attn_indices[0], self.num_heads, self.head_dim],
            dtype=value_vectors.dtype)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]
        
        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = paddle.matmul(
            attn_probs_only_global.transpose(perm=[0, 2, 1, 3]).clone(),
            value_vectors_only_global.transpose(perm=[0, 2, 1, 3]).clone()
        ).transpose(perm=[0, 2, 1, 3])
        
        # reshape attn probs
        attn_probs_without_global = paddle.slice(attn_probs, axes=[-1],
                                                 starts=max_num_global_attn_indices, ends=[attn_probs.shape[-1]])
        
        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global
    
    def _compute_global_attn_output_from_hidden(
            self,
            hidden_states,
            max_num_global_attn_indices,
            layer_head_mask,
            is_local_index_global_attn_nonzero,
            is_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
            is_index_masked,
    ):
        seq_len, batch_size = hidden_states.shape[:2]
        max_num_global_attn_indices = max_num_global_attn_indices.tolist()
        
        # prepare global hidden states
        global_attn_hidden_states = paddle.zeros(
            shape=[max_num_global_attn_indices[0], batch_size, self.embed_dim], dtype=hidden_states.dtype)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[
            is_index_global_attn_nonzero[::-1]
        ]
        
        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)
        
        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)
        # (max_num_global_attn_indices, batch_size, hidden_dim)
        
        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global
                .reshape(shape=[max_num_global_attn_indices[0], batch_size * self.num_heads, self.head_dim])
                .transpose(perm=[1, 0, 2])
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.reshape(shape=[-1, batch_size * self.num_heads, self.head_dim]).transpose(perm=[1, 0, 2])
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.reshape(shape=[-1, batch_size * self.num_heads, self.head_dim]).transpose(
                perm=[1, 0, 2])
        )  # batch_size * self.num_heads, seq_len, head_dim)
        
        # compute attn scores
        global_attn_scores = paddle.bmm(global_query_vectors_only_global, global_key_vectors.transpose(perm=[0, 2, 1]))
        
        assert list(global_attn_scores.shape) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices[0],
            seq_len,
        ], f"global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices[0], seq_len)}, but is {global_attn_scores.shape}."
        
        global_attn_scores = global_attn_scores.reshape(
            shape=[batch_size, self.num_heads, max_num_global_attn_indices[0], seq_len])
        
        global_attn_scores[
        is_local_index_no_global_attn_nonzero[0], :, is_local_index_no_global_attn_nonzero[1], :
        ] = -10000.0
        
        global_attn_scores = masked_fill(global_attn_scores, is_index_masked[:, None, None, :], -10000.0, )
        
        global_attn_scores = global_attn_scores.reshape(
            shape=[batch_size * self.num_heads, max_num_global_attn_indices[0], seq_len])
        
        # compute global attn probs
        # use fp32 for numerical stability
        global_attn_probs_float = F.softmax(global_attn_scores, axis=-1, dtype=paddle.float32)
        
        # apply layer head masking
        if layer_head_mask is not None:
            assert layer_head_mask.shape[0] == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.shape}"
            global_attn_probs_float = layer_head_mask.reshape([1, -1, 1, 1]) * global_attn_probs_float.reshape(
                shape=[batch_size, self.num_heads, max_num_global_attn_indices[0], seq_len]
            )
            global_attn_probs_float = global_attn_probs_float.reshape(
                shape=[batch_size * self.num_heads, max_num_global_attn_indices[0], seq_len]
            )
        
        global_attn_probs = nn.functional.dropout(
            paddle.cast(global_attn_probs_float, dtype=global_attn_scores.dtype), p=self.dropout, training=self.training
        )
        
        # global attn output
        global_attn_output = paddle.bmm(global_attn_probs, global_value_vectors)
        
        assert list(global_attn_output.shape) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices[0],
            self.head_dim,
        ], f"global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices[0], self.head_dim)}, but is {global_attn_output.shape}."
        
        global_attn_probs = global_attn_probs.reshape(
            shape=[batch_size, self.num_heads, max_num_global_attn_indices[0], seq_len])
        global_attn_output = global_attn_output.reshape(
            shape=[batch_size, self.num_heads, max_num_global_attn_indices[0], self.head_dim]
        )
        return global_attn_output, global_attn_probs


class LongformerSelfOutput(Layer):
    def __init__(
            self,
            hidden_size,
            hidden_dropout_prob,
            layer_norm_eps,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class LongformerAttention(Layer):
    def __init__(
            self,
            layer_id,
            hidden_size,
            hidden_dropout_prob,
            layer_norm_eps,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window,
    ):
        super(LongformerAttention, self).__init__()
        self.self = LongformerSelfAttention(
            layer_id,
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window, )
        self.output = LongformerSelfOutput(
            hidden_size,
            hidden_dropout_prob,
            layer_norm_eps,
        )
        self.pruned_heads = set()
    
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)
        
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            is_index_masked=None,
            is_index_global_attn=None,
            is_global_attn=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs


class LongformerIntermediate(Layer):
    def __init__(
            self,
            hidden_size,
            intermediate_size,
            hidden_act,
    ):
        super(LongformerIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LongformerOutput(Layer):
    def __init__(
            self,
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            layer_norm_eps,
    ):
        super(LongformerOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class LongformerLayer(Layer):
    def __init__(
            self,
            layer_id,
            hidden_size,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            layer_norm_eps,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window,
            chunk_size_feed_forward,
    ):
        super(LongformerLayer, self).__init__()
        self.attention = LongformerAttention(
            layer_id,
            hidden_size,
            hidden_dropout_prob,
            layer_norm_eps,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window, )
        self.intermediate = LongformerIntermediate(
            hidden_size,
            intermediate_size,
            hidden_act, )
        self.output = LongformerOutput(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            layer_norm_eps, )
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            is_index_masked=None,
            is_index_global_attn=None,
            is_global_attn=None,
            output_attentions=False,
    ):
        self_attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]
        
        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        outputs = (layer_output,) + outputs
        return outputs
    
    def ff_chunk(self, attn_output):
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output


class LongformerEncoder(Layer):
    def __init__(
            self,
            hidden_size,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            layer_norm_eps,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window,
            chunk_size_feed_forward,
            num_hidden_layers
    ):
        super().__init__()
        self.layer = nn.LayerList([LongformerLayer(
            layer_id,
            hidden_size,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            layer_norm_eps,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window,
            chunk_size_feed_forward, ) for layer_id in range(num_hidden_layers)])
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            padding_len=0,
            output_attentions=False,
            output_hidden_states=False,
    ):
        
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.numpy().flatten().any().item()
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None  # All local attentions.
        all_global_attentions = () if (output_attentions and is_global_attn) else None
        
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.shape[0] == (
                len(self.layer)
            ), f"The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.shape[0]}."
        for idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)
                
                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # undo padding
        if padding_len > 0:
            # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
            hidden_states = hidden_states[:, :-padding_len]
            if output_hidden_states:
                all_hidden_states = tuple([state[:, :-padding_len] for state in all_hidden_states])
            
            if output_attentions:
                all_attentions = tuple([state[:, :, :-padding_len, :] for state in all_attentions])
        
        return tuple(
            v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None
        )


class LongformerPooler(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LongformerLMHead(Layer):
    """Longformer Head for masked language modeling."""
    
    def __init__(
            self,
            vocab_size,
            hidden_size,
            layer_norm_eps,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.bias = self.bias = self.create_parameter(
            [vocab_size],
            is_bias=True,
            default_initializer=nn.initializer.Constant(value=0))
        self.decoder.bias = self.bias
    
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x
    
    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class LongformerClassificationHead(Layer):
    """Head for sentence-level classification tasks."""
    
    def __init__(
            self,
            hidden_size,
            hidden_dropout_prob,
            num_labels=2
    ):
        super(LongformerClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = paddle.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class LongformerPreTrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Longformer models. It provides Longformer related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "longformer-base-4096": {
            "vocab_size": 50265,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 4098,
            "type_vocab_size": 1,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-05,
            "pad_token_id": 1,
            "sep_token_id": 2,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "attention_probs_dropout_prob": 0.1,
            "attention_window": [
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512
            ],
            "ignore_attention_mask": False,
            "num_attention_heads": 12,
            "chunk_size_feed_forward": 0,
        },
        "longformer-large-4096": {
            "vocab_size": 50265,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 4098,
            "type_vocab_size": 1,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-05,
            "pad_token_id": 1,
            "sep_token_id": 2,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "attention_probs_dropout_prob": 0.1,
            "attention_window": [
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512
            ],
            "ignore_attention_mask": False,
            "num_attention_heads": 16,
            "chunk_size_feed_forward": 0,
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "longformer-base-4096":
                "D:/python/pyprojects/论文复现与实验/models/paddle-longformer-base/model_state.pdparams",
            "longformer-large-4096":
                "https://bj.bcebos.com/paddlenlp/models/transformers/fnet/fnet-large/model_state.pdparams",
        }
    }
    base_model_prefix = "longformer"
    
    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, layer):
        # Initialize the weights.
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else self.longformer.config[
                        "initializer_range"],
                    shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else self.longformer.config[
                        "initializer_range"],
                    shape=layer.weight.shape))
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(
                    paddle.zeros_like(layer.weight[layer._padding_idx]))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))


@register_base_model
class LongformerModel(LongformerPreTrainedModel):
    """
    This class overwrote standard self-attention with longformer self-attention to provide the ability to process
    long sequences following the self-attention approach described in [Longformer:
    the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan.
    Longformer self-attention combines a local (sliding window) and global attention to extend to long documents
    without the O(n^2) increase in memory and compute.
    """
    
    def __init__(
            self,
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
            pad_token_id,
            hidden_size,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            layer_norm_eps,
            initializer_range,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window,
            chunk_size_feed_forward,
            num_hidden_layers,
            sep_token_id,
            bos_token_id,
            eos_token_id,
            ignore_attention_mask,
            is_decoder=False,
            add_pooling_layer=True):
        super(LongformerModel, self).__init__()
        
        if isinstance(attention_window, int):
            assert attention_window % 2 == 0, "`attention_window` has to be an even value"
            assert attention_window > 0, "`attention_window` has to be positive"
            attention_window = [attention_window] * num_hidden_layers  # one value per layer
        else:
            assert len(attention_window) == num_hidden_layers, (
                "`len(attention_window)` should equal `num_hidden_layers`. "
                f"Expected {num_hidden_layers}, given {len(attention_window)}"
            )
        self.attention_window = attention_window
        self.initializer_range = initializer_range
        self.is_decoder = is_decoder
        self.embeddings = LongformerEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            pad_token_id, )
        self.encoder = LongformerEncoder(
            hidden_size,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            layer_norm_eps,
            num_attention_heads,
            attention_probs_dropout_prob,
            attention_window,
            chunk_size_feed_forward,
            num_hidden_layers)
        self.pooler = LongformerPooler(hidden_size) if add_pooling_layer else None
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: state_dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def _pad_to_window_size(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
            pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.attention_window
            if isinstance(self.attention_window, int)
            else max(self.attention_window)
        )
        
        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]
        
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            """
            logger.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`attention_window`: {attention_window}"
            )
            """
            if input_ids is not None:
                input_ids = paddle.squeeze(
                    F.pad(paddle.unsqueeze(input_ids, axis=0), (0, padding_len), value=pad_token_id,
                          data_format="NCL"),
                    axis=0)
            if position_ids is not None:
                position_ids = paddle.squeeze(
                    F.pad(paddle.unsqueeze(position_ids, axis=0), (0, padding_len),
                          value=pad_token_id,
                          data_format="NCL"),
                    axis=0)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config["pad_token_id"],
                    dtype=paddle.int64,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = paddle.concat([inputs_embeds, inputs_embeds_padding], axis=-2)
            
            # no attention on the padding tokens
            attention_mask = paddle.squeeze(
                F.pad(paddle.unsqueeze(attention_mask, axis=0), (0, padding_len),
                      value=False,
                      data_format="NCL"),
                axis=0)
            # pad with token_type_id = 0
            token_type_ids = paddle.squeeze(
                F.pad(paddle.unsqueeze(token_type_ids, axis=0), (0, padding_len),
                      value=0,
                      data_format="NCL"),
                axis=0)
        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds
    
    def _merge_to_attention_mask(self, attention_mask: paddle.Tensor, global_attention_mask: paddle.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask
    
    def get_extended_attention_mask(self, attention_mask: paddle.Tensor, input_shape: Tuple[int]) -> paddle.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`paddle.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `paddle.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config["is_decoder"]:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def create_extended_attention_mask_for_decoder(self, input_shape, attention_mask):
        batch_size, seq_length = input_shape
        seq_ids = paddle.arange(seq_length)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)
        
        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = paddle.concat(
                [
                    paddle.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""

        Returns:

        Examples:
        ```python
        >>> import torch
        >>> from transformers import LongformerModel, LongformerTokenizer

        >>> model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

        >>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  # long input document
        >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

        >>> attention_mask = torch.ones(
        ...     input_ids.shape, dtype=torch.long, device=input_ids.device
        >>> )  # initialize to local attention
        >>> global_attention_mask = torch.zeros(
        ...     input_ids.shape, dtype=torch.long, device=input_ids.device
        >>> )  # initialize to global attention to be deactivated for all tokens
        >>> global_attention_mask[
        ...     :,
        ...     [
        ...         1,
        ...         4,
        ...         21,
        ...     ],
        >>> ] = 1  # Set global attention to random tokens for the sake of this example
        >>> # Usually, set global attention based on the task. For example,
        >>> # classification: the <s> token
        >>> # QA: question tokens
        >>> # LM: potentially on the beginning of sentences and paragraphs
        >>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        >>> sequence_output = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        ```"""
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)
        
        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        
        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config["pad_token_id"],
        )
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: paddle.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
                                                 :, 0, 0, :
                                                 ]
        
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        return (sequence_output, pooled_output) + encoder_outputs[1:]


class LongformerForMaskedLM(LongformerPreTrainedModel):
    """Longformer Model with a `language modeling` head on top."""
    
    def __init__(self, longformer):
        super(LongformerForMaskedLM, self).__init__()
        
        self.longformer = longformer
        self.longformer.pooler = None  # add_pooling_layer=False
        self.lm_head = LongformerLMHead(
            longformer.config["vocab_size"],
            longformer.config["hidden_size"],
            longformer.config["layer_norm_eps"],
        )
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels (`paddle.Tensor, dtype=int64` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import LongformerForMaskedLM, LongformerTokenizer

        >>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

        >>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  # long input document
        >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

        >>> attention_mask = None  # default is local attention everywhere, which is a good choice for MaskedLM
        >>> # check `LongformerModel.forward` for more details how to set *attention_mask*
        >>> outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        >>> loss = outputs.loss
        >>> prediction_logits = outputs.logits
        ```"""
        
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.reshape([-1, self.longformer.config["vocab_size"]]),
                                      labels.reshape([-1]))
        
        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


class LongformerForSequenceClassification(LongformerPreTrainedModel):
    def __init__(
            self,
            longformer,
            num_labels=2
    ):
        super(LongformerForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        
        self.longformer = longformer
        self.longformer.pooler = None  # add_pooling_layer=False
        self.classifier = LongformerClassificationHead(
            hidden_size=longformer.config["hidden_size"],
            hidden_dropout_prob=longformer.config["hidden_dropout_prob"],
            num_labels=num_labels)
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels (`paddle.Tensor, dtype=paddle.int64` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if global_attention_mask is None:
            # Initializing global attention on CLS token...
            global_attention_mask = paddle.zeros(shape=input_ids.shape, dtype=input_ids.dtype)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
        
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                self.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"
            
            if self.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class LongformerForQuestionAnswering(LongformerPreTrainedModel):
    """
        Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /
        TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """
    
    def __init__(
            self,
            longformer,
            num_labels=2
    ):
        super(LongformerForQuestionAnswering, self).__init__()
        self.longformer = longformer
        self.longformer.pooler = None  # add_pooling_layer=False
        self.qa_outputs = nn.Linear(longformer.config["hidden_size"], num_labels)
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        start_positions (`paddle.Tensor, dtype=int64` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`paddle.Tensor, dtype=int64` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from tokenizer import LongformerTokenizer
        >>> import paddle

        >>> tokenizer = LongformerTokenizer.from_pretrained("longformer-large-4096")
        >>> model = LongformerForQuestionAnswering.from_pretrained("longformer-large-4096")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> encoding = tokenizer.encode(question, text, return_attention_mask=True)
        >>> input_ids = encoding["input_ids"]

        >>> # default is local attention everywhere
        >>> # the forward method will automatically set global attention on question tokens
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]

        >>> outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits
        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

        >>> answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
        >>> answer = tokenizer.decode(
        ...     tokenizer.convert_tokens_to_ids(answer_tokens)
        >>> )  # remove space prepending space token
        ```"""
        if global_attention_mask is None:
            if input_ids is None:
                pass
            # "It is not possible to automatically generate the `global_attention_mask` because input_ids is None. Please make sure that it is correctly set."
            else:
                # set global attention on question tokens automatically
                global_attention_mask = _compute_global_attention_mask(input_ids,
                                                                       self.longformer.config["sep_token_id"])
        
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = paddle.split(logits, num_or_sections=[1, 1], axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


class LongformerForTokenClassification(LongformerPreTrainedModel):
    """
        Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
        for Named-Entity-Recognition (NER) tasks.
    """
    
    def __init__(
            self,
            longformer,
            num_labels=2
    ):
        super(LongformerForTokenClassification, self).__init__()
        self.num_labels = num_labels
        
        self.longformer = longformer
        self.longformer.pooler = None  # add_pooling_layer=False
        self.dropout = nn.Dropout(longformer.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(longformer.config["hidden_size"], num_labels)
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels (`paddle.Tensor, dtype=int64` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        sequence_output = outputs[0]
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
        
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class LongformerForMultipleChoice(LongformerPreTrainedModel):
    """
        Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
        a softmax) e.g. for RocStories/SWAG tasks.
    """
    
    def __init__(self, longformer, ):
        super(LongformerForMultipleChoice, self).__init__()
        
        self.longformer = longformer
        self.dropout = nn.Dropout(longformer.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(longformer.config["hidden_size"], 1)
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            head_mask=None,
            labels=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels (`paddle.Tensor, dtype=int64` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        
        # set global attention on question tokens
        if global_attention_mask is None and input_ids is not None:
            # "Initializing global attention on multiple choice..."
            # put global attention on all tokens after `config.sep_token_id`
            global_attention_mask = paddle.stack(
                [
                    _compute_global_attention_mask(input_ids[:, i], self.longformer.config["sep_token_id"],
                                                   before_sep_token=False)
                    for i in range(num_choices)
                ],
                axis=1,
            )
        
        flat_input_ids = input_ids.reshape([-1, input_ids.size(-1)]) if input_ids is not None else None
        flat_position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.reshape(
            [-1, token_type_ids.size(-1)]) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.reshape(
            [-1, attention_mask.size(-1)]) if attention_mask is not None else None
        flat_global_attention_mask = (
            global_attention_mask.reshape([-1, global_attention_mask.size(-1)])
            if global_attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            inputs_embeds.reshape([-1, inputs_embeds.size(-2), inputs_embeds.size(-1)])
            if inputs_embeds is not None
            else None
        )
        
        outputs = self.longformer(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            global_attention_mask=flat_global_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape([-1, num_choices])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        
        output = (reshaped_logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


def prune_linear_layer(layer, index, axis: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.

    Args:
        layer (`paddle.nn.Linear`): The layer to prune.
        index (`paddle.Tensor(dtype=paddle.int64)`): The indices to keep in the layer.
        axis (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `paddle.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(axis, index).clone().detach()
    if layer.bias is not None:
        if axis == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.shape)
    new_size[axis] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias_attr=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W)
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b)
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
        heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], paddle.Tensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], paddle.Tensor(dtype=paddle.int64)]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = paddle.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.reshape([-1]).eq(1)
    index: paddle.Tensor(dtype=paddle.int64) = paddle.arange(len(mask))[mask].long()
    return heads, index


def apply_chunking_to_forward(
        forward_fn: Callable[..., paddle.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> paddle.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., paddle.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[paddle.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.


    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""
    
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    
    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )
    
    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )
        
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )
        
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        
        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return paddle.concat(output_chunks, axis=chunk_dim)
    
    return forward_fn(*input_tensors)


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def _get_question_end_index(input_ids, sep_token_id):
    """
    Computes the index of the first occurrence of `sep_token_id`.
    """
    
    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]
    
    assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
    assert (
            sep_token_indices.shape[0] == 3 * batch_size
    ), f"There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You might also consider to set `global_attention_mask` manually in the forward function to avoid this error."
    return sep_token_indices.reshape([batch_size, 3, 2])[:, 0, 1]


def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    """
    Computes global attention mask by putting attention on all tokens before `sep_token_id` if `before_sep_token is
    True` else after `sep_token_id`.
    """
    question_end_index = _get_question_end_index(input_ids, sep_token_id)
    question_end_index = question_end_index.unsqueeze(axis=1)  # size: batch_size x 1
    # bool attention mask with True in locations of global attention
    attention_mask = paddle.arange(end=input_ids.shape[1])
    if before_sep_token is True:
        attention_mask = paddle.cast((attention_mask.expand_as(input_ids) < question_end_index), paddle.int64)
    else:
        # last token is separation token and should not be counted and in the middle are two separation tokens
        attention_mask = paddle.cast(
            paddle.cast(attention_mask.expand_as(input_ids) > (question_end_index + 1), paddle.int64) * (
                    attention_mask.expand_as(input_ids) < input_ids.shape[-1]
            ), paddle.int64)
    
    return attention_mask
