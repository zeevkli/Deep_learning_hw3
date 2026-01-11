import torch
import torch.nn as nn
import math
import torch.nn.functional as F



def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size%2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0]


    # ====== YOUR CODE: ======
    orig_has_heads = (q.dim() == 4)
    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
    elif q.dim() != 4:
        raise ValueError(f"Expected q to have 3 or 4 dims, got {q.dim()}")
    
    B, H, L, D = q.shape
    w = window_size // 2
    neg_inf = -10e10

    k_pad = F.pad(k, (0, 0, w, w))  # [B,H,L+2w,D]
    v_pad = F.pad(v, (0, 0, w, w))

    # unfold -> [B,H,L,D,2w+1]  => permute -> [B,H,L,2w+1,D]
    k_win = k_pad.unfold(2, 2 * w + 1, 1).permute(0, 1, 2, 4, 3)
    v_win = v_pad.unfold(2, 2 * w + 1, 1).permute(0, 1, 2, 4, 3)

    # ---- local scores: [B,H,L,2w+1] ----
    scores_local = (q.unsqueeze(-2) * k_win).sum(dim=-1) / math.sqrt(D)

    # ---- mask out-of-range neighbors at boundaries ----
    offsets = torch.arange(-w, w + 1, device=q.device)                 # [2w+1]
    idx = torch.arange(L, device=q.device).unsqueeze(1) + offsets      # [L,2w+1]
    valid = (idx >= 0) & (idx < L)                                     # [L,2w+1]
    scores_local = scores_local.masked_fill(~valid[None, None, :, :], neg_inf)

    # ---- padding mask: mask KEYS before softmax; zero padded QUERIES after ----
    query_pad = None
    if padding_mask is not None:
        key_pad = (padding_mask == 0)  # [B,L] bool

        key_pad_win = F.pad(key_pad, (w, w), value=True).unfold(1, 2 * w + 1, 1)  # [B,L,2w+1]
        scores_local = scores_local.masked_fill(key_pad_win[:, None, :, :], neg_inf)

        query_pad = key_pad[:, None, :, None]  # [B,1,L,1]

    attn_local = torch.softmax(scores_local, dim=-1)
    attn_local = torch.nan_to_num(attn_local, nan=0.0)
    if query_pad is not None:
        attn_local = attn_local.masked_fill(query_pad, 0.0)

    # ---- local weighted sum ---- 
    values = (attn_local.unsqueeze(-1) * v_win).sum(dim=-2)  # [B,H,L,D]

    # ---- expand to full attention [B,H,L,L] ----
    attention = torch.zeros((B, H, L, L), device=q.device, dtype=q.dtype)

    idx_clamped = idx.clamp(0, L - 1)  # NOTE: duplicates at boundaries
    attn_to_scatter = attn_local * valid[None, None, :, :].to(attn_local.dtype)

    # CRITICAL FIX: scatter_add_ so duplicates don't overwrite (invalid contributes 0 anyway)
    attention.scatter_add_(
        dim=-1,
        index=idx_clamped[None, None, :, :].expand(B, H, L, 2 * w + 1),
        src=attn_to_scatter
    )

    # ---- return in original shape ----
    if not orig_has_heads:
        values = values.squeeze(1)       # [B,L,D]
        attention = attention.squeeze(1) # [B,L,L]
    # ======================
    # In sliding_window_attention, right before returning values:
    if padding_mask is not None:
        # If the query is padding, force output to 0
        # padding_mask is [Batch, SeqLen] (0=pad, 1=valid)
        values = values.masked_fill((padding_mask == 0).unsqueeze(-1).unsqueeze(1), 0.0)

    return values, attention

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        
        # Determine value outputs
        # call the sliding window attention function you implemented
        # ====== YOUR CODE: ======
        values, attention = sliding_window_attention(q, k, v, self.window_size, padding_mask)
        # ========================

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''

        # ====== YOUR CODE: ======
        attn_x = self.self_attn(x, padding_mask)
        attn_x = self.dropout(attn_x)
        x = self.norm1(attn_x + x)
        norm_x = self.feed_forward(x)
        norm_x = self.dropout(norm_x)
        x = self.norm2(x + norm_x)
        # ========================
        
        return x
    
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

        # ====== YOUR CODE: ======
        x = self.encoder_embedding(sentence)
        x = x * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        classifcation_tokens = x[:, 0, :]
        output = self.classification_mlp(classifcation_tokens)
        
        output = output.squeeze(-1) 
        # =======================
        
        return output  
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    