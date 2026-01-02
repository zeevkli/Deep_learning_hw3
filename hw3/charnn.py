import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator

def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    char_to_idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = ''.join(char for char in text if char not in chars_to_remove)
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    text_onehot = torch.zeros(len(text), len(char_to_idx), dtype=torch.int8)
    for i, char in enumerate(text):
        text_onehot[i, char_to_idx[char]] = 1
    # ========================
    return text_onehot


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    chars = [idx_to_char[float(idx)] for idx in embedded_text.argmax(dim=1)]
    # ========================
    return ''.join(chars)


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # 1. Embed the given text.
    embedded_text = chars_to_onehot(text, char_to_idx).to(device)
    N = (embedded_text.shape[0] - 1) // seq_len
    samples = embedded_text[:N * seq_len].reshape(N, seq_len, -1)
    labels = embedded_text[1:N * seq_len + 1].argmax(dim=1).reshape(N, seq_len)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    return torch.softmax(y / temperature, dim=dim)
    # ========================


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        hidden_state = None
        sequence_to_model = start_sequence
        for _ in range(n_chars - len(start_sequence)):
            x = chars_to_onehot(sequence_to_model, char_to_idx).to(device).float() 
            x = x.unsqueeze(0)                                                  
            y, hidden_state = model(x, hidden_state)
            logits = y[0, -1, :]              
            next_idx = torch.multinomial(hot_softmax(logits, dim=0, temperature=T), num_samples=1).item()
            next_char = idx_to_char[next_idx]
            out_text += next_char
            sequence_to_model = next_char

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.offset = 0

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        n = len(self.dataset)
        b = self.batch_size
        if b <= 0:
            raise ValueError("batch_size must be positive")

        n_trim = (n // b) * b          # drop last incomplete batch
        num_batches = n_trim // b

        idx = []
        # Produce indices in batch order (j = batch id)
        for j in range(num_batches):
            for i in range(b):
                idx.append(i * num_batches + j)
        
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.update_wx = nn.ModuleList()
        self.update_wh = nn.ModuleList()
        self.reset_wx  = nn.ModuleList()
        self.reset_wh  = nn.ModuleList()
        self.cand_wx   = nn.ModuleList()
        self.cand_wh   = nn.ModuleList()
        self.dropouts  = nn.ModuleList()
        
        for layer in range(n_layers):
            layer_in_dim = in_dim if layer == 0 else h_dim

            self.update_wx.append(nn.Linear(layer_in_dim, h_dim, bias=True))
            self.reset_wx.append(nn.Linear(layer_in_dim, h_dim, bias=True))
            self.cand_wx.append(nn.Linear(layer_in_dim, h_dim, bias=True))

            self.update_wh.append(nn.Linear(h_dim, h_dim, bias=False))
            self.reset_wh.append(nn.Linear(h_dim, h_dim, bias=False))
            self.cand_wh.append(nn.Linear(h_dim, h_dim, bias=False))

            self.dropouts.append(nn.Dropout(dropout))
        self.W_hy = nn.Linear(h_dim, out_dim, bias=True)

        # what forward unpacks
        self.layer_params = [
            (self.update_wx[l], self.update_wh[l],
            self.reset_wx[l],  self.reset_wh[l],
            self.cand_wx[l],   self.cand_wh[l],
            self.dropouts[l])
            for l in range(n_layers)
        ]

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # Loop over layers of the model

        activation_sigmoid, activation_tanh = nn.Sigmoid(), nn.Tanh()
        output_seq = torch.zeros_like(input)

        for time_step in range(seq_len):
            current_input = layer_input[:, time_step]

            for layer_index, prev_state in enumerate(layer_states):
                update_wx, update_wh, reset_wx, reset_wh, candidate_wx, candidate_wh, dropout_layer = self.layer_params[layer_index]

                if layer_index > 0:
                    current_input = layer_states[layer_index - 1]

                update_gate = activation_sigmoid(update_wx(current_input) + update_wh(prev_state))
                reset_gate = activation_sigmoid(reset_wx(current_input) + reset_wh(prev_state))
                candidate_state = activation_tanh(candidate_wx(current_input) + candidate_wh(reset_gate * prev_state))

                new_state = update_gate * prev_state + (1 - update_gate) * candidate_state
                layer_states[layer_index] = new_state

                if layer_index > 0:
                    current_input = dropout_layer(new_state)

            output_seq[:, time_step] = self.W_hy(layer_states[-1])

        final_hidden_state = torch.stack(layer_states, dim=1)
        hidden_state = final_hidden_state
        layer_output = output_seq

        return layer_output, hidden_state
