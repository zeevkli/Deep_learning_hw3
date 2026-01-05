r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    batch_size=256
    seq_len=64
    h_dim=2048
    n_layers=3
    dropout=0.3
    learn_rate=0.001
    lr_sched_factor=0.5
    lr_sched_patience=3
    hypers = dict(
        batch_size=batch_size,
        seq_len=seq_len,
        h_dim=h_dim,
        n_layers=n_layers,
        dropout=dropout,
        learn_rate=learn_rate,
        lr_sched_factor=lr_sched_factor,
        lr_sched_patience=lr_sched_patience,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I. The king "
    temperature = 0.04
    # ========================
    return start_seq, temperature


part1_q1 = r"""
There are a few reasons why we split the corpus into sequences and not train on the entire corpus as one batch:
1. If we'd use the entire corpus at once, we'd have a computational graph that's the length of the entire corpus - that is more than 6 milion steps in the gradient calculation and the forwrad calculation.
This would require to hole a huge amount of memory at once, and also the calculations would be very long.
2. Splitting the corpus into sequences optimizes on using a gpu, that can make the gradient claultaions parallel on batches.
3. There is no big advantage in using  the entire corpus at once, since language is mostly local.
The first word in the corpus belongs to 1 play, and mearly doesn't affect qwords from the second play.
"""

part1_q2 = r"""
The way the model can 'memorize' is through the hidden state.
When we generate test, we give the model an initial short sequence, and every time a character is generated, we make use of the hidden state outputted in the previous step.
GRU 'learns' how to tasnform the hidden state from iteration to iteraion and in that way memorize things for a long time.
"""

part1_q3 = r"""
In an RNN model, especially in ours that traines on shakspere's plays, there is no point in shuffling the order of the batches - 
we want the model to learn the structure of the corpus, and the plays by their order.
In an RNN model, the hidden state is used to 'memorize' the previous characters, and in that way the model can learn the structure of the corpus.
"""

part1_q4 = r"""
1. When sampling after the model was trained, we rely on the model's predictions and truat them, so we want to increase the chances of sampling the most likely characters.
2. When The temrpature is very high, prediction's that has different probabilities are 'squeezed' tighter. In this way we can make the model
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
