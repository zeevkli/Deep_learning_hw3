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
1. When sampling after the model was trained, we rely on the model's predictions and trust them, so we want to increase the chances of sampling the most likely characters.
2. When The temrpature is very high, predictions that have different probabilities are 'squeezed' tighter. In this way we can make the model act more randomly, and sometimes predict words that had a lower probability.
This will make the model more "creative" in a way - it would be able to generate text that has a different structure and meaning than wat it was trained on.
3. Tuning the temprature to a lower value will make the model more deterministic - the highest probability will have a higher value and the lower ones will be even lower.
The model will prefer the predictions with the higher probabilities, so it's behavior will not differ much from the text it was trained on.
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
    hypers['learn_rate'] = 0.0003
    hypers['batch_size'] = 128
    hypers['betas'] = (0.5, 0.999)
    hypers['z_dim'] = 128
    hypers['h_dim'] = 1024
    hypers['x_sigma2'] = 0.001
    # ========================
    return hypers


part2_q1 = r"""
The sigma squared here is a hyperparameter used to construct the loss function. It divides the first part of the loss
if the vae, which is the MSE loss of the prediction from the actual distribution. When x_sigma2 is increased, we panalyze less on this part of the loss, and we increase
the significance of the kldiv loss - which measures the similatiry of the distribution we produced to a normal distribution.
This means we only care about producing a normal distribution over the latent space, and we don't care about the proximity of our produced samples to the actual distribution.
What will happen is that even though sampling works good, the actual images produced will be not that close to the original images.

On the opposite, when we decrease x_sigma2, we increase the weight of the MSE loss of the prediction to the actual distribution, and we care less about the kldiv part of the loss.
The affect would be that even though we are able to memorize the training data and get good results on the samples from it,
the sampling will now work and when we try to create new samples by sampling from a gaussian we will get poorly produced samples.
"""

part2_q2 = r"""
1. The reconstruction loss is the MSE loss of the prediction from the actual distribution - this is important to actually produce images that are close to the 
original distribution.
The kldiv loss is the KL divergence between the distribution we produced and a normal distribution. This part is crucial for the sampling to work - 
we want to sample from a normal distribution for the sampling to work.
2. The kldiv term in the loss measures the simpliary of the posterior of the encoder in the latent space - how likely we are to get z for x given sample x.
Because for sampling we want to use a regular gaussian distribution, we want the posterior to be as close to a gaussian as possible.
The kldiv term minimizes the distance between the posterior and a gaussian distribution.
3. This affect is crucial for the sampling to work and create good samples in the input space.
It improves generalization - we require the model to create a smooth distribution of the encoder over the latent space, and not send different samples to different regions in that space.
"""

part2_q3 = r"""
In generative models, we want to study a distibution that matches our samples.
To do that, we want to maximize the likelihood of the samples given by some distribution.
This is a general term for training generative models - we want to maximize the likelihood of the samples given by the model.
"""

part2_q4 = r"""
The reason we use the log of the varience, is that because varience is strictly positive, but the model could learn negative values as well.
For the model to learn the varience, it needs to be positive, and the log of the varience is a way to ensure that.
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr = 0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers['embed_dim'] = 128
    hypers['num_heads'] = 8
    hypers['num_layers'] = 2
    hypers['hidden_dim'] = 256  
    hypers['window_size'] = 128
    hypers['dropout'] = 0.2
    hypers['lr'] = 0.001
    # ========================
    return hypers


part3_q1 = r"""
Stacking up attention layers with sliding windows is similar to the way stacking up CNN layers
in a convolutional neural network increses the receptive fields of the window in later layers in the model.
In the first layer, each token is only affected by it's neighbors from the small window.
Then, in the nest layer, when the token's already reprsent not just themself but the window surrounding them.
Then, apllying sliding window attention again allos the model to have a wider context in the sentence.
"""

part3_q2 = r"""
One way to make the context bigger is to use a kinf of dialitaon when we apply the attention.
For half of the window we have, we will use the reular method suggeted - look w/4 to the left and to the right when apllying the attention.
Then, we will use the other half of the window size we have left as a dialitaed window, multiplying keys in indexes with jumps of 2.
Overall, each of the n tokens will be multiplied by w keys, so we get a complecity of O(n*w).
analyzing the amount of layers taken for all the tokens to influence each other, we will get that if the original amount of layers was n
and the context to every side was w/2, now the context to each side is 3w/4, so it will take 3n/4 layers to influence each other.
That way, every token will be influenced by a smaller window adjacent to it, but a wider window overall.
After the first layer, when the tokens in the dialited window have been influenced by their surrounding tokens, 
the multiplication in the dialited window will also be influenced by these tokens, even though we havent directly calculated the multiplication with their indexes.
The limitation to this method, is that the first windows will be smaller in the area adjcant to the token, so the tokens after w/4 to the sides of this token will take a longer time to influence this token.
"""

# ==============
