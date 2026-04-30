### A Pluto.jl notebook ###
# v0.20.3
#
# Bayesian GPT-style text generation from first principles in Julia.
# Run with:  julia -e 'using Pluto; Pluto.run()'
# Then open: notebooks/decoder_tutorial.jl

using Markdown
using InteractiveUtils

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
end

# ╔═╡ title
md"""
# A Bayesian GPT from First Principles
### Next-token prediction with variational weights and learned sparsity

This notebook builds a **decoder-only transformer** — GPT-style — that
learns to generate text from a plain `.txt` file.

The twist: the MLP layers use **Variational Dropout** (Molchanov et al. 2017)
instead of standard `Dense` layers, giving us:
- Bayesian uncertainty over weights
- Automatic weight pruning (sparsity) as a free byproduct of training

What you'll learn:
1. How encoder and decoder transformers differ
2. The causal mask — why token t cannot see token t+1
3. Next-token prediction as a training objective
4. Autoregressive generation — feeding outputs back as inputs
5. Temperature sampling — controlling creativity
6. How variational weights prune themselves during training
"""

# ╔═╡ section_1
md"""
## 1  Encoder vs Decoder

The encoder we built for sentiment classification attends
**bidirectionally** — every token sees every other token.
This is great for understanding but cannot generate text.

A decoder attends **causally** — token at position t can only
see positions ≤ t.  This means the model can be trained to predict
the next token, and at inference time we can generate autoregressively.

```
Encoder (BERT-style)          Decoder (GPT-style)
────────────────────          ──────────────────
bidirectional attention   →   causal masked attention
mean pool → classify      →   predict next token at every position
cross-entropy on label    →   cross-entropy on next token
```

The causal mask is a lower-triangular matrix added to the attention
scores before softmax:

```
         key positions
         t1    t2    t3    t4
query ┌                       ┐
t1    │  0    -∞    -∞    -∞  │
t2    │  0     0    -∞    -∞  │
t3    │  0     0     0    -∞  │
t4    │  0     0     0     0  │
      └                       ┘
```

After softmax, `-∞` becomes exactly `0` — future positions are
completely invisible to the model.
"""

# ╔═╡ section_2
md"""
## 2  Next-Token Prediction

Given a sequence of tokens, the model predicts the next token
at **every position simultaneously** during training:

```
Input  X : [the,  cat,  sat,  on]   ← positions 1..N-1
Target Y : [cat,  sat,  on,  mat]   ← positions 2..N
```

So at position t, the model sees tokens 1..t (due to causal mask)
and must predict token t+1.

The loss is cross-entropy averaged over all positions and all documents:

$$\mathcal{L}_{NLL} = -\frac{1}{T \cdot B} \sum_{t=1}^{T} \sum_{b=1}^{B} \log p(Y_{t,b} \mid X_{1:t, b})$$

Plus the KL term from the variational layers (same as the encoder):

$$\mathcal{L} = N \cdot \mathcal{L}_{NLL} + \beta \cdot \frac{1}{N} \sum_l \text{KL}(q_l \| p)$$
"""

# ╔═╡ load_packages
begin
    using BayesianTransformer
    using Random, Statistics, Flux, Plots
    md"*Packages loaded ✓*"
end

# ╔═╡ load_corpus
begin
    # ── point this at your .txt file ─────────────────────────
    corpus_path = joinpath(@__DIR__, "..", "data", "corpus.txt")

    corpus = read_corpus(corpus_path)

    Random.seed!(42)
    corpus = corpus[randperm(length(corpus))]

    split_idx   = Int(floor(0.9 * length(corpus)))
    train_corpus = corpus[1:split_idx]
    val_corpus   = corpus[split_idx+1:end]

    md"""
    **Corpus**: $(basename(corpus_path))
    Total documents : **$(length(corpus))**
    Train : **$(length(train_corpus))** | Val : **$(length(val_corpus))**
    """
end

# ╔═╡ build_vocab
begin
    vocab     = select_vocabulary(train_corpus; min_document_frequency = 3)
    tokenizer = IndexTokenizer(vocab, "<UNK>")
    md"Vocabulary size: **$(length(vocab))** tokens"
end

# ╔═╡ section_3
md"""
## 3  The Causal Mask in Code

The mask is built once per forward pass based on the current
sequence length N, and broadcast over all heads and batch items:

```julia
function causal_mask(N::Int, T::Type)
    mask = fill(typemin(T), N, N)       # -Inf everywhere
    mask[tril(trues(N, N))] .= zero(T) # zero on/below diagonal
    mask
end

# Inside scaled_dot_attention_causal:
score = score .+ mask    # (N, N, H, B) + (N, N) → broadcasts over H, B
score = softmax(score; dims = 1)
```

Using `typemin(T)` instead of `-Inf` directly ensures the mask
matches the floating point type of the data (Float32 vs Float64).
"""

# ╔═╡ section_4
md"""
## 4  The Variational MLP

Each `TransformerDecoderBlock` contains two `VariationalDropoutMolchanov`
layers in the feed-forward position.

Instead of learning a single weight matrix W, the model learns
a distribution over weights:

$$W_{ij} \sim \mathcal{N}(\theta_{ij}, \sigma^2_{ij})$$

During the forward pass, weights are sampled via the
**reparameterisation trick**:

$$W = \theta + \exp(0.5 \cdot \log\sigma^2) \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

The signal-to-noise ratio $\log\alpha = \log\sigma^2 - 2\log|\theta|$
drives sparsity. When $\log\alpha \geq 3$, the weight is effectively
pure noise and is pruned at inference time.

**Why does this matter for generation?**
A standard GPT can be overconfident — it assigns near-zero probability
to unlikely continuations.  Variational weights introduce calibrated
uncertainty, making the model more robust to out-of-distribution inputs
and giving more meaningful temperature sampling.
"""

# ╔═╡ create_loaders
begin
    train_loader = create_batches(train_corpus, tokenizer; batch_size=32, max_length=128)
    val_loader   = create_batches(val_corpus,   tokenizer; batch_size=32, max_length=128)
    md"DataLoaders ready ✓"
end

# ╔═╡ build_model
begin
    vocab_size        = length(vocab)
    embed_dim         = 64
    n_decoder_layers  = 2
    n_heads           = 4
    hidden_dim        = 4 * embed_dim    # standard GPT ratio

    position_encoding = PositionEncoding(embed_dim)

    model = Chain(
        # (1, T, B) → (T, B)
        x -> reshape(x, size(x, 2), size(x, 3)),

        # (T, B) → (embed_dim, T, B)
        Embed(embed_dim, vocab_size),

        # add sinusoidal position encoding
        x -> x .+ position_encoding(size(x, 2)),

        Dropout(0.1),

        # causal decoder blocks
        [TransformerDecoderBlock(n_heads, embed_dim, hidden_dim; pdrop=0.1)
         for _ in 1:n_decoder_layers]...,

        # project to vocab at every position → (vocab_size, T, B)
        # Dense applied to first dim, broadcast over T and B automatically
        x -> reshape(
            hcat([Flux.Dense(embed_dim, vocab_size)(x[:, t, :])
                  for t in axes(x, 2)]...),
            vocab_size, size(x, 2), size(x, 3)
        ),
    )

    total_params = sum(length, Flux.params(model))
    md"Model built ✓ — **$total_params** trainable parameters"
end

# ╔═╡ better_model
begin
    # Cleaner version using a proper output projection layer
    output_proj = Flux.Dense(embed_dim, vocab_size)

    model = Chain(
        x -> reshape(x, size(x, 2), size(x, 3)),
        Embed(embed_dim, vocab_size),
        x -> x .+ position_encoding(size(x, 2)),
        Dropout(0.1),
        [TransformerDecoderBlock(n_heads, embed_dim, hidden_dim; pdrop=0.1)
         for _ in 1:n_decoder_layers]...,
        # (embed_dim, T, B) → (vocab_size, T, B)
        # Dense in Flux applies to first dim and broadcasts over the rest
        output_proj,
    )

    total_params = sum(length, Flux.params(model))
    md"Model (clean version) ✓ — **$total_params** trainable parameters"
end

# ╔═╡ section_5
md"""
## 5  Training

We use the same ELBO loss as the encoder, but now computed at
every token position:

```julia
ŷ_flat   = reshape(ŷ, vocab_size, T * B)    # flatten positions and batch
Y_onehot = onehotbatch(vec(Y), 1:vocab_size) # one-hot targets
nll      = N * logitcrossentropy(ŷ_flat, Y_onehot)
loss     = nll + β * kl_sum / N
```

KL warm-up is especially important here — with a large vocabulary
the NLL dominates early training, and aggressive pruning before
the model learns basic word patterns would be catastrophic.
"""

# ╔═╡ train
begin
    history = train_decoder!(
        model, train_loader, val_loader;
        epochs      = 20,
        lr          = 1e-3,
        kl_schedule = kl_schedule,
    )
    md"Training complete ✓"
end

# ╔═╡ plot_accuracy
begin
    p1 = plot(history.train_accuracies; label="Train", lw=2,
              title="Token Accuracy", xlabel="Epoch", ylabel="Accuracy")
    plot!(p1, history.val_accuracies; label="Val", lw=2, ls=:dash)
    p1
end

# ╔═╡ plot_sparsity
begin
    p2 = plot(history.sparsities_dense1; label="dense1", lw=2,
              title="MLP Sparsity (log α ≥ 3)", xlabel="Epoch",
              ylabel="Fraction pruned")
    plot!(p2, history.sparsities_dense2; label="dense2", lw=2, ls=:dash)
    p2
end

# ╔═╡ plot_loss
begin
    p3 = plot(history.train_losses; label="Train ELBO", lw=2, color=:purple,
              title="Variational Free Energy", xlabel="Epoch", ylabel="Loss")
    plot!(p3, history.val_losses; label="Val ELBO", lw=2, color=:orange, ls=:dash)
    p3
end

# ╔═╡ section_6
md"""
## 6  Generation — Autoregressive Sampling

At inference time we generate one token at a time, feeding
each prediction back as input for the next step:

```
seed = "the cat"
step 1: [the, cat]          → model → logits → sample "sat"
step 2: [the, cat, sat]     → model → logits → sample "on"
step 3: [the, cat, sat, on] → model → logits → sample "the"
...
```

At each step we only use the logits at the **last position** —
that's the prediction for what comes next.

**Temperature** controls the randomness:
- Low  (0.2) → conservative, repetitive
- High (1.2) → creative, sometimes incoherent
"""

# ╔═╡ generate_text
begin
    seed = "the"   # ← change this to any word in your vocabulary

    println("\n--- Low temperature (conservative) ---")
    generate_samples(model, tokenizer, seed;
                     n_samples=3, max_length=30, temperature=0.3f0)

    println("\n--- Medium temperature ---")
    generate_samples(model, tokenizer, seed;
                     n_samples=3, max_length=30, temperature=0.8f0)

    println("\n--- High temperature (creative) ---")
    generate_samples(model, tokenizer, seed;
                     n_samples=3, max_length=30, temperature=1.2f0)

    md"Generation complete ✓ — see output above"
end

# ╔═╡ section_7
md"""
## 7  Takeaways

### Encoder vs Decoder — one structural change
The only architectural difference between our sentiment classifier
and this language model is the causal mask and removing the pooling layer.
The variational MLP, residual connections, and layer norm are identical.

### What the sparsity plot tells you
Early epochs: β ≈ 0, no pruning pressure — the model learns freely.
As β → 1, weights with high log α are squeezed to zero.
The fraction of pruned weights is a proxy for **model compression** —
a sparse model can be deployed faster with no loss in accuracy.

### Temperature and uncertainty
Because weights are stochastic, the model produces slightly different
logits on each forward pass — even for the same input.
This is Bayesian uncertainty, and temperature sampling interacts
with it naturally: low temperature concentrates on the mode of the
distribution, high temperature explores the tails.

### Next steps
- Increase `n_decoder_layers` and `embed_dim` for richer text
- Try subword tokenisation (BPE) to reduce `<UNK>` tokens
- Compare sparsity and accuracy against a standard `Dense` MLP baseline
- Export the Pluto notebook to HTML for a self-contained blog post:
  ```julia
  Pluto.Export.export_notebook("notebooks/decoder_tutorial.jl")
  ```
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
# Cell required by Pluto for package management