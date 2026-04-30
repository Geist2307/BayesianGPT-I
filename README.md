# BayesGPT-I 🎲

> What if a GPT was honest about what it doesn't know?

BayesGPT-I is a minimal Bayesian GPT decoder implemented from scratch in Julia.
It is a transformer that doesn't simply learn -- it **learns what to forget**, and
**tells you when it is uncertain**.

Built as a direct exntesion to my MSc Dissertation at St Andrews : `Practical Variational Inference of Bayesian Neural Networks`, BayesGPT-I is the decoder version of the Variational Dropout-enhanced Transformer Network in my dissertation. 


The pretrained model was trained on:
- *The Golden Man* —- Philip K. Dick
- *Lone Star Planet* —- John J. McGuire
- *At the mountains of Madness* -- H.P. Lovecraft

 Naturally, slightly paranoid about its own weights and overwhelmed by lovecraftian cosmi horror at complex inputs. 

---

## Core concept

A standard GPT learns **point estimates** for its weights:

```
W = 0.73   ← this weight is exactly 0.73, always
```

BayesGPT-I learns **distributions** over its MLP weights:

```
W ~ N(0.73, σ²)   ← this weight is probably around 0.73, but uncertain
```

This single change gives you three things for free:

**1. Automatic weight pruning**

The ELBO loss simultaneously fits the data and removes unnecessary weights:

```
L = fit the data  +  β · don't be more complex than you need to be
```

Weights the model is uncertain about get pruned automatically.
No separate pruning step. No hand-tuned regularisation. Just Bayes.

**2. Honest uncertainty at generation time**

Each forward pass samples different weights `W ~ N(θ, σ²)`.
Run the same prompt 20 times → 20 slightly different outputs.
The variance across those outputs is genuine model uncertainty:

```
"he knows thing menace room"

  thing          σ=0.295  moderate    ← somewhat confident
  menace  σ=0.332  uncertain   ← many alternatives here
```

**3. Calibrated generation**

Standard GPT says "her" with 95% confidence even when looping.
BayesGPT-I says "her" with σ=0.35 — honestly uncertain.

---

## Results

Trained on ~7,700 lines from 3 books (Philip K. Dick + Project Gutenberg).
613K parameters, 50 epochs, CPU only (Apple M2, ~12 hours).

```
Final metrics:
  train_acc        : 0.998
  val_acc          : 0.992
  sparsity_dense1  : 0.008
  sparsity_dense2  : 0.010
```

**Three generation modes, same seed:**

```
Seed: "he knows"

Standard (k=vocab):    "he knows he he he"          ← loops
Top-k=50 + penalty:    "he knows thing years later" ← literary!
Bayesian (20 passes):  "he knows thing menace room"
                        thing  σ=0.295  moderate
                        menace σ=0.332  uncertain  ← peaks here
                        room   σ=0.314  uncertain
```

**An honest finding:**

Sparsity stayed low (~1%) because the model is not overparameterised
relative to 3 books. With a larger model or more data, the KL term
would prune more aggressively. The uncertainty signal however works
correctly at any scale -- the model knows what it doesn't know.

---

## Architecture

```
Input tokens  (1, T, B)

Embed                            <- learned token embeddings
      ↓
PositionEncoding                 <- fixed sinusoidal (Vaswani et al. 2017)
      ↓
Dropout
      ↓
TransformerDecoderBlock × 2
  ├── CausalMultiheadAttention   ← standard point estimate weights
  ├── LayerNorm + residual
  ├── VariationalDropout         ← BAYESIAN (dm → dhid)
  ├── VariationalDropout         ← BAYESIAN (dhid → dm)
  └── LayerNorm + residual
      ↓
Dense(embed_dim, vocab_size)     ← logits at every position
      ↓
Output  (vocab_size, T, B)
```

Only the MLP layers inside each decoder block are Bayesian.
Attention weights are standard point estimates — a deliberate choice:
attention learns *routing* (hard to prune), MLP learns *knowledge* (safe to prune).

---

## The ELBO Loss

Training minimises the negative Evidence Lower Bound:

```
L = N · NLL(ŷ, y)  +  β · (1/N) · Σ KL(q(W|θ,σ) ‖ p(W))
```

Where KL uses the closed-form approximation from Molchanov et al. (2017):

```
KL ≈ Σᵢⱼ [ -k₁ + k₁·σ(k₂ + k₃·logα) - 0.5·log(1 + exp(-logα)) ]

logα = logσ² - 2·log|θ|   ← signal-to-noise ratio per weight
```

When `logα ≥ 3`, the weight is effectively noise and is pruned.

**KL warm-up** grows β from 0 -> 1 over 30 epochs, preventing
aggressive pruning before the model has learned anything useful.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/yourname/BayesGPT-I
cd BayesGPT-I

# 2. Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Add your text data
cp your_book.txt data/

# 4. Train
julia --project=. run.jl

# 5. Generate
julia --project=. generate_only.jl
```

---

## Try the Pretrained Model

Training on CPU can take a very long time? Download the pretrained weights and generate immediately:

```bash
# 1. Clone
git clone https://github.com/yourname/BayesGPT-I
cd BayesGPT-I

# 2. Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Download pretrained weights (trained on Gutenberg novels)
# model.bson and tokenizer.bson are included in the repo

# 4. Generate immediately
julia --project=. generate_only.jl
```

Vocabulary size: **3,476 words** | Parameters: **613,524**

Try these seeds out of the box:

```julia
# in generate_only.jl, change the seed to any of these:

bayesian_generate_report(model, tokenizer, "he knows";
                         n_forward=20, max_length=5,
                         temperature=0.8f0, k=50, penalty=1.5f0)

bayesian_generate_report(model, tokenizer, "she wanted";
                         n_forward=20, max_length=5,
                         temperature=0.8f0, k=50, penalty=1.5f0)

bayesian_generate_report(model, tokenizer, "the war was";
                         n_forward=20, max_length=5,
                         temperature=0.8f0, k=50, penalty=1.5f0)
```

Expected output:

```
[Bayesian n_forward=20, penalty=1.5] Seed: "he knows"
──────────────────────────────────────────────────
Generated : he knows thing menace room

Token uncertainty (σ) and 95% CI:
──────────────────────────────────────────────────
  thing           σ=0.295  CI=±0.578  ████████████ moderate
  menace          σ=0.332  CI=±0.651  █████████████ uncertain
  room            σ=0.314  CI=±0.616  █████████████ uncertain

  Mean uncertainty : 0.314
```

## Interactive Demo

The project comes with a julia app, where you can add your input directly into the window and select params.

Run locally with:
```bash
julia --project=. app.jl
```
Then open `http://localhost:8080`

## Project Structure

```
BayesGPT-I/
├── run.jl                      # training entry point
├── generate_only.jl            # generation + uncertainty demo
├── Project.toml
├── src/
│   ├── layers.jl               # Embed, PositionEncoding, VariationalDropout
│   ├── attention.jl            # bidirectional MultiheadAttention
│   ├── attention_causal.jl     # causal mask + CausalMultiheadAttention
│   ├── transformer_decoder.jl  # TransformerDecoderBlock
│   ├── tokenizer_decoder.jl    # plain .txt tokenization pipeline
│   ├── training_decoder.jl     # ELBO loss + training loop
│   └── generate.jl             # standard / top-k / Bayesian generation
└── data/                       # put your .txt files here
```

---

## Key Hyperparameters

```julia
embed_dim  = 64       # embedding dimension
n_heads    = 4        # attention heads
hidden_dim = 256      # MLP hidden dimension (4 × embed_dim)
n_layers   = 2        # number of decoder blocks
max_length = 128      # context window
batch_size = 32
epochs     = 50
lr         = 1e-3

# Bayesian
kl_schedule(epoch) = min(1.0f0, epoch / 30f0)   # warm-up over 30 epochs
SPARSITY_THRESHOLD = 3.0f0                        # logα ≥ 3 → pruned
```

---

## Generation Modes

```julia
# 1. Standard — temperature sampling
generate_samples(model, tokenizer, "he knows";
                 n_samples=5, temperature=0.8f0)

# 2. Top-k with repetition penalty — fixes loops
generate_samples_topk(model, tokenizer, "he knows";
                      n_samples=5, temperature=0.8f0, k=10, penalty=1.5f0)

# 3. Bayesian — uncertainty per token
bayesian_generate_report(model, tokenizer, "he knows";
                         n_forward=20, temperature=0.8f0, k=10, penalty=1.5f0)
```

---

## Why Julia?

Because if you're going to be uncertain about your weights,
you might as well be certain about your types.

Julia's parametric type system means custom layers like
`VariationalDropoutMolchanov` run at near-C speed without
writing a single line of C++. In PyTorch, a custom layer
written in pure Python would be orders of magnitude slower.

---

## Limitations & Future Work

**Current limitations:**
- Small dataset means model memorises rather than generalises
- Sparsity is low because model is not overparameterised
- GPU support incomplete (Metal.jl gaps in `batched_mul`)



---

## References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Molchanov et al. (2017) — *Variational Dropout Sparsifies Deep Neural Networks*
- Kingma & Welling (2013) — *Auto-Encoding Variational Bayes*
- Karpathy (2025) — *Minimal GPT-2 in pure Python*

---

## Citation

```bibtex
@misc{bayesgpt1,
  title  = {BayesGPT-I: A Minimal Bayesian GPT in Julia},
  author = {Andrei Bleahu},
  year   = {2025},
  url    = {https://github.com/yourname/BayesGPT-I}
}
```