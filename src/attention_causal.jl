# ============================================================
# attention_causal.jl
# Causal (decoder-style) multi-head attention.
#
# The only difference from attention.jl is the causal mask
# applied before softmax — token at position t can only
# attend to positions <= t.
#
# Everything else (projections, head splitting, batched_mul)
# is identical to the encoder version.
# ============================================================

using NNlib: batched_mul
using Flux: softmax, Dense
using LinearAlgebra: tril



# ------------------
# Causal mask
# -------------------

"""
    causal_mask(N, T) => Matrix{T}

Lower-triangular mask of shape (N, N).

    0     on and below the diagonal  → positions the query CAN attend to
    -Inf  above the diagonal         → future positions, zeroed by softmax

Example for N=4:

    pos1  pos2  pos3  pos4
    0     -Inf  -Inf  -Inf
    0     0     -Inf  -Inf
    0     0     0     -Inf
    0     0     0     0
"""
function causal_mask(N::Int, T::Type)
    mask = fill(typemin(T), N, N)          # -Inf everywhere
    mask[tril(trues(N, N))] .= zero(T)    # zero on/below diagonal
    mask
end


# ---------------------------------
# Causal scaled dot-product attention
# -------------------------------------

"""
 Scaled Dot-Product Attention(Q, K, V)

Computes  softmax(K^T Q / √dh) · V  for a 4-D batched input.

# Shapes  (dh, N, H, B)
- Q, K, V : `(dh, N, nhead, batch)`
- Returns  : `(dh, N, nhead, batch)`

# Args 

dh : this is length of each vector token
N :: this is the number of tokens (sequence lenght)
H :: number of heads (we already split into heads)
B :: bach size

# Rationale

In the function below, we are just interested in  multiplying them while carrying 
H and B along. This function actually receives Q, K, V after splitting them into heads. The causal 
mask ensures tokens are applied to positions <=t
"""
function scaled_dot_attention_causal(
        Q::A, K::A, V::A) where {T, A <: AbstractArray{T, 4}}
    dh = size(Q, 1)
    N  = size(Q, 2)

    Kt    = permutedims(K, (2, 1, 3, 4))                       # (N, dh, H, B)
    score = one(T)/convert(T, sqrt(dh)) .* batched_mul(Kt, Q)  # (N, N, H, B)

    # apply causal mask 
    # mask is (N, N) —- broadcast automatically over H and B
    mask  = causal_mask(N, T)
    score = score .+ mask                                       # (N, N, H, B)

    score = softmax(score; dims = 1)
    batched_mul(V, score)                                       # (dh, N, H, B)
end


# ────────────────────────────────────────────────────────────
# Causal multi-head attention
# ────────────────────────────────────────────────────────────

"""
    multi_head_scaled_dot_attention_causal(nhead, Q, K, V)

Identical to `multi_head_scaled_dot_attention` but calls
`scaled_dot_attention_causal` internally.
"""
function multi_head_scaled_dot_attention_causal(
        nhead::Int, Q::A, K::A, V::A) where {T, A <: AbstractArray{T, 3}}
    dm = size(Q, 1)
    dh = dm ÷ nhead

    to_heads(X) = permutedims(
        reshape(X, dh, nhead, size(X, 2), size(X, 3)),
        [1, 3, 2, 4])

    Qh = to_heads(Q)
    Kh = to_heads(K)
    Vh = to_heads(V)

    attn_out = scaled_dot_attention_causal(Qh, Kh, Vh)
    attn_out  = permutedims(attn_out, [1, 3, 2, 4])
    reshape(attn_out, dm, size(attn_out, 3), size(attn_out, 4))
end


# ------------------------------------
# CausalMultiheadAttention struct
#------------------------------------------

"""
    CausalMultiheadAttention(nhead, dim_model, dim_out)

Drop-in replacement for `MultiheadAttention` that applies a
causal mask — token at position t can only attend to positions ≤ t.

Identical interface; only the internal attention function differs.
"""
struct CausalMultiheadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense}
    nhead  :: Int
    denseQ :: Q
    denseK :: K
    denseV :: V
    denseO :: O
end

Flux.@layer :ignore CausalMultiheadAttention trainable=(denseQ, denseK, denseV, denseO)

function CausalMultiheadAttention(nhead::Int, dim_model::Int, dim_head::Int, dim_out::Int)
    CausalMultiheadAttention(
        nhead,
        Dense(dim_model, dim_head * nhead; bias = false),
        Dense(dim_model, dim_head * nhead; bias = false),
        Dense(dim_model, dim_head * nhead; bias = false),
        Dense(dim_head * nhead, dim_out),
    )
end

function CausalMultiheadAttention(nhead::Int, dim_model::Int, dim_out::Int)
    dim_model % nhead == 0 || error(
        "dim_model=$dim_model must be divisible by nhead=$nhead")
    CausalMultiheadAttention(nhead, dim_model, dim_model ÷ nhead, dim_out)
end

Base.show(io::IO, mha::CausalMultiheadAttention) =
    print(io, "CausalMultiheadAttention(nhead=$(mha.nhead), " *
              "head_dim=$(size(mha.denseQ.weight,1) ÷ mha.nhead), " *
              "$(size(mha.denseQ.weight,2))⇒$(size(mha.denseO.weight,1)))")

# Forward pass (batched 3-D) -------------------------
function (mha::CausalMultiheadAttention)(
        query::A, key::A, value::A) where {T, A <: AbstractArray{T, 3}}
    Q = mha.denseQ(query)
    K = mha.denseK(key)
    V = mha.denseV(value)
    attn_out = multi_head_scaled_dot_attention_causal(mha.nhead, Q, K, V)

    mha.denseO(attn_out)
end

# Convenience: single sample (2-D) -------------------------
function (mha::CausalMultiheadAttention)(
        query::M, key::M, value::M) where {T, M <: AbstractMatrix{T}}
    wrap(x)   = reshape(x, size(x, 1), size(x, 2), 1)
    unwrap(x) = reshape(x, size(x, 1), size(x, 2))
    unwrap(mha(wrap(query), wrap(key), wrap(value)))
end