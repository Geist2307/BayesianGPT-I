# ============================================================
# layers.jl
# Core building blocks:
#   - Embed                       : learned token embeddings
#   - PositionEncoding            : fixed sinusoidal position encoding
#   - VariationalDropoutMolchanov : Bayesian linear layer (Molchanov et al. 2017)
# ============================================================


# ─------------------------------------
# 1.  Token Embedding
# -------------------------------------

"""
    Embed(output_dim, vocab_size)

A learned lookup table mapping integer token ids to dense vectors.
Parametric assignment means a concrete type is known at compile time,
speeding up things every loop. Much better performance than Python.

# Shape
- Input  : `(seq_len, batch_size)` – integer token ids
- Output : `(embed_dim, seq_len, batch_size)`
"""
struct Embed{W <: AbstractArray}
    weight::W
end

Flux.@layer Embed

# No need to type check at runtime
Embed(output_dim::Int, vocab_size::Int) =
    Embed(randn(Float32, output_dim, vocab_size))


# this is just custom pretty-printing
Base.size(e::Embed) = size(e.weight)

Base.show(io::IO, e::Embed) =
    print(io, "Embed($(size(e.weight, 1)) × $(size(e.weight, 2)))")

# integer keeps the type of integer flexible  
function (e::Embed)(x::AbstractArray{<:Integer})
    # x: (seq_len, batch_size) of token indices
    # Return: (embed_dim, seq_len, batch_size)
    reshape(e.weight[:, vec(x)], size(e.weight, 1), size(x)...) # splat operator
end


# ----------------------------------
# 2.  Sinusoidal Position Encoding
# -----------------------------------

"""
    PositionEncoding(dim_embedding, max_length=1000)

Fixed sinusoidal position encoding (Vaswani et al. 2017).
Not trainable – weights are frozen after construction.

# Formula
    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
"""
struct PositionEncoding{W <: AbstractArray}
    weight::W
end

Flux.@layer PositionEncoding trainable=()          # frozen – never updated

function PositionEncoding(dim_embedding::Int, max_length::Int = 1000)
    W = _make_sinusoidal(dim_embedding, max_length)
    PositionEncoding(W)
end

function _make_sinusoidal(dim_embedding::Int, seq_length::Int, n::Int = 10_000)
    encoding = Matrix{Float32}(undef, dim_embedding, seq_length)
    for pos in 1:seq_length
        for row in 0:2:(dim_embedding - 1)
            denom = 1f0 / (n ^ (row / dim_embedding))
            encoding[row + 1, pos] = sin(pos * denom)
            encoding[row + 2, pos] = cos(pos * denom)
        end
    end
    encoding
end

Base.show(io::IO, pe::PositionEncoding) =
    print(io, "PositionEncoding(dim=$(size(pe.weight, 1)), max_len=$(size(pe.weight, 2)))")

# Called with a 3-D array  (embed_dim, seq_len, batch),  add positional slice
function (pe::PositionEncoding)(x::AbstractArray{T, 3}) where T
    seq_len   = size(x, 2)
    max_length = size(pe.weight, 2)
    seq_len > max_length && error(
        "sequence length $seq_len exceeds max position encoding length $max_length")
    x .+ pe.weight[:, 1:seq_len]           # broadcast over batch dimension
end

# Called with an integer -> return the positional encoding slice
function (pe::PositionEncoding)(seq_length::Int)
    max_length = size(pe.weight, 2)
    seq_length > max_length && error(
        "sequence length $seq_length exceeds max position encoding length $max_length")
    pe.weight[:, Base.OneTo(seq_length)]
end


# --------------------------------------------------------
# 3.  Variational Dropout Layer  (Molchanov et al. 2017)
# ------------------------------------------------

"""
VariationalDropoutMolchanov(in, out, activation=identity)

A Bayesian linear layer that learns a *distribution* over weights rather
than point estimates.

Each weight is modelled as a Gaussian distribution with given mean and variance:

    W_ij ~ N(θ_ij,  σ²_ij)

During the forward pass we use the **reparameterisation trick** to sample from the distribution:

    W = θ + exp(0.5 · log σ²) · ε,   ε ~ N(0, I)

so gradients flow through `θ` and `log σ²`.

The signal-to-noise ratio `α = σ² / θ²` (equivalently `log α = log σ² - 2 log |θ|`)
drives sparsity: when `log α` is large the weight is pure noise and can be pruned.
"""
struct VariationalDropoutMolchanov{F, M <: AbstractMatrix, V <: AbstractVector}
    θ      :: M
    logσ2  :: M
    bias   :: V
    activation :: F
end

@functor VariationalDropoutMolchanov

# GPU compatibility 
import Adapt

function Adapt.adapt_structure(to, layer::VariationalDropoutMolchanov)
    VariationalDropoutMolchanov(
        Adapt.adapt(to, layer.θ),
        Adapt.adapt(to, layer.logσ2),
        Adapt.adapt(to, layer.bias),
        layer.activation
    )
end

# Constructor
function VariationalDropoutMolchanov(
        in::Int, out::Int, activation = identity;
        θ_init    = Flux.glorot_uniform,
        logσ2_init = () -> -10f0)

    θ     = θ_init(out, in)
    logσ2 = fill(logσ2_init(), out, in)
    bias  = zeros(Float32, out)
    VariationalDropoutMolchanov(θ, logσ2, bias, activation)
end

Base.show(io::IO, l::VariationalDropoutMolchanov) =
    print(io, "VariationalDropout($(size(l.θ, 2))→$(size(l.θ, 1)), $(l.activation))")

#  Forward pass (2-D matrix input)
function (layer::VariationalDropoutMolchanov)(x::AbstractMatrix)
    ϵ = randn(Float32, size(layer.θ))    # sample noise
    W = @. layer.θ + exp(0.5f0 * layer.logσ2) * ϵ   # reparameterise
    layer.activation.(W * x .+ layer.bias)
end

# Forward pass (3-D batch: embed_dim × seq_len × batch)
function (layer::VariationalDropoutMolchanov)(x::AbstractArray{<:Real, 3})
    map(b -> layer(view(x, :, :, b)), 1:size(x, 3)) |>
        slices -> cat(slices...; dims = 3)
end



