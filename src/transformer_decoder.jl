# ============================================================
# transformer_decoder.jl
# TransformerDecoderBlock : causal attention + variational MLP
#
# Identical to TransformerBlock except:
#   - Uses CausalMultiheadAttention (masked) instead of MultiheadAttention
#   - No pooling — preserves full (dm, N, B) output so we can
#predict the next token at every position
# ============================================================

using Flux: LayerNorm, Dropout


"""
TransformerDecoderBlock(nhead, dm, dhid; pdrop=0.1)

One GPT-style transformer decoder block:

    x -> CausalMHA(x,x,x) -> dropout _> residual - > LayerNorm
      -> VariationalMLP    ->          residual -> LayerNorm

Token at position t can only attend to positions ≤ t (causal mask).
Output shape is identical to input: (dm, N, B).

# Arguments
- `nhead`  : number of attention heads
- `dm`     : model (embedding) dimension
- `dhid`   : hidden dimension of the feed-forward MLP (typically 4·dm)
- `pdrop`  : dropout probability applied after attention
"""
struct TransformerDecoderBlock{
        MHA <: CausalMultiheadAttention,
        N1  <: LayerNorm,
        D1  <: VariationalDropoutMolchanov, # this is the implemented layer
        D2  <: VariationalDropoutMolchanov,
        N2  <: LayerNorm,
        DO  <: Dropout}

    causal_attention :: MHA
    norm_attention   :: N1
    dense1           :: D1   # dm  → dhid  (with relu)
    dense2           :: D2   # dhid → dm
    norm_feedforward :: N2
    dropout          :: DO
end

Flux.@layer TransformerDecoderBlock

function TransformerDecoderBlock(nhead::Int, dm::Int, dhid::Int; pdrop::Float64 = 0.1)
    TransformerDecoderBlock(
        CausalMultiheadAttention(nhead, dm, dm),
        LayerNorm(dm),
        VariationalDropoutMolchanov(dm, dhid, relu),
        VariationalDropoutMolchanov(dhid, dm),
        LayerNorm(dm),
        Dropout(pdrop),
    )
end

function Base.show(io::IO, b::TransformerDecoderBlock)
    print(io, "TransformerDecoderBlock(")
    print(io, b.causal_attention, ", ")
    print(io, b.dense1, ", ")
    print(io, b.dense2, ")")
end


# -------------
# Forward pass
# ----------------

"""
Forward pass of one decoder block.

    Input  x : (dm, N, B)
    Output h : (dm, N, B) , same shape, no pooling

Steps:
  1. Causal self-attention -- token t only sees positions <= t
  2. Dropout + residual + LayerNorm
  3. Variational feed-forward MLP
  4. Residual + LayerNorm
"""
function (t::TransformerDecoderBlock)(x::AbstractArray)
    # 1. Causal multi-head self-attention 
    h = t.causal_attention(x, x, x)       # (dm, N, B)

    # 2. Dropout + residual + norm 
    h = t.dropout(h)
    h = x + h
    h = t.norm_attention(h)               # (dm, N, B)

    # 3. Variational feed-forward MLP 
    ## these are the only layers that contain variational dropout
    hff = t.dense1(h)                     # (dhid, N, B)
    hff = t.dense2(hff)                   # (dm,   N, B)

    # 4. Residual + norm 
    h = h + hff
    h = t.norm_feedforward(h)             # (dm, N, B)
    h
end