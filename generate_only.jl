using Pkg; Pkg.activate(@__DIR__)
using Flux, NNlib, Statistics, Functors, Unicode, DataStructures, Random, BSON
using Printf

include("src/layers.jl")
include("src/attention.jl")
include("src/attention_causal.jl")
include("src/transformer.jl")
include("src/transformer_decoder.jl")
include("src/tokenizer_decoder.jl")
include("src/generate.jl")

# load tokenizer
BSON.@load "tokenizer.bson" tokenizer
println("Tokenizer loaded ✓ — vocab size: $(length(tokenizer))")

# load saved model
BSON.@load "model.bson" model
println("Model loaded ✓")

# generate

# ── standard (shows repetition problem) ──────────────────────
generate_samples(model, tokenizer, "he knows";
                 n_samples=3, max_length=3, temperature=0.8f0)

generate_samples_topk(model, tokenizer, "he knows";
                      n_samples=3, max_length=3, temperature=0.8f0, 
                      k=50, penalty=1.5f0)

bayesian_generate_report(model, tokenizer, "he knows";
                         n_forward=20, max_length=3, temperature=0.8f0,
                         k=50, penalty=1.5f0)