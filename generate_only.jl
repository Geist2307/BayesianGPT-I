using Pkg; Pkg.activate(@__DIR__)
using Flux, NNlib, Statistics, Functors, Unicode, DataStructures, Random, BSON
using Printf

include("src/layers.jl")
include("src/attention_causal.jl")
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
generate_samples(model, tokenizer, "What am I supposed to do now?";
                 n_samples=5, max_length=20, temperature=1.5f0)

generate_samples_topk(model, tokenizer, "What am I supposed to do now?";
                      n_samples=5, max_length=20, temperature=1.5f0, 
                      k=50, penalty=3.0f0)

bayesian_generate_report(model, tokenizer, "What am I supposed to do now?";
                         n_forward=200, max_length=20, temperature=1.5f0,
                         k=50, penalty=3.0f0)