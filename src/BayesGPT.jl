module BayesGPT

using Flux
using NNlib
using Statistics
using Functors
using Unicode
using DataStructures
using Random
using Printf
using Adapt

include("layers.jl")
include("attention_causal.jl")
include("transformer_decoder.jl")
include("tokenizer_decoder.jl")
include("training_decoder.jl")
include("generate.jl")

export
    # Layers
    Embed,
    PositionEncoding,
    VariationalDropoutMolchanov,
    # Attention
    CausalMultiheadAttention,
    causal_mask,
    # Transformer
    TransformerDecoderBlock,
    # Tokenizer
    simplify,
    select_vocabulary,
    IndexTokenizer,
    preprocess,
    pad_sequences,
    create_batches,
    read_corpus,
    # Training
    kl,
    decoder_energy_loss,
    layer_sparsity,
    get_mlp_sparsities,
    evaluate_decoder,
    train_decoder!,
    kl_schedule,
    # Generation
    sample_token,
    sample_token_topk,
    sample_token_topk_norep,
    generate,
    generate_topk,
    generate_samples,
    generate_samples_topk,
    bayesian_generate,
    bayesian_generate_report

end
