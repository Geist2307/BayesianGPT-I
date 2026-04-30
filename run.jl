# run.jl
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Flux, NNlib, Statistics, Functors, Unicode, DataStructures, Random
using BSON: @save


# load all source files
include("src/layers.jl")
include("src/attention.jl")
include("src/attention_causal.jl")
include("src/transformer.jl")
include("src/transformer_decoder.jl")
include("src/tokenizer_decoder.jl")
include("src/training_decoder.jl")
include("src/generate.jl")

# data : all text files
corpus = vcat([read_corpus(joinpath("data", f)) 
               for f in readdir("data") 
               if endswith(f, ".txt")]...)

println("Total lines: $(length(corpus))")
println("Files loaded: $(filter(f -> endswith(f, ".txt"), readdir("data")))")

## Let's seed it 

Random.seed!(42)
corpus = corpus[randperm(length(corpus))] # randomize lines

split_idx    = Int(floor(0.9 * length(corpus)))
train_corpus = corpus[1:split_idx]
val_corpus   = corpus[split_idx+1:end] # 10% for validation 

vocab     = select_vocabulary(train_corpus; min_document_frequency = 3)
tokenizer = IndexTokenizer(vocab, "<UNK>")
println("Vocab size: $(length(vocab))")

train_loader = create_batches(train_corpus, tokenizer; batch_size=16, max_length=256)
val_loader   = create_batches(val_corpus,   tokenizer; batch_size=16, max_length=256)

#  model hyperparams
embed_dim  = 64
n_heads    = 4
hidden_dim = 4 * embed_dim
pe         = PositionEncoding(embed_dim)

model = Chain(
    x -> reshape(x, size(x, 2), size(x, 3)),
    Embed(embed_dim, length(vocab)),
    PositionEncoding(embed_dim),   # self-contained layer, no closure
    Dropout(0.1f0),
    TransformerDecoderBlock(n_heads, embed_dim, hidden_dim; pdrop=0.1), # VariationalMLP inside these
    TransformerDecoderBlock(n_heads, embed_dim, hidden_dim; pdrop=0.1),
    Dense(embed_dim, length(vocab)),
)

total_params = sum(length, Flux.params(model))
println("Total parameters: $total_params")




#  train (CPU only)
history = train_decoder!(
    model, train_loader, val_loader;
    epochs = 50,
    lr     = 1e-3,
)





# -- save
@save "model.bson" model
println("Model saved")
@save "tokenizer.bson" tokenizer
println("Model and tokenizer saved")



# generate (one test generation)
println("\n--- Generation ---")
generate_samples(model, tokenizer, "he could not remember";
                 n_samples=5, max_length=40, temperature=0.8f0)