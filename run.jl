# run.jl
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Flux, NNlib, Statistics, Functors, Unicode, DataStructures, Random
using BSON: @save

# ==================
# hyperparams
# ===========================
const MIN_DOC_FREQ  = 10
const BATCH_SIZE    = 32
const MAX_LENGTH    = 64
const EMBED_DIM     = 128 
const N_HEADS       = 4
const HIDDEN_DIM    = 4 * EMBED_DIM
const EPOCHS        = 50
const KL_WARMUP_EPOCHS = 20 # where the KL term is gradually increased
const LR            = 0.6 * 1e-4
const DROPOUT       = 0.3 # more aggresive
const MIN_DOC_LENGTH = 10 # document length in words
const MIN_CHARS = 200 # minimum paragraph length to be extracted from corpus


# load all source files
include("src/layers.jl")
include("src/attention_causal.jl")
include("src/transformer_decoder.jl")
include("src/tokenizer_decoder.jl")
include("src/training_decoder.jl")
include("src/generate.jl")

# data : all text files
corpus = vcat([read_corpus(joinpath("data", f); min_chars = MIN_CHARS) 
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

vocab     = select_vocabulary(train_corpus; min_document_frequency = MIN_DOC_FREQ)
tokenizer = IndexTokenizer(vocab, "<UNK>")
println("Vocab size: $(length(vocab))")

train_loader = create_batches(
    train_corpus, tokenizer; 
    batch_size=BATCH_SIZE,
     max_length=MAX_LENGTH,
     min_length = MIN_DOC_LENGTH)


val_loader   = create_batches(
    val_corpus,  tokenizer;
     batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    min_length = MIN_DOC_LENGTH)

#  model hyperparams
pe  = PositionEncoding(EMBED_DIM)

model = Chain(
    x -> reshape(x, size(x, 2), size(x, 3)),
    Embed(EMBED_DIM, length(vocab)),
    PositionEncoding(EMBED_DIM),   # self-contained layer, no closure
    Dropout(0.1f0),
    TransformerDecoderBlock(N_HEADS, EMBED_DIM, HIDDEN_DIM; pdrop=DROPOUT), # VariationalMLP inside these
    TransformerDecoderBlock(N_HEADS, EMBED_DIM, HIDDEN_DIM; pdrop=DROPOUT),
    Dense(EMBED_DIM, length(vocab)),
)

total_params = sum(p -> length(p), Flux.trainable(model))
println("Total parameters: $total_params")




#  train (CPU only)
history = train_decoder!(
    model, train_loader, val_loader;
    epochs = EPOCHS,
    kl_warmup_epochs = KL_WARMUP_EPOCHS,
    lr     = LR,
)





# -- save
@save "model.bson" model
println("Model saved")
@save "tokenizer.bson" tokenizer
println("Model and tokenizer saved")



# generate (one test generation)
println("\n--- Generation ---")
generate_samples(model, tokenizer, "tell me about yourself";
                 n_samples=5, max_length=40, temperature=1.2f0)