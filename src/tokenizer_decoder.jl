# ============================================================
# tokenizer_decoder.jl
# Tokenization pipeline for next-token prediction (decoder/GPT style).
#
# Differences from tokenizer.jl:
#   - reads from a plain .txt file instead of a DataFrame
#   - create_batches builds (X, Y) shifted token pairs
#     instead of (tokens, sentiment_label)
#
# Everything else is identical:
#   simplify, select_vocabulary, IndexTokenizer, preprocess, pad_sequences
# ============================================================

using Unicode
using DataStructures
using Flux: DataLoader, onehotbatch


# -----------------------------------
# Text normalisation 
# ---------------------------------

function simplify(s::AbstractString)
    s = lowercase(s)
    s = Unicode.normalize(s, :NFD)
    s = replace(s, r"['`'\u200d\p{M}]" => "")
    s = replace(s, r"\n"               => " ")
    s = replace(s, r"<br\s*/?>"        => " ")
    s
end


# ────────────────────────────────────────────────────────────
# Read corpus from .txt files
# ────────────────────────────────────────────────────────────

"""
    read_corpus(path) → Vector{String}

Read a plain .txt file and return a vector of non-empty lines.
Each line is treated as one document (sentence / paragraph).
"""
function read_corpus(path::AbstractString)
    lines = readlines(path)
    filter(!isempty, strip.(lines))
end


# ────────────────────────────────────────────────────────────
# Vocabulary selection  (unchanged)
# ────────────────────────────────────────────────────────────

"""
    select_vocabulary(corpus; min_document_frequency, pattern, transform)

Build a vocabulary from a corpus of strings.
Only words appearing in at least `min_document_frequency` documents are kept.
Returns a Vector{String} with <PAD> and <UNK> prepended.
"""
function select_vocabulary(
        corpus::AbstractVector{<:AbstractString};
        min_document_frequency :: Int    = 5,
        pattern                :: Regex  = r"\w\w+\b",
        transform = simplify)

    doc_freq = DefaultDict{String, Int}(0)

    for doc in corpus
        seen = Set{String}()
        for m in eachmatch(pattern, transform(doc))
            word = m.match
            if !(word in seen)
                push!(seen, word)
                doc_freq[word] += 1
            end
        end
    end

    filter!(kv -> kv[2] ≥ min_document_frequency, doc_freq)
    vocab_pairs = sort(collect(doc_freq), by = kv -> kv[2], rev = true)
    words       = [kv[1] for kv in vocab_pairs]

    return ["<PAD>", "<UNK>"] ∪ words
end


# ────────────────────────────────────────────────────────────
# IndexTokenizer  (unchanged)
# ────────────────────────────────────────────────────────────

struct IndexTokenizer{T}
    vocabulary :: Vector{T}
    lookup     :: Dict{T, Int}
    unksym     :: T
    unkidx     :: Int

    function IndexTokenizer(vocab::Vector{T}, unksym::T) where T
        if !(unksym ∈ vocab)
            pushfirst!(vocab, unksym)
        end
        unkidx = findfirst(isequal(unksym), vocab)
        lookup = Dict(x => i for (i, x) in enumerate(vocab))
        new{T}(vocab, lookup, unksym, unkidx)
    end
end

Base.length(t::IndexTokenizer) = length(t.vocabulary)

Base.show(io::IO, t::IndexTokenizer) =
    print(io, "IndexTokenizer(vocab_size=$(length(t)), unk=\"$(t.unksym)\")")

function (t::IndexTokenizer)(tokens::Vector{<:AbstractString})
    [get(t.lookup, tok, t.unkidx) for tok in tokens]
end


# ────────────────────────────────────────────────────────────
# Document preprocessing  (unchanged)
# ────────────────────────────────────────────────────────────

function preprocess(
        document::AbstractString,
        tokenizer;
        pattern    :: Regex               = r"\w\w+\b",
        max_length :: Union{Nothing, Int} = nothing,
        transform                         = simplify)

    words  = [m.match for m in eachmatch(pattern, transform(document))]
    tokens = tokenizer(words)

    if !isnothing(max_length) && length(tokens) > max_length
        tokens = tokens[1:max_length]
    end
    tokens
end


# ----------------------------
# Padding  
# --------------------------------

function pad_sequences(sequences, max_length::Int, pad_idx::Int)
    padded = fill(pad_idx, max_length, length(sequences))
    for (i, seq) in enumerate(sequences)
        len = min(length(seq), max_length)
        padded[1:len, i] .= seq[1:len]
    end
    padded
end


# ------------------------
# Next-token training pairs
# ----------------------------

"""
make_pairs(tokens) → (Vector{Int}, Vector{Int})

Split a token sequence into input/target pairs shifted by one:

    tokens = [the, cat, sat, on, mat]
    X      = [the, cat, sat, on]      ← input  (positions 1..N-1)
    Y      = [cat, sat, on,  mat]     ← target (positions 2..N)

This is the core of next-token prediction: given X[t], predict Y[t].
"""
function make_pairs(tokens::Vector{Int})
    X = tokens[1:end-1]
    Y = tokens[2:end]
    X, Y
end


# -----------------------------------
# Batch creation  (decoder version)
# -------------------------

"""
    create_batches(corpus, tokenizer; batch_size, max_length)

Full preprocessing pipeline for next-token prediction.

Returns a `Flux.DataLoader` where each batch is:
- `X` : `(1, max_length-1, batch_size)`  -- input token ids
- `Y` : `(1, max_length-1, batch_size)`  -- target token ids (shifted by 1)

# Arguments
- `corpus`     : Vector{String} of documents (from `read_corpus`)
- `tokenizer`  : an `IndexTokenizer`
- `batch_size` : mini-batch size (default: 32)
- `max_length` : truncate/pad documents to this many tokens (default: 128)

# Note
Documents shorter than 2 tokens after tokenization are skipped
since they cannot form a valid (X, Y) pair.
"""
function create_batches(
        corpus    :: AbstractVector{<:AbstractString},
        tokenizer :: IndexTokenizer;
        batch_size :: Int = 32,
        max_length :: Int = 128)

    pad_idx = tokenizer.lookup["<PAD>"]

    all_X = Vector{Int}[]
    all_Y = Vector{Int}[]

    for doc in corpus
        tokens = preprocess(doc, tokenizer; max_length = max_length + 1)
        length(tokens) < 2 && continue       # skip too-short documents

        X, Y = make_pairs(tokens)
        push!(all_X, X)
        push!(all_Y, Y)
    end

    isempty(all_X) && error("No valid documents found. Check your corpus and vocabulary.")

    seq_len = max_length - 1

    X_pad = pad_sequences(all_X, seq_len, pad_idx)   # (seq_len, n_docs)
    Y_pad = pad_sequences(all_Y, seq_len, pad_idx)   # (seq_len, n_docs)

    # reshape to (1, seq_len, n_docs) to match model input convention
    X_batch = reshape(X_pad, 1, seq_len, :)
    Y_batch = reshape(Y_pad, 1, seq_len, :)

    DataLoader((X_batch, Y_batch); batchsize = batch_size, shuffle = true)
end