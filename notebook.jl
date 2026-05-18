### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 7f226751-ed4a-4aea-a68e-95f565ae552d
using Pkg

# ╔═╡ cab68644-80d1-48c4-9bd0-7cd2210679a3
Pkg.activate(@__DIR__)

# ╔═╡ 728fd22e-aefc-4c3c-a319-0dcafbb3f4ef
Pkg.add("Plots")

# ╔═╡ 3b813e4d-0418-4f07-9bbd-d13793fe832e
Pkg.add("Languages")

# ╔═╡ 4e314c12-3d11-4eb6-b667-37dd9ec4a6f0
using BayesGPT

# ╔═╡ 9c165c1d-a8f6-4f6c-be8e-951760508edd
begin 
	using Unicode
	using DataStructures
end

# ╔═╡ d323dfa2-163f-4656-b53c-920639f354d0
using Plots

# ╔═╡ 3dbf0138-5600-421b-8783-0dd82193a8b4
using Languages

# ╔═╡ 8f7f3ea5-b42b-42a6-96c8-73da38fd3e47
using Flux: DataLoader, onehotbatch


# ╔═╡ 95a43dc9-5566-40f5-8496-21af172acb1a
using Statistics

# ╔═╡ 115336ef-45bd-4084-a8ee-d5c02b8e5101
using Flux

# ╔═╡ 4de8f4aa-2ef8-408a-8119-3e16322f6c39
include("src/tokenizer_decoder.jl")

# ╔═╡ a0bb92ec-505e-11f1-3704-8b56700c9b37
md""" 
# Tokenizer and data
"""

# ╔═╡ 28db7890-7d6d-4085-b74c-a85939441ba3
# ╠═╡ disabled = true
#=╠═╡
using Pkg
  ╠═╡ =#

# ╔═╡ 2fc40ef1-0ca4-4748-95ee-4a248473551b
md"""
## Upload data

- We will first setup a way to upload data from text
- This is to get words and frequecies
- We will do data processing and understand how words appear
"""

# ╔═╡ 5de57b9b-d1f5-4470-a80a-d6ca9e741058
md"""

Now, we want to apply this function to the path. The path is data and contains
exactly three files

"""


# ╔═╡ 3aabaf33-ebaa-41b7-991d-b4c184279e20
function read_corpus_new(path::AbstractString; min_chars::Int = 200)
	
    text = read(path, String)
    
    # strip Gutenberg boilerplate
    start_match = findfirst(r"\*\*\* START OF.*?\*\*\*", text)
    end_match   = findfirst(r"\*\*\* END OF.*?\*\*\*", text)
    if start_match !== nothing && end_match !== nothing
        text = text[last(start_match)+1 : first(end_match)-1]
    end
    
    # split into paragraphs and filter short ones
    paragraphs = split(text, r"\n\s*\n")
    filter(p -> length(strip(p)) >= min_chars, strip.(paragraphs))
end

# ╔═╡ 8950fecd-7034-4bd2-90ac-8feb9f76d635
md"""
Let's see how this splits my corpus
"""

# ╔═╡ bc163526-d1d5-41e4-9426-5bdeda318215
lines_one_new = read_corpus_new("data/input.txt") 

# ╔═╡ f77d1b1a-8962-46c3-843a-e215771a4d2e
for (j, line) in enumerate(lines_one_new)
	println("Line $j | length=$(length(line)) chars")
	println(first(line, 80))  # preview first 80 chars
	println()
end

# ╔═╡ 39b63a61-61ad-4c57-98c3-993e504e358a
length(lines_one_new)

# ╔═╡ ea20020c-6917-496b-ad0e-af94b2c1627b
md"""
Let's then simply upload all files and see how many lines we have. We will use the new corpus function that selectively uses longer lines.
"""

# ╔═╡ c7dda7f8-bb2e-4396-82d1-ad26085af609
corpus = vcat([read_corpus_new(joinpath("data", f)) 
               for f in readdir("data") 
               if endswith(f, ".txt")]...)

# ╔═╡ 95fb4cd5-57cc-4523-8cd4-cf23b132aee6
println("Total lines: $(length(corpus))")

# ╔═╡ 98b36bee-192e-4360-9e48-f9cd938044e9
println("Files loaded: $(filter(f -> endswith(f, ".txt"), readdir("data")))")

# ╔═╡ 998f204e-fbf3-4c00-9a61-a893afa613af
md"""
Notice:

- 8663 lines of text
- 11 files loaded
- higher quality paragraphs (longer)

This is a very small corpus for our transformer, but lines are higher quality.

"""

# ╔═╡ 21552fdb-ec7e-43ce-b2c2-e229b4df12f4
md"""
## Tokenize
"""

# ╔═╡ a47fc023-0eae-4045-98a6-9dd4d5283425
md"""
Let's define a simple tokenizer. We will select one token per word.
"""

# ╔═╡ e971fcd8-8440-4a62-abe9-f606964e1ae6
md"""
First, a function to simplify my text (normalise it)
"""

# ╔═╡ dd0e3d73-e972-4943-825e-5684a19ebf7f
md"""
### Simplify
"""

# ╔═╡ ed470a47-9bca-4d32-8018-60648914aaa2
function simplify(s::AbstractString)

	s = lowercase(s) # make everything lower case
	s = Unicode.normalize(s, :NFD)
	s = replace(s, r"['`'\u200d\p{M}]" => "")
    s = replace(s, r"\n"               => " ")
    s = replace(s, r"<br\s*/?>"        => " ")
    s
end

# ╔═╡ 562bb060-881d-44d2-88c7-76f75535311c
md"""
Let's test this function
"""

# ╔═╡ a10ec7d7-44d5-44db-8553-a23f7bb431d4
lines_one_new

# ╔═╡ 857d0d6c-46a6-44db-8685-b3aa53a121d4
# ╠═╡ disabled = true
#=╠═╡
simplify_text = simplify("The Golden Man")
  ╠═╡ =#

# ╔═╡ cf8a8b73-10dc-44d1-b0a2-6df5917d4d74
simplify_text = simplify("What is my purpose during spring?")

# ╔═╡ 5e9bcaf4-14fb-4a19-b246-251494a9c0e7
md"""
### Vocabulary
"""

# ╔═╡ f0824b60-04b9-45d0-9355-2140df4c6a4e
md"""
Let's select a vocabulary from the corpus
"""

# ╔═╡ 1dbef844-fdad-4492-b7d2-4349e1a380c4
"""
    select_vocabulary(corpus; min_document_frequency, pattern, transform)

Build a vocabulary from a corpus of strings.
Only words appearing in at least `min_document_frequency` documents are kept.
Returns a Vector{String} with <PAD> and <UNK> prepended.
"""
function select_vocabulary(
        corpus::AbstractVector{<:AbstractString};
        min_document_frequency :: Int    = 1,
        pattern                :: Regex  = r"\w\w+\b", # word with boundary
        transform = simplify)

    doc_freq = DefaultDict{String, Int}(0) # simple dict

    # for each doc in corpus (line)
	# we update flag; if word is seen increase
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

    filter!(kv -> kv[2] ≥ min_document_frequency, doc_freq) #filter frequencies
    vocab_pairs = sort(collect(doc_freq), by = kv -> kv[2], rev = true)
    words       = [kv[1] for kv in vocab_pairs]
	freq 		= [kv[2] for kv in vocab_pairs]

    return ["<PAD>", "<UNK>"] ∪ words, freq
end

# ╔═╡ f5ee80c0-6f5a-4b04-8d22-70048eaafa95
select_vocabulary(corpus)

# ╔═╡ 2d561449-c043-4333-8cb9-14a7c5c04988
words, freqs = select_vocabulary(corpus)

# ╔═╡ 2b7e2ff9-0e91-47bf-9d5d-05b4cdf62d83
println(words)

# ╔═╡ 2a761ed3-575a-48c5-8a75-19c84b164149
println(freqs)

# ╔═╡ 07f07562-8870-427a-8a5a-13e59cd79fa0
# Get the frenquency dict
doc_freq = DataStructures.DefaultDict{String, Int}(0)

# ╔═╡ 283050e3-45ce-4d7b-81c9-501b43067b0e
for doc in corpus
    seen = Set{String}()
    for m in eachmatch(r"\w\w+\b", simplify(doc))
        word = m.match
        if !(word in seen)
            push!(seen, word)
            doc_freq[word] += 1
        end
    end
end

# ╔═╡ 8340a775-f5a7-48f4-be2c-fe3d5cd09920
top20 = sort(collect(doc_freq), by = kv -> kv[2], rev = true)[1:20]

# ╔═╡ 549a5c3b-8a5e-4ddc-ae20-800013c2370a
bar(
    [kv[1] for kv in top20],
    [kv[2] for kv in top20],
    xlabel = "Word",
    ylabel = "Document frequency",
    title  = "Top 20 words",
    xrotation = 45,
    legend = false
)

# ╔═╡ cd99e93a-320b-4728-a8cb-fa260ee840fc
md"""
Notice:

- The most frequent are actually stop words, which is expected
- A different tokenization strategy would have looked various parts, allowing the GPT to predict next useful part (effectively building words where none exist)
"""

# ╔═╡ 3130fe4b-741c-41a1-ab5d-7789a992e72a
md"""

Let's quickly check that simplify works correctly for these
"""

# ╔═╡ c33f9c8e-9762-403b-a1c9-e3af8f341768
simplify("The")

# ╔═╡ e51cf592-788e-4b3f-8408-cc1c8f177c96
simplify("of")

# ╔═╡ 3594fbf3-d9ac-410e-977e-6a8016e122d4
md"""
Note:

- It works very nicely
"""

# ╔═╡ f4a85f67-1204-4e57-a8ef-3bae5e796e1a
stopwords_corps = stopwords(Languages.English())

# ╔═╡ e55bb10c-2ca9-4112-b5a3-433d6d12bfa3


# ╔═╡ 906eb80a-d0fe-45a0-9239-64afaffcbe46
top20_wstop = sort(
    filter(kv -> kv[1] ∉ stopwords_corps, collect(doc_freq)),
    by = kv -> kv[2],
    rev = true
)[1:20]

# ╔═╡ 08251321-0d8f-4659-b152-3ecbaeda99b9


# ╔═╡ 6e2b9b9e-459d-4135-a9d2-972481669031
bar(
	[kv[1] for kv in top20_wstop],
	[kv[2] for kv in top20_wstop],
	xlabel="Word",
	ylabel="Document frequency",
	title="Top 20 words excluding stopwords",
	xrotation=45,
	legend=false,
	xticks= (1:20, [kv[1] for kv in top20_wstop]),
    xtickfontsize= 7,
	size=(1500, 800),
	bottom_margin = 10Plots.mm
)

# ╔═╡ 463958e0-1e82-4292-8703-9187db183a1b
md"""

- gutenberg and project are the most common words, which can be seen as skewing the data
- so the files need to be better pre-processed for an authentic encoder-like transformer, however we are using this small corpus for learning sentence strcture

"""

# ╔═╡ ad797ef1-43c1-4ba8-baea-f4313ac0a880
md"""
## Index tokenizer
"""

# ╔═╡ 27f0617b-bf40-467f-83bb-7f6b3ae243da
md"""
Previously we have selected all the words
"""

# ╔═╡ d2374e61-23e5-456f-9766-1f9f49e4e3bd
words

# ╔═╡ 156d2301-df64-4371-be1b-7ab4b4f263e4


# ╔═╡ 992933d9-f3f0-4ae5-858a-354036812a82
"of" ∈ words

# ╔═╡ 8c6c5f89-5ede-412f-9e39-295b85eacb3a
!("of" ∈ words)

# ╔═╡ 2336969f-373d-43f8-af42-38da967cf201
md"""
Simple expression to effectively test if symbol is in vocab or not
"""

# ╔═╡ 298ccf31-9c8c-4006-b918-a1510a1f52d7
findfirst(isequal("of"), words)

# ╔═╡ d03aae98-3c91-486f-a79b-40717516503d
words[1:5]

# ╔═╡ 03513e58-2cec-4ec0-b8a2-a1f43fc0cd45
md"""
Literally the forth position (notice list starts at 1)
"""

# ╔═╡ 6cb4b1b0-cf5d-4d50-acc1-42e5d431cc6a
md"""
Two types of dictionaries can be created in Julia:

- untyped
- typed
"""

# ╔═╡ 6d6e2afe-68ad-4612-b194-6c77e84ad3fe
# let's create an untyped collection of key/value pairs (dicts in Julia)

test_dict = Dict("the" => 324, "of" => 567)

# ╔═╡ 4d9cab90-23f9-422c-95e7-195899669ece
test_dict_typed = Dict{String, Integer}("the" => 324, "of" => 567)

# ╔═╡ 9bd3a45e-8931-49ce-b5b8-5afc3aa0fb9c
enumerate(words)

# ╔═╡ eceafa6f-bfd8-4af8-9e17-cda738d28286


# ╔═╡ 790eaedc-7469-4710-ac15-3a2e72650460
function test_tokenizer_logic(vocab::Vector{T}, unksym::T) where T

	# if symbol is not in vocab, push it there
    if !(unksym ∈ vocab)
        pushfirst!(vocab, unksym)
    end

	# find the first id of this vocab
    unkidx = findfirst(isequal(unksym), vocab)

	# create a lookup dictionary, that associates word to token
	
    lookup = Dict(x => i for (i, x) in enumerate(vocab))

	# return the new vocab
    return (vocab=vocab, lookup=lookup, unksym=unksym, unkidx=unkidx)
end

# ╔═╡ 1afe2ee3-b0a1-4c37-adcc-c5e59550c5b6
md"""
Notice:

- Takes the vocabulary and and (unknown) symbol
- If the  symbol is not in the vocabulary(so unknown) it pushed it there
- finds the first index of the unknown vocabulary
- creates a new (implicitly typed) Dictionary. The typed structure is actually handles by the IndexTokenizer{T} struct. Lookup is actually a dictionary of T and integer. Caches errors early
- [Structs](https://forem.julialang.org/ifihan/structs-in-julia-2ojf) in Julia are very powerful. They let you create your own data type. simillar to classes in Python, but structs cannot be changed afterwards
"""

# ╔═╡ d224feaa-921f-4dfa-8b32-21864374b725
"<UNK>" ∈ words

# ╔═╡ d59f83d6-f1f8-4af1-bc1f-9909c471d750
typeof(words[1])

# ╔═╡ 24b1d515-9459-483a-86da-50629bb43718
findfirst(isequal("sky"), words)

# ╔═╡ 6598ae53-b6f2-4740-80e5-1d7ed945f75a
md"""
Notice

- "sky" is element 450 in the bag of words( the dictionary)
"""

# ╔═╡ 96e6314d-1b2a-48ba-90e3-f72791ef6799
filter(w -> occursin("sky", w), words) # we filter only that word

# ╔═╡ a5317aae-f953-40d6-ab03-def6f2581189
md"""
Let's test the behaviour of this where:

- We add the word "Romania" (name of a country, likely does not exist in corpus)
- We see how the `tokenizer` objects is created with current bag of words and Romania
"""

# ╔═╡ 6398b420-4bbe-4595-8ebd-b751eace4e4e
md"""
Note:

- The unknown word is actually added at the beggining of my words
"""

# ╔═╡ 2a59ee46-5c99-4fc9-9a9c-471db4aced36
md"""
Let's see how an encoding/decoding function would look like
"""

# ╔═╡ 4203b46e-fa8d-4229-a9b0-9cb7dd7c2f8f
# encode a sentence
sentence = ["the", "quick", "brown", "fox"]

# ╔═╡ b8c8f36f-3a91-4382-8ad5-85d3bd5e5249
md"""
Note:

- A simple encoder/decoder function is very easy to implement
- But we want to take advantage of the structure of the test
- Below, let's examine the first words from the extracted vocabulary
"""

# ╔═╡ a0e295b0-d6fa-46c8-8068-4f24a5b9a3a0
words 

# ╔═╡ ffd47e13-7b7f-4a87-b242-378a873a2366
md"""
### Preprocess
"""

# ╔═╡ f61415cb-8ec2-4728-9a5f-55a560d9b40e
function preprocess(
        document::AbstractString,
        tokenizer;
        pattern    :: Regex               = r"\w\w+\b", # word with boundary
        max_length :: Union{Nothing, Int} = nothing,
        transform                         = simplify) # the only transforms is simplify 

    words  = [m.match for m in eachmatch(pattern, transform(document))]
    tokens = tokenizer(words)

    # if tokens greater than max length, we clip them
	if !isnothing(max_length) && length(tokens) > max_length
        tokens = tokens[1:max_length]
    end
    tokens
end

# ╔═╡ f346ff56-e442-40e5-a580-cc3bcaee9a32
md"""
Let's test this function with a simple expression
"""

# ╔═╡ b9f51f4a-59d1-49bd-82db-c4994465fcb1
# ╠═╡ disabled = true
#=╠═╡
doc = "The quick brown fox jumps over the lazy dog"
  ╠═╡ =#

# ╔═╡ 15b404d1-9e45-41d4-bf45-40316fa58392
# ╠═╡ disabled = true
#=╠═╡
using Flux
  ╠═╡ =#

# ╔═╡ f4772a57-3db2-44df-b531-e0e47d110bc1
export IndexTokenizer

# ╔═╡ 548bacd4-5fa5-423f-abd7-3fac78c56a4e
tokenizer = IndexTokenizer(words, "<UNK>")

# ╔═╡ 916855d4-3a4d-414a-9c1a-f2e29f6be03e
md"""

We will test in two ways:

- apply preprocess to the words as a sentences
- apply lookup to each word


"""

# ╔═╡ c138f8da-b282-4707-96cf-1f2534ae8904
tokenizer.lookup["the"]

# ╔═╡ cb217241-cfec-4634-8bb2-5ccf1c17d762
tokenizer.lookup["quick"]

# ╔═╡ 783932db-bbc2-4ba1-81ca-109430d5bc28
tokenizer.lookup["brown"]

# ╔═╡ 38a7bf73-2295-422c-9eeb-7b40396c2bd0
tokenizer.lookup["over"]

# ╔═╡ e8e09494-55a2-437e-a598-a662ab777e56
tokenizer.lookup["the"]

# ╔═╡ 3df679bb-a566-49b0-804c-3cabbfdae9bd
tokenizer.lookup["dog"]

# ╔═╡ 6eeebf2c-9424-4f53-b9a2-83ab0c3dd9f5
# ╠═╡ disabled = true
#=╠═╡
doc = "the man walked into the room and sat down"
  ╠═╡ =#

# ╔═╡ ffbfa835-e912-4011-b8fc-bfad82447e4d
md"""
- The sentence above has been tokenized
- we can look up each token and confirm this is correct
"""

# ╔═╡ c21e38b7-2645-470c-8664-437d6545a2eb
tokenizer.lookup["the"]

# ╔═╡ 1d3bd1c4-f49e-4d30-ad75-58b062e8ca4d
tokenizer.lookup["man"]

# ╔═╡ 2713f1cd-4d91-45e6-a32c-334a1ecb542e
tokenizer.lookup["walked"]

# ╔═╡ 182eae98-cd26-4788-a676-4fefbfc20131


# ╔═╡ 79a3cf96-983d-4b6f-a500-9e4060241f1c
length(words)

# ╔═╡ d854c5b1-9104-48d1-93c9-90cbd4758b8d
words

# ╔═╡ 06652db7-e2b1-413e-93f5-3f825f7c3cff
md"""
### Summary

- Very naive implementation of tokenization
- Each token is exactly one word
- think that GPT3 has about 50000 tokens. Our GPT has 9774 tokens
- "the" is then by far the most common (if we would have selected subwords), we would have had "th" for example
"""

# ╔═╡ 46d520fa-e2cd-4c64-9e85-52b7f2cde687
md"""
## Batch creation
"""

# ╔═╡ 3e25408c-a09c-4b1c-a83f-fd81cc65ba18
"""
Padding ensures we have sequences of the same length
"""

# ╔═╡ 3adfa881-f904-440a-9a9a-b2c2fa1dff4c
function pad_sequences(sequences, max_length::Int, pad_idx::Int)
    padded = fill(pad_idx, max_length, length(sequences))
    for (i, seq) in enumerate(sequences)
        len = min(length(seq), max_length)
        padded[1:len, i] .= seq[1:len]
    end
    padded
end

# ╔═╡ 2da79be5-36c8-4066-9dde-1721bf0e7857

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

# ╔═╡ 58bdb6a6-c04b-4b73-a23c-71c95ae69f87
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

# ╔═╡ efbcdf50-c46f-4642-80fd-17a5252cf20f
corpus

# ╔═╡ 13b8a87c-4a25-4987-8a51-e5ac1f3cd122
# ╠═╡ disabled = true
#=╠═╡
loader = create_batches(corpus, tokenizer; batch_size=4, max_length=32)
  ╠═╡ =#

# ╔═╡ 56704f9b-cb72-45bd-b6c4-286fc2928563
md"""
Now that we tested the functions let us: 

- look at each line of code
- understand what it does
"""

# ╔═╡ b899fc16-a7ff-4acb-9690-67727cac32ff
# find the PAD character id
# we can use the tokenizer object

tokenizer.lookup["<PAD>"]

# ╔═╡ d5750bc2-6d2a-417a-b64c-133da58571af
md"""
This is actually the very first element in our tokenizer
"""

# ╔═╡ dae95712-a107-439e-a590-332f9b9cfb5f
all_X = Vector{Int}[]

# ╔═╡ 7806898f-0abf-427e-b9b0-e59c94369fbd
all_Y = Vector{Int}[]

# ╔═╡ 08e6dab3-54e4-422d-a559-c4c309571ec9
md"""
Notice:

- In Julia, we define a priori that these are lists which contain vectors that 
consist of integers! this is powerful, because if something goes badly along 
the way, the compiler throws an error
- This is unlike Python, where types are more flexible

"""

# ╔═╡ f28d0ea8-21cc-4ed1-b4dd-a7e490048204
# let's print out some docs in corpus

# ╔═╡ d5244840-8473-4b81-89f2-40a72428f849
for doc in corpus[1:10]
	println(doc)
end

# ╔═╡ 932ef313-e007-4253-ae20-3f67e21cd71f
# Cell 1 — token lengths for first 10 docs
for (j, doc) in enumerate(corpus[1:10])
    tokens = preprocess(doc, tokenizer; max_length=129)
    println("Doc $j | $(length(tokens)) tokens")
end

# ╔═╡ f932ad27-6bfe-407e-8eaa-e622b2ffa167
lengths = [length(preprocess(doc, tokenizer; max_length=129)) for doc in corpus]

# ╔═╡ 50cae8f3-da93-4337-857f-14fe39e82d4b
println("mean:   $(round(mean(lengths), digits=1))")

# ╔═╡ c186a039-d9dd-4f83-86ec-4a6b47d3a46e
println("median: $(median(lengths))")

# ╔═╡ d251e3f3-6c6b-4e58-8ff4-949cd186ea57
println("min:    $(minimum(lengths))")

# ╔═╡ 353069da-dad5-4f44-aeff-91f11299ece2
println("max:    $(maximum(lengths))")

# ╔═╡ cb002694-4bdc-4b3b-af8f-ff92d7cc4d9f
println("total docs: $(length(corpus))")

# ╔═╡ d3906fff-e3d4-4d09-9fe8-ec5b37dde56b
println(corpus[6])

# ╔═╡ 391dd814-9e18-4dc6-9771-6f08a98a3027
md"""
Notice that one doc in corpus is quite long.

Next, let's:

- preprocess one doc
- skip short documents
"""

# ╔═╡ 176676cc-f7d5-4ef5-9d71-613d785b1714
for doc in corpus[1:10] # first 10 lines in corpus
	tokens = preprocess(doc, tokenizer; max_length = 128 + 1)
	# at least half length
	length(tokens) < 129 % 2  && continue 

	# and now we want to push these into the vectors
	X, Y = make_pairs(tokens) # creates the specific pairs for next token
	push!(all_X, X) 
	push!(all_Y, Y)
	
end

# ╔═╡ 154bd343-97ea-4a14-8cae-b444eb32597f
println(all_X[1:5])

# ╔═╡ f3bf7dce-52e8-41a8-9c36-2809b301da79
print(all_Y[1:5])

# ╔═╡ 869634ff-8897-455e-99ad-5ad790d1b5de
md"""
Notice:

- some files are actually very short (5276) is always followed by 4868 (not a good rule)
- but ultimately the predictions happen on quite long context windows. This is what makes it easy to predict next tokens, it is as-if we would predict time series.

Solution?:

- Padding to be the exact same lenght
"""

# ╔═╡ b48a6987-7177-4575-9952-4c25469eea1e
seq_length = 128 -1 

# ╔═╡ e8ae3cc9-b748-48ab-a627-ab55ff809f15
pad_idx = tokenizer.lookup["<PAD>"]

# ╔═╡ cd3419c6-6e3a-4e59-b1d9-8553cc966f2b
X_pad = pad_sequences(all_X, seq_length, pad_idx)

# ╔═╡ 7dd468b9-83d1-4f16-a2f7-99911a8eb7af
length(all_X)

# ╔═╡ 6c4d5b45-4109-419f-9e4b-411805269950
md"""
You pad with 1 (which is the index of the unknown) these vectors.
 """

# ╔═╡ 220e21a2-dff0-423f-82ef-bef7bb5f6b22


# ╔═╡ be58cb6b-29d2-40be-a717-7c4e094329e4
md"""
Vectors that are very long and padded learn mostly nothing. We should see what the distribution of lines actually is in my docs
"""

# ╔═╡ 7d19ea62-00ea-4bb3-9d6f-1809d3fb24de
# ╠═╡ disabled = true
#=╠═╡
# distribution of document length
lengths = [length(doc) for doc in corpus]
  ╠═╡ =#

# ╔═╡ 8b602745-7108-4048-b9bf-5d4d0d2e4add


# ╔═╡ 0a9b1fde-63a8-40ef-9a43-777055cefea2
println("mean:   ", mean(lengths))

# ╔═╡ 12e61292-3455-4637-9035-a0e682bfad24
println("median: ", median(lengths))

# ╔═╡ 89aa0a4a-5892-47af-87f0-2c14ab347683
println("max:    ", maximum(lengths))

# ╔═╡ 2a1c149e-142f-4502-9cd6-74766a1dafff
println("95th percentile: ", sort(lengths)[Int(floor(0.95 * length(lengths)))])

# ╔═╡ 5299be81-055f-4667-9d20-cdd818b00a50
md"""
## Embedding

Aim is to see:

- What embedding does
- How are tokens after embedding
"""

# ╔═╡ b20fda08-afa4-459b-a905-452a762cd8a2
vocab_size = length(words)

# ╔═╡ 456bcd5a-50c9-4446-8029-285cb4bc3a76
length(tokenizer.vocabulary)

# ╔═╡ 01f355b0-3ad5-4b31-971f-e308ace731a5
md"""
Notice:

- We have our vocabulary here
"""

# ╔═╡ 9e9c3ed2-14bf-4528-9689-826d3969e735
readlines("/Users/andreibleahu/Documents/BayesGPT-I/src/layers.jl")[135:142]

# ╔═╡ 41c83842-5a94-45c1-bb05-9d364b658dd8
readlines(joinpath(@__DIR__, "src/layers.jl"))[12:16]

# ╔═╡ bdac0e51-3666-49ed-80da-b4dbccccd425
export Embed

# ╔═╡ b59c81d3-a1c2-485d-bb01-1b4ba1ad72df
embed = Embed(16, length(tokenizer.vocabulary))

# ╔═╡ c9f2700c-b6d6-40bb-873f-f22d05bf1e7a
md"""
Let's get a real tokenized input
"""

# ╔═╡ d6dec374-6860-4b3d-813e-5d994a68941a
doc = "the man walked into the room and sat down"

# ╔═╡ 974df40a-380d-4c4a-9455-54c9d8ced720
preprocess(doc, tokenizer)

# ╔═╡ 6d0ae192-31d6-4438-af3a-5f253d4fc04c
preprocess(doc, tokenizer)

# ╔═╡ 7c87a4af-6c76-4238-bcf2-dd89dca2c224
tokens = preprocess(doc, tokenizer)

# ╔═╡ 05457c76-8d34-4573-9539-9a58f3be1299
tokens

# ╔═╡ f9fbe381-c10c-4179-ba50-91d9a7c39df3
md"""
The function of the embedding object proceeses:

- sequence length
- batch size
"""

# ╔═╡ e6c50436-60de-427b-97d1-a223a62e7053
# ╠═╡ disabled = true
#=╠═╡
seq_length = 9 # this is the exact sequence
  ╠═╡ =#

# ╔═╡ 94b07502-9e07-4c53-b95a-c162240e9fbd
B = 1 # one bathc, we process in one go

# ╔═╡ 42cf9647-dae9-463b-90f2-80ac8561dc7f
# form x, which contais both

x = (seq_length, B)

# ╔═╡ a94e7f2a-add5-4744-894a-2da135a5a6a3
embed.weight

# ╔═╡ 250a2291-bca0-4e20-8478-baaf267a77a3
size(embed.weight) 

# ╔═╡ 2c090a3d-cb1b-4bbb-a8f8-617a4eefff4b
md"""
This embedding has: 

- Float32 (we have defined the type)
- 16-long embeddings
- 9974 words(vocabulary) -- each a 16-long embedding
"""

# ╔═╡ a63a0ef0-d587-474f-b45b-7e651cb8d686
# Cell 1 get a real batch
loader = create_batches(corpus, tokenizer; batch_size=4, max_length=32)

# ╔═╡ 31e0403b-2bd1-468c-94a3-4ec3505c18bc
X_batch, Y_batch = first(loader)

# ╔═╡ f3914460-f4f5-4946-a7a4-b322d827511b
md"""
- vec will flatten, hence we get 1 x 31 x 4 = 124
"""

# ╔═╡ cabedd5d-0bbc-4dde-90c2-dec0f1003425
size(X_batch)   # (1, 31, 4)

# ╔═╡ 635cb47d-f036-43d8-a2bc-7db4fceeeadd
embed.weight[:, vec(X_batch)]

# ╔═╡ 49b12520-8174-4dc7-8f39-565f7e1138f6
size(embed.weight[:, vec(X_batch)])

# ╔═╡ c8374ad0-7f43-49de-8919-7282ce80ce9e
md"""
- The total corpus is 9974, but here we process just 124
- In reality we have 4 batches that are padded to size (32) (but recall how we define our structs) (always maximum context shifted by 1)
"""

# ╔═╡ 8305a932-2366-410e-a5f1-404d15795dca
md"""
We want the following: 

- A batch of incoming data :
- Embeddings
- position encoding (this is fixed)
"""

# ╔═╡ beb4a7d9-afa0-4f1c-89d1-40c226d9166a
md"""
first step: we use the dataloder to load both X and Y
"""

# ╔═╡ 6e22f729-3afc-4b86-bc23-36c39539abcd
X_batch

# ╔═╡ 61ae6c83-76d9-40fc-9104-91185239e53d
Y_batch

# ╔═╡ 6d5c296b-8cc9-4688-a8e5-d0a480bb9fbc
md"""
Notice:

- Y_bathc are shifted by 1

Let's examine the data structure
"""

# ╔═╡ 91a65545-f89b-4fbd-b726-5cbe20fa2f71
X_batch[1, :, 1]

# ╔═╡ bb61a819-f2eb-4b67-ae06-409cafb46848
md"""
Note:

- Wasted compute
- mostly padded
"""

# ╔═╡ 761fd7c2-2866-424e-b462-9172dcb8a7d2
X_batch[1, :, 2]

# ╔═╡ 428f92cf-30ac-4ce9-a1d4-1d8f7721359b
X_batch[1, :, 3]

# ╔═╡ 2c2c1599-6e1f-4027-aaee-268e70adee5c
X_batch[1, :, 4]

# ╔═╡ 39af41b5-e593-43c8-a175-e5ed67d8f987
md""""
conclusion: 

- We need to set realistic hard limits based on our data distribution
- for a length of 128, docs that are 1 token are very short
"""

# ╔═╡ b1cc553d-2604-48b3-8a25-f2e03984bae1
md"""
Let's create the position encoding
"""

# ╔═╡ b54144d0-e3b6-4a41-8a79-61d6483400a5
pe = PositionEncoding(16, 31) 

# ╔═╡ fd086d2d-c067-4955-ad54-c20a85c5642b
size(pe.weight)

# ╔═╡ 4f62a98e-247c-41ec-bd3c-a8861fc7ee5a


# ╔═╡ c4c89250-0714-4e90-b3f5-2bc56efc3b98
md"""
Note:

- This is very natural, because an object is a weight, beloging to AbstractArray, that takes two arguments
- The next step is to make it sinusoidal, that is, apply sin to it
- This encodes the tokens as varying in the text
"""

# ╔═╡ 144c91eb-a2da-4480-bcac-41b6f3c6eaf8
X = dropdims(X_batch, dims=1) # omly the 4 vectors


# ╔═╡ cdfca536-961f-4ebf-a1cb-a96fb031b005
X_embed = embed(X)

# ╔═╡ 496ace7a-baae-4f53-b27d-2e026471e62d
md"""
Note:

- Each vector actually becomes an embedding
- So you have an object that transforms the vectors into embedding, that is, a 16 (embedding dimension) times 31 matrix
- Notice how for this vector, only the first column matters: this is because everything else was padded to 1
"""

# ╔═╡ a5e20aab-fcef-4210-8bcb-9f1f3dcd3f15
X_embed[:, :, 1]

# ╔═╡ 6e146a07-80e2-4488-86ad-6e8f3b9a0d13
X_pos = X_embed .+ pe.weight

# ╔═╡ 2add7d9f-3fc8-410b-8b19-4d700961da0d
md"""
- PE embedding means that we have added sine to the vectors
- So values sit between -1 and 1 (but they are added to existing embeddings, which is why they are not 0 or 1)
- We alternate sine with cosine for positions
"""

# ╔═╡ 21874a7a-6bde-4317-9326-93703d4143fb
pe.weight

# ╔═╡ de033738-a08a-4888-9896-8cf7b862bf6b
md"""

## Multi-head attention
"""

# ╔═╡ 0fa864e9-86fa-4d36-b8bb-a9e4ebd240a2
size(X_pos)

# ╔═╡ aaa74f93-cb0b-4e08-acc2-0e87dcf831ab
md"""
Let's define a multihead attention 
"""

# ╔═╡ 35bae422-5780-43bd-855c-5cc25e7b4c25
mha = CausalMultiheadAttention(2, 16, 16) 

# ╔═╡ c6210931-8e22-46d1-9e6a-59340a36dbcf
Q = mha.denseQ(X_pos)

# ╔═╡ ad02db61-6486-415c-b7e5-db0c7da39d0b
K = mha.denseK(X_pos)

# ╔═╡ 9345df13-d04c-4b80-86df-805deebf5708
V = mha.denseV(X_pos)

# ╔═╡ 1b06a529-dffb-419a-b4c5-d01229b401fb
out = mha(Q, K, V)

# ╔═╡ 631eb9dc-917c-4537-be45-62293da71b89
# 2 heads
dh = 8

# ╔═╡ 05ac1794-a43f-40d8-ad2a-6c4d3b260625
to_heads(X) = permutedims(reshape(X, dh, 2, size(X,2), size(X,3)), [1,3,2,4])


# ╔═╡ d6635206-dc67-4194-a854-79e93195fa0e
Qh = to_heads(Q)

# ╔═╡ 9c89cf9c-e131-44aa-b475-908bd5d2f09d
exp(-Inf)

# ╔═╡ 4dce53d9-f160-4b92-9d90-fc3f038b9b81
ϵ =1

# ╔═╡ b84d6627-dd36-4d34-9335-7428db62da1c
θ = 0.0

# ╔═╡ Cell order:
# ╟─a0bb92ec-505e-11f1-3704-8b56700c9b37
# ╠═28db7890-7d6d-4085-b74c-a85939441ba3
# ╠═cab68644-80d1-48c4-9bd0-7cd2210679a3
# ╠═4e314c12-3d11-4eb6-b667-37dd9ec4a6f0
# ╟─2fc40ef1-0ca4-4748-95ee-4a248473551b
# ╠═9c165c1d-a8f6-4f6c-be8e-951760508edd
# ╟─5de57b9b-d1f5-4470-a80a-d6ca9e741058
# ╠═3aabaf33-ebaa-41b7-991d-b4c184279e20
# ╟─8950fecd-7034-4bd2-90ac-8feb9f76d635
# ╠═bc163526-d1d5-41e4-9426-5bdeda318215
# ╠═f77d1b1a-8962-46c3-843a-e215771a4d2e
# ╠═39b63a61-61ad-4c57-98c3-993e504e358a
# ╠═ea20020c-6917-496b-ad0e-af94b2c1627b
# ╠═c7dda7f8-bb2e-4396-82d1-ad26085af609
# ╠═95fb4cd5-57cc-4523-8cd4-cf23b132aee6
# ╠═98b36bee-192e-4360-9e48-f9cd938044e9
# ╟─998f204e-fbf3-4c00-9a61-a893afa613af
# ╟─21552fdb-ec7e-43ce-b2c2-e229b4df12f4
# ╟─a47fc023-0eae-4045-98a6-9dd4d5283425
# ╟─e971fcd8-8440-4a62-abe9-f606964e1ae6
# ╟─dd0e3d73-e972-4943-825e-5684a19ebf7f
# ╠═ed470a47-9bca-4d32-8018-60648914aaa2
# ╠═562bb060-881d-44d2-88c7-76f75535311c
# ╠═a10ec7d7-44d5-44db-8553-a23f7bb431d4
# ╠═857d0d6c-46a6-44db-8685-b3aa53a121d4
# ╠═cf8a8b73-10dc-44d1-b0a2-6df5917d4d74
# ╟─5e9bcaf4-14fb-4a19-b246-251494a9c0e7
# ╟─f0824b60-04b9-45d0-9355-2140df4c6a4e
# ╠═1dbef844-fdad-4492-b7d2-4349e1a380c4
# ╠═f5ee80c0-6f5a-4b04-8d22-70048eaafa95
# ╠═2d561449-c043-4333-8cb9-14a7c5c04988
# ╠═2b7e2ff9-0e91-47bf-9d5d-05b4cdf62d83
# ╠═2a761ed3-575a-48c5-8a75-19c84b164149
# ╠═07f07562-8870-427a-8a5a-13e59cd79fa0
# ╠═283050e3-45ce-4d7b-81c9-501b43067b0e
# ╠═8340a775-f5a7-48f4-be2c-fe3d5cd09920
# ╠═7f226751-ed4a-4aea-a68e-95f565ae552d
# ╠═728fd22e-aefc-4c3c-a319-0dcafbb3f4ef
# ╠═d323dfa2-163f-4656-b53c-920639f354d0
# ╠═549a5c3b-8a5e-4ddc-ae20-800013c2370a
# ╟─cd99e93a-320b-4728-a8cb-fa260ee840fc
# ╟─3130fe4b-741c-41a1-ab5d-7789a992e72a
# ╠═c33f9c8e-9762-403b-a1c9-e3af8f341768
# ╠═e51cf592-788e-4b3f-8408-cc1c8f177c96
# ╟─3594fbf3-d9ac-410e-977e-6a8016e122d4
# ╠═3b813e4d-0418-4f07-9bbd-d13793fe832e
# ╠═3dbf0138-5600-421b-8783-0dd82193a8b4
# ╠═f4a85f67-1204-4e57-a8ef-3bae5e796e1a
# ╠═e55bb10c-2ca9-4112-b5a3-433d6d12bfa3
# ╠═906eb80a-d0fe-45a0-9239-64afaffcbe46
# ╠═08251321-0d8f-4659-b152-3ecbaeda99b9
# ╠═6e2b9b9e-459d-4135-a9d2-972481669031
# ╟─463958e0-1e82-4292-8703-9187db183a1b
# ╟─ad797ef1-43c1-4ba8-baea-f4313ac0a880
# ╟─27f0617b-bf40-467f-83bb-7f6b3ae243da
# ╠═d2374e61-23e5-456f-9766-1f9f49e4e3bd
# ╠═156d2301-df64-4371-be1b-7ab4b4f263e4
# ╠═992933d9-f3f0-4ae5-858a-354036812a82
# ╠═8c6c5f89-5ede-412f-9e39-295b85eacb3a
# ╟─2336969f-373d-43f8-af42-38da967cf201
# ╠═298ccf31-9c8c-4006-b918-a1510a1f52d7
# ╠═d03aae98-3c91-486f-a79b-40717516503d
# ╟─03513e58-2cec-4ec0-b8a2-a1f43fc0cd45
# ╟─6cb4b1b0-cf5d-4d50-acc1-42e5d431cc6a
# ╠═6d6e2afe-68ad-4612-b194-6c77e84ad3fe
# ╠═4d9cab90-23f9-422c-95e7-195899669ece
# ╠═9bd3a45e-8931-49ce-b5b8-5afc3aa0fb9c
# ╠═eceafa6f-bfd8-4af8-9e17-cda738d28286
# ╠═790eaedc-7469-4710-ac15-3a2e72650460
# ╟─1afe2ee3-b0a1-4c37-adcc-c5e59550c5b6
# ╠═d224feaa-921f-4dfa-8b32-21864374b725
# ╠═d59f83d6-f1f8-4af1-bc1f-9909c471d750
# ╠═24b1d515-9459-483a-86da-50629bb43718
# ╠═6598ae53-b6f2-4740-80e5-1d7ed945f75a
# ╠═96e6314d-1b2a-48ba-90e3-f72791ef6799
# ╟─a5317aae-f953-40d6-ab03-def6f2581189
# ╟─6398b420-4bbe-4595-8ebd-b751eace4e4e
# ╟─2a59ee46-5c99-4fc9-9a9c-471db4aced36
# ╠═4203b46e-fa8d-4229-a9b0-9cb7dd7c2f8f
# ╟─b8c8f36f-3a91-4382-8ad5-85d3bd5e5249
# ╠═a0e295b0-d6fa-46c8-8068-4f24a5b9a3a0
# ╟─ffd47e13-7b7f-4a87-b242-378a873a2366
# ╠═f61415cb-8ec2-4728-9a5f-55a560d9b40e
# ╟─f346ff56-e442-40e5-a580-cc3bcaee9a32
# ╠═b9f51f4a-59d1-49bd-82db-c4994465fcb1
# ╠═15b404d1-9e45-41d4-bf45-40316fa58392
# ╠═4de8f4aa-2ef8-408a-8119-3e16322f6c39
# ╠═f4772a57-3db2-44df-b531-e0e47d110bc1
# ╠═548bacd4-5fa5-423f-abd7-3fac78c56a4e
# ╟─916855d4-3a4d-414a-9c1a-f2e29f6be03e
# ╠═974df40a-380d-4c4a-9455-54c9d8ced720
# ╠═c138f8da-b282-4707-96cf-1f2534ae8904
# ╠═cb217241-cfec-4634-8bb2-5ccf1c17d762
# ╠═783932db-bbc2-4ba1-81ca-109430d5bc28
# ╠═38a7bf73-2295-422c-9eeb-7b40396c2bd0
# ╠═e8e09494-55a2-437e-a598-a662ab777e56
# ╠═3df679bb-a566-49b0-804c-3cabbfdae9bd
# ╠═6eeebf2c-9424-4f53-b9a2-83ab0c3dd9f5
# ╠═6d0ae192-31d6-4438-af3a-5f253d4fc04c
# ╟─ffbfa835-e912-4011-b8fc-bfad82447e4d
# ╠═c21e38b7-2645-470c-8664-437d6545a2eb
# ╠═1d3bd1c4-f49e-4d30-ad75-58b062e8ca4d
# ╠═2713f1cd-4d91-45e6-a32c-334a1ecb542e
# ╠═182eae98-cd26-4788-a676-4fefbfc20131
# ╠═79a3cf96-983d-4b6f-a500-9e4060241f1c
# ╠═d854c5b1-9104-48d1-93c9-90cbd4758b8d
# ╟─06652db7-e2b1-413e-93f5-3f825f7c3cff
# ╠═46d520fa-e2cd-4c64-9e85-52b7f2cde687
# ╠═8f7f3ea5-b42b-42a6-96c8-73da38fd3e47
# ╠═3e25408c-a09c-4b1c-a83f-fd81cc65ba18
# ╠═3adfa881-f904-440a-9a9a-b2c2fa1dff4c
# ╠═2da79be5-36c8-4066-9dde-1721bf0e7857
# ╠═58bdb6a6-c04b-4b73-a23c-71c95ae69f87
# ╠═efbcdf50-c46f-4642-80fd-17a5252cf20f
# ╠═13b8a87c-4a25-4987-8a51-e5ac1f3cd122
# ╠═56704f9b-cb72-45bd-b6c4-286fc2928563
# ╠═b899fc16-a7ff-4acb-9690-67727cac32ff
# ╠═d5750bc2-6d2a-417a-b64c-133da58571af
# ╠═dae95712-a107-439e-a590-332f9b9cfb5f
# ╠═7806898f-0abf-427e-b9b0-e59c94369fbd
# ╟─08e6dab3-54e4-422d-a559-c4c309571ec9
# ╠═f28d0ea8-21cc-4ed1-b4dd-a7e490048204
# ╠═d5244840-8473-4b81-89f2-40a72428f849
# ╠═932ef313-e007-4253-ae20-3f67e21cd71f
# ╠═f932ad27-6bfe-407e-8eaa-e622b2ffa167
# ╠═50cae8f3-da93-4337-857f-14fe39e82d4b
# ╠═c186a039-d9dd-4f83-86ec-4a6b47d3a46e
# ╠═d251e3f3-6c6b-4e58-8ff4-949cd186ea57
# ╠═353069da-dad5-4f44-aeff-91f11299ece2
# ╠═cb002694-4bdc-4b3b-af8f-ff92d7cc4d9f
# ╠═d3906fff-e3d4-4d09-9fe8-ec5b37dde56b
# ╠═391dd814-9e18-4dc6-9771-6f08a98a3027
# ╠═176676cc-f7d5-4ef5-9d71-613d785b1714
# ╠═154bd343-97ea-4a14-8cae-b444eb32597f
# ╠═f3bf7dce-52e8-41a8-9c36-2809b301da79
# ╟─869634ff-8897-455e-99ad-5ad790d1b5de
# ╠═b48a6987-7177-4575-9952-4c25469eea1e
# ╠═e8ae3cc9-b748-48ab-a627-ab55ff809f15
# ╠═cd3419c6-6e3a-4e59-b1d9-8553cc966f2b
# ╠═7dd468b9-83d1-4f16-a2f7-99911a8eb7af
# ╠═6c4d5b45-4109-419f-9e4b-411805269950
# ╟─220e21a2-dff0-423f-82ef-bef7bb5f6b22
# ╠═05457c76-8d34-4573-9539-9a58f3be1299
# ╠═95a43dc9-5566-40f5-8496-21af172acb1a
# ╟─be58cb6b-29d2-40be-a717-7c4e094329e4
# ╠═7d19ea62-00ea-4bb3-9d6f-1809d3fb24de
# ╠═8b602745-7108-4048-b9bf-5d4d0d2e4add
# ╠═0a9b1fde-63a8-40ef-9a43-777055cefea2
# ╠═12e61292-3455-4637-9035-a0e682bfad24
# ╠═89aa0a4a-5892-47af-87f0-2c14ab347683
# ╠═2a1c149e-142f-4502-9cd6-74766a1dafff
# ╟─5299be81-055f-4667-9d20-cdd818b00a50
# ╠═b20fda08-afa4-459b-a905-452a762cd8a2
# ╠═456bcd5a-50c9-4446-8029-285cb4bc3a76
# ╟─01f355b0-3ad5-4b31-971f-e308ace731a5
# ╠═9e9c3ed2-14bf-4528-9689-826d3969e735
# ╠═41c83842-5a94-45c1-bb05-9d364b658dd8
# ╠═bdac0e51-3666-49ed-80da-b4dbccccd425
# ╠═b59c81d3-a1c2-485d-bb01-1b4ba1ad72df
# ╠═c9f2700c-b6d6-40bb-873f-f22d05bf1e7a
# ╠═d6dec374-6860-4b3d-813e-5d994a68941a
# ╠═7c87a4af-6c76-4238-bcf2-dd89dca2c224
# ╟─f9fbe381-c10c-4179-ba50-91d9a7c39df3
# ╠═e6c50436-60de-427b-97d1-a223a62e7053
# ╠═94b07502-9e07-4c53-b95a-c162240e9fbd
# ╠═42cf9647-dae9-463b-90f2-80ac8561dc7f
# ╠═a94e7f2a-add5-4744-894a-2da135a5a6a3
# ╠═250a2291-bca0-4e20-8478-baaf267a77a3
# ╟─2c090a3d-cb1b-4bbb-a8f8-617a4eefff4b
# ╠═115336ef-45bd-4084-a8ee-d5c02b8e5101
# ╠═a63a0ef0-d587-474f-b45b-7e651cb8d686
# ╠═31e0403b-2bd1-468c-94a3-4ec3505c18bc
# ╟─f3914460-f4f5-4946-a7a4-b322d827511b
# ╠═cabedd5d-0bbc-4dde-90c2-dec0f1003425
# ╠═635cb47d-f036-43d8-a2bc-7db4fceeeadd
# ╠═49b12520-8174-4dc7-8f39-565f7e1138f6
# ╟─c8374ad0-7f43-49de-8919-7282ce80ce9e
# ╠═8305a932-2366-410e-a5f1-404d15795dca
# ╟─beb4a7d9-afa0-4f1c-89d1-40c226d9166a
# ╠═6e22f729-3afc-4b86-bc23-36c39539abcd
# ╠═61ae6c83-76d9-40fc-9104-91185239e53d
# ╟─6d5c296b-8cc9-4688-a8e5-d0a480bb9fbc
# ╠═91a65545-f89b-4fbd-b726-5cbe20fa2f71
# ╟─bb61a819-f2eb-4b67-ae06-409cafb46848
# ╠═761fd7c2-2866-424e-b462-9172dcb8a7d2
# ╠═428f92cf-30ac-4ce9-a1d4-1d8f7721359b
# ╠═2c2c1599-6e1f-4027-aaee-268e70adee5c
# ╟─39af41b5-e593-43c8-a175-e5ed67d8f987
# ╟─b1cc553d-2604-48b3-8a25-f2e03984bae1
# ╠═b54144d0-e3b6-4a41-8a79-61d6483400a5
# ╠═fd086d2d-c067-4955-ad54-c20a85c5642b
# ╠═4f62a98e-247c-41ec-bd3c-a8861fc7ee5a
# ╠═c4c89250-0714-4e90-b3f5-2bc56efc3b98
# ╠═144c91eb-a2da-4480-bcac-41b6f3c6eaf8
# ╠═cdfca536-961f-4ebf-a1cb-a96fb031b005
# ╠═496ace7a-baae-4f53-b27d-2e026471e62d
# ╠═a5e20aab-fcef-4210-8bcb-9f1f3dcd3f15
# ╠═6e146a07-80e2-4488-86ad-6e8f3b9a0d13
# ╟─2add7d9f-3fc8-410b-8b19-4d700961da0d
# ╠═21874a7a-6bde-4317-9326-93703d4143fb
# ╟─de033738-a08a-4888-9896-8cf7b862bf6b
# ╠═0fa864e9-86fa-4d36-b8bb-a9e4ebd240a2
# ╟─aaa74f93-cb0b-4e08-acc2-0e87dcf831ab
# ╠═35bae422-5780-43bd-855c-5cc25e7b4c25
# ╠═c6210931-8e22-46d1-9e6a-59340a36dbcf
# ╠═ad02db61-6486-415c-b7e5-db0c7da39d0b
# ╠═9345df13-d04c-4b80-86df-805deebf5708
# ╠═1b06a529-dffb-419a-b4c5-d01229b401fb
# ╠═631eb9dc-917c-4537-be45-62293da71b89
# ╠═05ac1794-a43f-40d8-ad2a-6c4d3b260625
# ╠═d6635206-dc67-4194-a854-79e93195fa0e
# ╠═9c89cf9c-e131-44aa-b475-908bd5d2f09d
# ╠═4dce53d9-f160-4b92-9d90-fc3f038b9b81
# ╠═b84d6627-dd36-4d34-9335-7428db62da1c
