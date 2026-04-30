# ============================================================
# generate.jl
# Autoregressive text generation for the decoder model.
#
# Three generation modes:
#   1. Standard sampling     → temperature only
#   2. Top-k sampling        → fixes repetition loops
#   3. Bayesian generation   → N forward passes, mean/variance
# ============================================================

using Flux: softmax
using Statistics: mean, var, std
using Printf


# ────────────────────────────────────────────────────────────
# 1. Standard temperature sampling
# ────────────────────────────────────────────────────────────

function sample_token(logits::AbstractVector, temperature::Float32)
    scaled = logits ./ temperature
    probs  = softmax(scaled)
    r = rand(Float32)
    cumsum = 0f0
    for (i, p) in enumerate(probs)
        cumsum += p
        if r < cumsum
            return i
        end
    end
    return length(probs)
end


# ────────────────────────────────────────────────────────────
# 2. Top-k sampling
# ────────────────────────────────────────────────────────────

function sample_token_topk(logits::AbstractVector, temperature::Float32; k::Int = 10)
    scaled    = logits ./ temperature
    sorted    = sort(scaled, rev = true)
    threshold = sorted[min(k, length(sorted))]
    masked    = [l >= threshold ? l : typemin(Float32) for l in scaled]
    probs     = softmax(masked)
    r = rand(Float32)
    cumsum = 0f0
    for (i, p) in enumerate(probs)
        cumsum += p
        if r < cumsum
            return i
        end
    end
    return length(probs)
end


# ────────────────────────────────────────────────────────────
# Helper: tokenize seed
# ────────────────────────────────────────────────────────────

function _tokenize_seed(tokenizer, seed::AbstractString)
    words     = [m.match for m in eachmatch(r"\w\w+\b", simplify(seed))]
    token_ids = tokenizer(words)
    if isempty(token_ids)
        @warn "Seed produced no known tokens, using <UNK>"
        return [tokenizer.unkidx]
    end
    token_ids
end

# ────────────────────────────────────────────────────────────
# 3. Top-k sampling with repetition penalty
# ────────────────────────────────────────────────────────────

"""
    sample_token_topk_norep(logits, temperature, generated; k, penalty)

Top-k sampling with repetition penalty.
Recently generated tokens have their logits divided by `penalty`,
making repetition less likely without changing the model weights.

- `penalty = 1.0`  → no penalty (standard top-k)
- `penalty = 1.5`  → moderate penalty
- `penalty = 2.0`  → strong penalty
"""
function sample_token_topk_norep(
        logits    :: AbstractVector,
        temperature :: Float32,
        generated :: Vector{Int};
        k         :: Int     = 10,
        penalty   :: Float32 = 1.5f0)

    scaled = copy(logits) ./ temperature

    # penalise last 5 tokens
    recent = generated[max(1, end-5):end]
    for id in recent
        scaled[id] /= penalty
    end

    # top-k mask
    sorted    = sort(scaled, rev = true)
    threshold = sorted[min(k, length(sorted))]
    masked    = [l >= threshold ? l : typemin(Float32) for l in scaled]

    probs = softmax(masked)
    r = rand(Float32)
    cumsum = 0f0
    for (i, p) in enumerate(probs)
        cumsum += p
        if r < cumsum
            return i
        end
    end
    return length(probs)
end

# ────────────────────────────────────────────────────────────
# 3. Standard generation
# ────────────────────────────────────────────────────────────

function generate(
        model,
        tokenizer,
        seed        :: AbstractString;
        max_length  :: Int     = 50,
        temperature :: Float32 = 0.8f0,
        end_token   :: String  = "<PAD>")

    generated = _tokenize_seed(tokenizer, seed)
    end_id    = get(tokenizer.lookup, end_token, tokenizer.unkidx)

    for _ in 1:max_length
        x           = reshape(generated, 1, length(generated), 1)
        logits_all  = model(x)
        next_logits = logits_all[:, end, 1]
        next_id     = sample_token(next_logits, temperature)
        next_id == end_id && break
        push!(generated, next_id)
    end

    join([tokenizer.vocabulary[id] for id in generated], " ")
end


# ────────────────────────────────────────────────────────────
# 4. Top-k generation
# ────────────────────────────────────────────────────────────
function generate_topk(
        model,
        tokenizer,
        seed        :: AbstractString;
        max_length  :: Int     = 50,
        temperature :: Float32 = 0.8f0,
        k           :: Int     = 10,
        penalty     :: Float32 = 1.5f0,
        end_token   :: String  = "<PAD>")

    generated = _tokenize_seed(tokenizer, seed)
    end_id    = get(tokenizer.lookup, end_token, tokenizer.unkidx)

    for _ in 1:max_length
        x           = reshape(generated, 1, length(generated), 1)
        logits_all  = model(x)
        next_logits = logits_all[:, end, 1]
        next_id     = sample_token_topk_norep(next_logits, temperature, generated;
                                              k, penalty)
        next_id == end_id && break
        push!(generated, next_id)
    end

    join([tokenizer.vocabulary[id] for id in generated], " ")
end


# ────────────────────────────────────────────────────────────
# 5. Bayesian generation
# ────────────────────────────────────────────────────────────

function bayesian_generate(
        model,
        tokenizer,
        seed        :: AbstractString;
        n_forward   :: Int     = 20,
        max_length  :: Int     = 50,
        temperature :: Float32 = 0.8f0,
        k           :: Int     = 10,
        penalty     :: Float32 = 1.5f0,    # 
        end_token   :: String  = "<PAD>")

    generated      = _tokenize_seed(tokenizer, seed)
    end_id         = get(tokenizer.lookup, end_token, tokenizer.unkidx)
    uncertainties  = Float32[]
    conf_intervals = Float32[]
    chosen_words   = String[]

    for _ in 1:max_length
        x = reshape(generated, 1, length(generated), 1)

        # N forward passes — each samples different weights ε ~ N(0,I)
        logits_samples = hcat([model(x)[:, end, 1] for _ in 1:n_forward]...)
        # shape: (vocab_size, n_forward)

        mean_logits = vec(mean(logits_samples; dims = 2))
        std_logits  = vec(std(logits_samples;  dims = 2))

        # uncertainty = mean std across vocab at this position
        token_uncertainty = mean(std_logits)
        push!(uncertainties,  token_uncertainty)
        push!(conf_intervals, 2f0 * 1.96f0 * token_uncertainty)

        next_id = sample_token_topk_norep(mean_logits, temperature, generated;
                                   k, penalty)
        next_id == end_id && break

        push!(generated,    next_id)
        push!(chosen_words, tokenizer.vocabulary[next_id])
    end

    text = join([tokenizer.vocabulary[id] for id in generated], " ")
    return (
        text           = text,
        uncertainties  = uncertainties,
        conf_intervals = conf_intervals,
        new_words      = chosen_words,
    )
end


# ────────────────────────────────────────────────────────────
# 6. Display functions
# ────────────────────────────────────────────────────────────

function generate_samples(
        model,
        tokenizer,
        seed        :: AbstractString;
        n_samples   :: Int     = 5,
        max_length  :: Int     = 50,
        temperature :: Float32 = 0.8f0)

    println("\nSeed: \"$seed\"\n" * "─"^50)
    for i in 1:n_samples
        result = generate(model, tokenizer, seed; max_length, temperature)
        println("Sample $i: $result")
    end
end

function generate_samples_topk(
        model,
        tokenizer,
        seed        :: AbstractString;
        n_samples   :: Int     = 5,
        max_length  :: Int     = 50,
        temperature :: Float32 = 0.8f0,
        k           :: Int     = 10,
        penalty     :: Float32 = 1.5f0)

    println("\n[Top-k=$k, penalty=$penalty] Seed: \"$seed\"\n" * "─"^50)
    for i in 1:n_samples
        result = generate_topk(model, tokenizer, seed; max_length, temperature, k, penalty)
        println("Sample $i: $result")
    end
end


function bayesian_generate_report(
        model,
        tokenizer,
        seed        :: AbstractString;
        n_forward   :: Int     = 20,
        max_length  :: Int     = 40,
        temperature :: Float32 = 0.8f0,
        k           :: Int     = 10,
        penalty     :: Float32 = 1.5f0)    # ← add this

    println("\n[Bayesian n_forward=$n_forward, penalty=$penalty] Seed: \"$seed\"\n" * "─"^50)

    result = bayesian_generate(model, tokenizer, seed;
                               n_forward, max_length, temperature, k, penalty)  # ← pass through

                               
    println("Generated : $(result.text)\n")
    println("Token uncertainty (σ) and 95% CI:")
    println("─"^50)

    for (word, σ, ci) in zip(result.new_words, result.uncertainties, result.conf_intervals)
        bar        = "█" ^ min(Int(round(σ * 40)), 30)
        confidence = σ < 0.1f0 ? "confident"      :
                     σ < 0.3f0 ? "moderate"        :
                     σ < 0.6f0 ? "uncertain"       : "very uncertain"
        @printf("  %-15s σ=%.3f  CI=±%.3f  %s %s\n",
                word, σ, ci/2f0, bar, confidence)
    end

    isempty(result.uncertainties) && return

    println()
    @printf("  Mean uncertainty : %.3f\n", mean(result.uncertainties))
    @printf("  Max uncertainty  : %.3f\n", maximum(result.uncertainties))
    @printf("  Min uncertainty  : %.3f\n", minimum(result.uncertainties))
end