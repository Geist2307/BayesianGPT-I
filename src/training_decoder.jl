# ============================================================
# training_decoder.jl
# Training infrastructure for next-token prediction.
#
# Differences from training.jl:
#   - loss computed at every sequence position, not per document
#   - Y is a token id sequence, not a one-hot label
#   - accuracy = fraction of correctly predicted next tokens
#
# KL divergence and sparsity functions are identical to training.jl
# ============================================================

using Flux: logitcrossentropy, onecold, Adam
using Statistics: mean


# ------------------
# KL divergence 
# ------------------

# Constants from Molchanov et al. (2017)
const _K1 = 0.63576f0
const _K2 = 1.87320f0
const _K3 = 1.48695f0

function kl(layer::VariationalDropoutMolchanov)
    @inbounds begin
        logα = @. layer.logσ2 - 2f0 * log(abs(layer.θ) + 1f-8)
        return sum(@. -(_K1 - _K1 * sigmoid(_K2 + _K3 * logα) - 0.5f0 * log1p(1f0 / exp(logα))))
    end
end

function get_variational_layers(model)
    vcat([
        if layer isa TransformerDecoderBlock
            [layer.dense1, layer.dense2]
        elseif layer isa VariationalDropoutMolchanov
            [layer]
        else
            VariationalDropoutMolchanov[]
        end
        for layer in model.layers
    ]...)
end


# ----------------
# Sparsity  
# ---------------

const SPARSITY_THRESHOLD_α = 3.0f0 # threshold from Molchanov et al. (2017)

@inline function layer_sparsity(
        layer::VariationalDropoutMolchanov;
        custom_threshold::Float32 = SPARSITY_THRESHOLD_α)
    logα = @. layer.logσ2 - 2f0 * log(abs(layer.θ) + 1f-8)
    Float32(count(x -> x ≥ custom_threshold, logα) / length(logα))
end

function get_mlp_sparsities(model; threshold::Float32 = SPARSITY_THRESHOLD_α)
    sparsities = Float32[]
    for layer in model.layers
        if layer isa TransformerDecoderBlock
            push!(sparsities, layer_sparsity(layer.dense1; custom_threshold = threshold))
            push!(sparsities, layer_sparsity(layer.dense2; custom_threshold = threshold))
        end
    end
    sparsities
end


# ----------------------------
# Next-token prediction loss
# ─--------------------------

"""
decoder_energy_loss(model, X, Y, N; kl_scale) → scalar

Variational free energy for next-token prediction:

    L = N · (1/N·T · Σ_t Σ_b NLL(ŷ[:,t,b], Y[t,b]))  +  β · KL/N

where:
- T = sequence length
- N = total number of tokens in the dataset
- ŷ : (vocab_size, T, B)  model output logits
- Y : (1, T, B)           target token ids

Cross-entropy is computed at every position t independently,
then averaged over positions and batch.
"""
function decoder_energy_loss(
        model, X, Y, N::Int;
        kl_scale :: Float32 = 1.0f0)

    # forward pass---------------------
    # X : (1, T, B) → model → (vocab_size, T, B)
    ŷ = hcat([model(X[:, :, i]) for i in axes(X, 3)]...)  

    vocab_size = size(ŷ, 1)
    T          = size(X, 2)
    B          = size(X, 3)

    # reshape for cross entropy -------------------------
    # ŷ : (vocab_size, T*B)
    # Y : (T*B,)  as integer class indices
    ŷ_flat = reshape(ŷ, vocab_size, T * B)
    Y_flat = vec(reshape(Y, T * B))             # integer token ids

    # one-hot encode targets  -> (vocab_size, T*B)
    Y_onehot = Flux.onehotbatch(Y_flat, 1:vocab_size)

    nll = N * logitcrossentropy(ŷ_flat, Y_onehot)

    #  KL term --------------------------------------
    kl_sum = sum(kl(l) for l in get_variational_layers(model))

    nll + kl_scale * (kl_sum / N)
end


# -----------------------------
# Accuracy
# --------------------------------

"""
    token_accuracy(model, loader) → Float32

Fraction of next tokens correctly predicted across the dataset.
"""
function token_accuracy(model, loader)
    correct = 0
    total   = 0

    for (X, Y) in loader
        ŷ = hcat([model(X[:, :, i]) for i in axes(X, 3)]...)
        # predicted token at each position = argmax over vocab
        pred = vec(mapslices(argmax, ŷ; dims = 1))  # (T*B,)
        true_ids = vec(Y)                            # (T*B,)

        correct += sum(pred .== true_ids)
        total   += length(true_ids)
    end

    Float32(correct / total)
end


# ------------------------------------
# Evaluation
# -----------------------------------

"""
evaluate_decoder(model, loader; N_total, kl_scale) → NamedTuple

Compute loss, token accuracy, and MLP sparsities over a DataLoader.
"""
function evaluate_decoder(model, loader; N_total::Int, kl_scale::Float32 = 1.0f0)
    total_loss = 0.0f0
    correct    = 0
    total      = 0
    sparsities = NamedTuple[]

    for (X, Y) in loader
        total_loss += decoder_energy_loss(model, X, Y, N_total; kl_scale)

        ŷ = hcat([model(X[:, :, i]) for i in axes(X, 3)]...)
        pred     = vec(mapslices(argmax, ŷ; dims = 1))
        true_ids = vec(Y)
        correct += sum(pred .== true_ids)
        total   += length(true_ids)

        for layer in model.layers
            if layer isa TransformerDecoderBlock
                push!(sparsities, (
                    dense1 = layer_sparsity(layer.dense1),
                    dense2 = layer_sparsity(layer.dense2),
                ))
            end
        end
    end

    (
        loss            = total_loss,
        accuracy        = Float32(correct / total),
        sparsity_dense1 = isempty(sparsities) ? 0f0 : last(sparsities).dense1,
        sparsity_dense2 = isempty(sparsities) ? 0f0 : last(sparsities).dense2,
    )
end


# ---------------------------------------------
# KL warmup schedule  (identical to training.jl)
# ─--------------------------------------------

"""
We use 30 epochs as warmup for the linear schedule. The warmup
period is a hyperparam that ensures the KL divergence term does 
not dominate during the first epochs

"""

kl_schedule(epoch::Int) = min(1.0f0, epoch / 30f0)


# ---------------------------------
# Training loop
# ----------------------------------------

"""
    train_decoder!(model, train_loader, val_loader; epochs, kl_schedule, lr)

Full training loop for next-token prediction with:
- Adam optimiser
- KL warm-up via `kl_schedule`
- Per-epoch logging of loss, token accuracy, and MLP sparsity

Returns a NamedTuple of training history vectors.
"""
function train_decoder!(
        model,
        train_loader,
        val_loader;
        epochs      :: Int      = 10,
        kl_schedule :: Function = kl_schedule,
        lr          :: Float64  = 1e-3)

    history = (
        train_losses      = Float32[],
        val_losses        = Float32[],
        train_accuracies  = Float32[],
        val_accuracies    = Float32[],
        sparsities_dense1 = Float32[],
        sparsities_dense2 = Float32[],
    )

    opt_state = Flux.setup(Adam(lr), model)

    # total number of tokens in training set
    N_total = sum(size(X, 2) * size(X, 3) for (X, _) in train_loader)

    for epoch in 1:epochs
        β = kl_schedule(epoch)

        Flux.train!(model, train_loader, opt_state) do m, X, Y
            decoder_energy_loss(m, X, Y, N_total; kl_scale = Float32(β))
        end

        N_train = sum(size(X, 2) * size(X, 3) for (X, _) in train_loader)
        N_val   = sum(size(X, 2) * size(X, 3) for (X, _) in val_loader)

        tr = evaluate_decoder(model, train_loader; N_total = N_train, kl_scale = Float32(β))
        vl = evaluate_decoder(model, val_loader;   N_total = N_val,   kl_scale = Float32(β))

        println("Epoch $epoch / $epochs | β=$(round(β, digits=3)) | " *
            "train_loss=$(round(tr.loss, digits=4)) | " *
            "train_acc=$(round(tr.accuracy, digits=3)) | " *
            "val_acc=$(round(vl.accuracy, digits=3)) | " *
            "sparsity_dense1=$(round(tr.sparsity_dense1, digits=3)) | " *
            "sparsity_dense2=$(round(tr.sparsity_dense2, digits=3))")

        push!(history.train_losses,      tr.loss)
        push!(history.val_losses,        vl.loss)
        push!(history.train_accuracies,  tr.accuracy)
        push!(history.val_accuracies,    vl.accuracy)
        push!(history.sparsities_dense1, tr.sparsity_dense1)
        push!(history.sparsities_dense2, tr.sparsity_dense2)
    end

    history
end