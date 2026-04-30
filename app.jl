using Genie, Genie.Router, Genie.Renderer.Html, Genie.Requests
using Flux, NNlib, Statistics, Functors, Unicode, DataStructures, Random, BSON, Printf

include("src/layers.jl")
include("src/attention_causal.jl")
include("src/transformer_decoder.jl")
include("src/tokenizer_decoder.jl")
include("src/generate.jl")

BSON.@load "tokenizer.bson" tokenizer
BSON.@load "model.bson" model
println("Model loaded ✓")

route("/") do
    html("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BayesGPT-I</title>
        <style>
            body { font-family: monospace; max-width: 800px; margin: 40px auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
            h1 { color: #58a6ff; }
            input, select { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; padding: 8px; border-radius: 6px; width: 100%; margin: 5px 0; }
            button { background: #238636; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; margin-top: 10px; }
            button:hover { background: #2ea043; }
            .result { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; margin-top: 20px; white-space: pre-wrap; }
            .label { color: #8b949e; font-size: 0.85em; margin-top: 10px; }
            .uncertain { color: #f85149; }
            .moderate  { color: #e3b341; }
            .confident { color: #3fb950; }
        </style>
    </head>
    <body>
        <h1>BayesGPT-I 🎲</h1>
        <p>A minimal Bayesian GPT trained on PKD, Lovecraft and a space western.<br>
        <em>Slightly paranoid about its own weights.</em></p>

        <form action="/generate" method="POST">
            <div class="label">Seed phrase</div>
            <input type="text" name="seed" value="he knows" />

            <div class="label">Temperature (0.1 - 1.5)</div>
            <input type="range" name="temperature" min="0.1" max="1.5" step="0.1" value="0.8"
                   oninput="this.nextElementSibling.value=this.value">
            <output>0.8</output>

            <div class="label">Top-k (10 - 100)</div>
            <input type="range" name="k" min="10" max="100" step="10" value="50"
                   oninput="this.nextElementSibling.value=this.value">
            <output>50</output>

            <div class="label">Max new tokens (3 - 15)</div>
            <input type="range" name="max_length" min="3" max="15" step="1" value="5"
                   oninput="this.nextElementSibling.value=this.value">
            <output>5</output>

            <div class="label">Bayesian forward passes (5 - 30)</div>
            <input type="range" name="n_forward" min="5" max="30" step="5" value="20"
                   oninput="this.nextElementSibling.value=this.value">
            <output>20</output>

            <button type="submit">Generate ✨</button>
        </form>
        <div class="result">Enter a seed and click Generate!</div>
    </body>
    </html>
    """)
end

route("/generate", method=POST) do
    seed        = postpayload(:seed, "he knows")
    temperature = parse(Float32, postpayload(:temperature, "0.8"))
    k           = parse(Int,     postpayload(:k, "50"))
    max_length  = parse(Int,     postpayload(:max_length, "5"))
    n_forward   = parse(Int,     postpayload(:n_forward, "20"))

    # standard samples
    std_samples = [generate(model, tokenizer, seed;
                            max_length, temperature) for _ in 1:3]

    # top-k samples
    topk_samples = [generate_topk(model, tokenizer, seed;
                                  max_length, temperature, k, penalty=1.5f0) for _ in 1:3]

    # bayesian
    result = bayesian_generate(model, tokenizer, seed;
                               n_forward, max_length, temperature, k, penalty=1.5f0)

    # build uncertainty rows
    uncertainty_rows = join([
        begin
            confidence = σ < 0.1f0 ? "confident" : σ < 0.3f0 ? "moderate" : "uncertain"
            bar = "█" ^ min(Int(round(σ * 40)), 30)
            """<tr>
                <td>$word</td>
                <td>$(round(σ, digits=3))</td>
                <td>±$(round(ci/2, digits=3))</td>
                <td class="$confidence">$bar $confidence</td>
            </tr>"""
        end
        for (word, σ, ci) in zip(result.new_words, result.uncertainties, result.conf_intervals)
    ], "\n")

    html("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BayesGPT-I</title>
        <style>
            body { font-family: monospace; max-width: 800px; margin: 40px auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
            h1 { color: #58a6ff; }
            h2 { color: #58a6ff; font-size: 1em; margin-top: 20px; }
            a { color: #58a6ff; }
            .box { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; margin: 10px 0; }
            .sample { margin: 5px 0; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
            th { color: #8b949e; text-align: left; padding: 6px 10px; }
            .uncertain { color: #f85149; }
            .moderate  { color: #e3b341; }
            .confident { color: #3fb950; }
            .generated { color: #58a6ff; font-size: 1.1em; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <h1>BayesGPT-I 🎲</h1>
        <a href="/">← back</a>

        <h2>Seed: "$(seed)"</h2>

        <h2>Standard sampling</h2>
        <div class="box">
            $(join(["<div class='sample'>$(i). $(s)</div>" for (i,s) in enumerate(std_samples)], "\n"))
        </div>

        <h2>Top-k=$k + repetition penalty</h2>
        <div class="box">
            $(join(["<div class='sample'>$(i). $(s)</div>" for (i,s) in enumerate(topk_samples)], "\n"))
        </div>

        <h2>Bayesian ($n_forward forward passes)</h2>
        <div class="box">
            <div class="generated">$(result.text)</div>
            <table>
                <tr><th>token</th><th>σ</th><th>95% CI</th><th>confidence</th></tr>
                $uncertainty_rows
            </table>
        </div>
    </body>
    </html>
    """)
end

Genie.up(8080, "0.0.0.0", async=false)
