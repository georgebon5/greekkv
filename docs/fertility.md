# Tokenizer fertility — Greek vs English

**Measured, not assumed.** Numbers below come from `results/fertility.csv`,
produced by `experiments/fertility.py` on 500 FLORES-200 `devtest` sentence
pairs (same content in both languages), with 1000 bootstrap resamples and
seed = 42. Reproduce with:

```bash
python experiments/fertility.py --n-samples 500 --n-boot 1000 --seed 42 \
       --out results/fertility.csv
```

## Setup

- **Corpus**: FLORES-200 (`Muennighoff/flores200`), `devtest` split, aligned
  English↔Greek. Each pair is the same sentence in two languages, so any
  difference in token count is attributable to the tokenizer + language pair,
  not content.
- **Tokenizers**: `ilsp/Llama-Krikri-8B-Instruct`, `ilsp/Meltemi-7B-Instruct-v1.5`,
  `Qwen/Qwen2.5-7B-Instruct`.
- **Metrics**:
  - **TPW** = tokens per whitespace-separated word.
  - **TPC** = tokens per non-whitespace character.
  - 95% CI via percentile bootstrap (1000 resamples with replacement).

## Results

### Tokens per word (95% CI)

| Tokenizer | English | Greek | Greek / English |
|---|---|---|---|
| `ilsp/Llama-Krikri-8B-Instruct` | **1.269** [1.253, 1.284] | **1.529** [1.508, 1.552] | **1.20×** |
| `ilsp/Meltemi-7B-Instruct-v1.5` | **1.372** [1.353, 1.391] | **1.429** [1.409, 1.449] | **1.04×** |
| `Qwen/Qwen2.5-7B-Instruct`      | **1.295** [1.276, 1.315] | **5.674** [5.623, 5.725] | **4.38×** |

### Tokens per non-whitespace character (95% CI)

| Tokenizer | English | Greek | Greek / English |
|---|---|---|---|
| `ilsp/Llama-Krikri-8B-Instruct` | **0.251** [0.248, 0.254] | **0.273** [0.269, 0.277] | **1.09×** |
| `ilsp/Meltemi-7B-Instruct-v1.5` | **0.270** [0.267, 0.274] | **0.255** [0.252, 0.259] | **0.94×** |
| `Qwen/Qwen2.5-7B-Instruct`      | **0.256** [0.252, 0.261] | **1.007** [1.003, 1.010] | **3.93×** |

## Reading the numbers

- **Qwen2.5 is catastrophically inefficient on Greek.** It spends ~4× more
  tokens to encode the same content in Greek compared to English. At TPC ≈ 1.0
  it is essentially falling back to one token per character — the BPE merges
  for Greek sub-words are missing.
- **Meltemi is tuned for Greek.** TPW is near parity (1.04×), and on a *per
  character* basis Greek is slightly **cheaper** than English (0.94×). This is
  consistent with a Greek-adapted vocabulary.
- **Krikri sits in between.** A ~20% TPW penalty for Greek — much better than
  general multilingual models, but it does not fully erase the gap.

The confidence intervals are tight (n = 500), so none of the Greek/English
differences above are plausibly noise.

## Implications for KV-cache cost

KV memory grows linearly in `n_ctx` (tokens in cache). If the same Greek
document costs **4.4×** more tokens in Qwen2.5 vs English, the KV cache for
that document is also 4.4× larger — which translates directly into fewer
concurrent users served on the same GPU at the same context budget. This is
the opening argument the project makes quantitative in M5.

## Honest caveats

- **FLORES-200 is written text from Wikipedia/news genres.** Chat / informal
  Greek may fragment differently. Follow-up with a Greek Wikipedia sample is
  on the roadmap.
- **TPW depends on how you count words.** We use whitespace split. Punctuation
  attaches to the preceding word, which inflates "word length" but does so
  identically across tokenizers, keeping the comparison fair.
- **This measures only the tokenizer.** It does not yet say how well the model
  *uses* its tokens. That is what T17 (perplexity) and T18 (benchmarks) are
  for.
