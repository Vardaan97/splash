# Compact QSA prefill attention (PF7, class N)

| Flag | Default | Meaning | Promotion / sunset |
| --- | --- | --- | --- |
| `SPLASH_PREFILL_QSA_COMPACT` | unset | `validate` runs the candidate after U2; `1` refuses production promotion and leaves the baseline active while quality gates are pending | Keep opt-in until every gate below passes on the exact SHA/model/device. Owner review of that report may authorize a follow-up enabling `1`; never silently default on |
| `SPLASH_PREFILL_QSA_CAPTURE` | unset | Path to a fresh binary full-vocabulary logit capture, from completed score-only requests; works with the flag off for reference capture | Diagnostic only; remove once the shared logit capture harness replaces it |
| `SPLASH_KERNEL_SELFTEST` | 0 | `1` makes U2 failures fatal; the candidate always self-tests when requested | Replace with the universal device hook when available |

Only GQA24/2 prefill with a live bitmap changes. Decode, verify, dense attention,
27B/35B and absent-bitmaps use their original paths. Apple9 stays unavailable.
The ascending bitmap union is shared by both KV heads of an eight-row tile.
It is gathered into Page32-shaped threadgroup operands using the compact decode
implementation's int8/BF16 copies, scale handling and per-row causal mask (6c81c78).
Nine fixed splits target at most 64 gathered cells (512 selected four-token blocks)
per split for the production 8×513-block union. Arbitrary dense bitmaps remain correct,
with longer splits. Each split and the existing LSE merge consume cells in fixed order.
Full physical-context list capacity avoids a top-k capacity assumption. Scratch is
bounded, allocation-accounted and request-owned; there are no arrival counters.
The serial union scan is a first implementation; optimize it only after measuring it.

## Mandatory quality gate (all conditions, separately at each length)

Reference is the same candidate build with all experiment flags unset, same model
revision/quant/KV/state/chunk boundaries/token IDs. Run ABBA off/on/on/off with clean
SPLASH environments. Neutral long text must be genuinely tokenized source text,
not repeated padding. Record source, tokenizer, model, token-file and binary SHA-256.
Use 32768 / 131072 / 204800 tokens, scoring 256 checkpoints (stride 64) across the
last 16384 text tokens, with the complete vocabulary at every position. The checked-in
`model-runtime-oracle --long-prompt-file ... --long-score-stride 64` captures those
positions while restoring its previous checkpoint; this preserves long history
without repeating each entire prefill. The normal oracle runs independently too.

`dev/gates/pf7-quality.py MANIFEST.json` must PASS:

* Direct KLD P(bitwise)||P(compact): mean ≤2e-4 nats/token, p95 ≤5e-4, worst ≤2e-3
  at EACH context length. This uses the neutral KLD METHOD.md pattern with a Splash
  bitwise reference, not AtomicChat BF16 logits from 4096-token llama.cpp runs.
  All full logits finite; baseline repeat and candidate repeat KLD ≤1e-8.
* Model-runtime-oracle class **N / partition tolerance**, never bitwise: every
  existing oracle assertion passes off and on, with its existing partition-state cosine bounds
  ≥0.999 for fp32 state/int8 KV, ≥0.99 for bf16 state/BF16 KV. The capture comparator
  applies a separate cross-arm cosine ≥0.999 to full-logit rows. No tolerance is relaxed to accept PF7.
* Teacher-forced top-1 agreement ≥99% at each length; teacher-forced acceptance
  absolute delta ≤0.01. Report both arms and the denominator, including baseline repeat controls.
* Long-prompt greedy continuations: ≥14 distinct prompts per context, ≥128 generated
  tokens each, aligned output-token agreement ≥99%, and ≥13/14 complete continuations
  identical at EACH length. Record early EOS and failures; neither counts as agreement.
* Needle checks: 64K/128K/200K × depths 10/50/90% × two seeds, 18 exact-answer passes
  in both arms. No candidate regression and no failed baseline trial may count as a pass.

The manifest includes `vocabulary`, `contexts` (tokens, prompt_sha256, off0/on0/on1/off1
capture paths), oracle rows (kv, state, off_pass, on_pass, min_state_cosine),
greedy rows (context, off/on token lists), needles (context, depth, seed, expected,
off, on), and acceptance (off/on rates). Capture paths are relative to the manifest.
Records are little-endian uint32 vocabulary followed by that many little-endian float32 logits.
The comparator refuses missing lengths, short/unequal/truncated captures, non-finite
logits, shortlists, missing oracle configurations, insufficient continuations or needles.

U2 checks finite absolute error ≤0.02, cosine ≥0.999 against the baseline and candidate
repeat memcmp on this device, int8/BF16, shared/independent/page/empty masks. U3 expands
this across tail rows, 2K rows and long histories under shader validation. The normal
fp64 attention oracle must also retain its existing error envelope before any timing;
class N does not excuse a relaxed kernel reference test.

Only after U0–U5/U7, this full quality report, recovery/cache checks, and the fp64
reference gate pass may performance be measured: attention sweep and 38K/114K/227K
TTFT ABBA, per-dispatch GPU times and off/off spread. Expected +15%/+25%/+40%
throughput at 32K/128K/200K are hypotheses, not results. No promotion is authorized
by this implementation, a build, or a microbenchmark alone.
