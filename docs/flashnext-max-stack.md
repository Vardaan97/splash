# Combined Flash-Next test stack

Local branch `flashnext-max-stack` starts at stack-review
`0a08372e45874cf8e79ed6c77ad4c949e91e81a0`: #253
`e5ce29f80affff9ba98722f35913f77a3bac9b78`, #410 PLE streaming,
the draftless oracle fix, A hyper norm, B draft vocabulary and C lane depth.
See `flashnext-stack.md` for inherited provenance and merge decisions.

Folded in order, each as one topic with its review fixes:

| Item | Source base | Reviewed source head | Purpose |
| --- | --- | --- | --- |
| D | e5ce29f | ac81e2a55bb859d074c5b23d41f8263debc6e6eb | Retained GDN checkpoints (#387) |
| E | 0a08372e | 8a15e912beae67fcf97c6fb28ce1678774f2d076 | Explicit concurrent dispatch groups |
| F | 891a0fc75be775785caf132a60964b52d0f5c0b7 | 6d8e3db0469c8a60a9bb34c36a9d1fe7cd0faf85 | Prompt lookup in MTP lanes |

## Flags

No D/E/F behavior is enabled by default. C retains the base's default cost
policy; A/B retain the base's existing behavior. Source
`dev/scripts/flashnext-max.env` to opt into the complete test configuration.

| Flag | Default / off | Test value / meaning |
| --- | --- | --- |
| SPLASH_MTP | 1 / 0 disables MTP | 1 |
| SPLASH_MTP_DRAFT_VOCAB | 1 / 0 uses full head | 1 (B) |
| SPLASH_MTP_DEPTH | cost / rule selects older policy | cost (C) |
| SPLASH_MTP_ADAPTIVE | 1 / 0 fixes draft depth | 1 (C) |
| SPLASH_MTP_DRAFTS | 7 | 7, maximum head proposals |
| SPLASH_GDN_CHECKPOINT_TOKENS | unset or 0 | 16384 (D); multiples of 2048 >=4096; 8192 uses more memory |
| SPLASH_DECODE_DISPATCH | unset or 0 | exactly 1 (E) |
| SPLASH_MTP_LOOKUP | unset or 0 | exactly 1 (F) |
| SPLASH_MTP_LOG | unset or 0 | optional 1 for plan/lookup diagnostics |

C's internal probe/exploration intervals retain their defaults of 8/32
cycles; they are component parameters rather than environment flags. `--ple-ssd-cache 2G` enables inherited PLE streaming;
it is a CLI option and is not forced by the environment file.

## Integration and caveats

D's oracle conflict preserves `model.hasDraft()` in the inherited capture
check and adds the new grid fixture only for Flash-Next targets. E/F's
receipt README conflicts preserve all three source locations. F's lookup
mix dispatch remains before verify-input construction and PLE hashing in
the first streamed command; E gives it a conservative dependency barrier.

Use fp32 state for the mandatory greedy identity gates. D's fp32 split
oracle checks complete cells and logits bitwise. bf16 state rounds at extra
grid boundaries, so bf16 plus D is a numerics-changing experiment. D's
states consume governor-managed memory; optional reservation refusal can
split prefill commands and regress latency. Persistent cache restart does
not guarantee every historical grid point survives.

E's independence annotations require manual alias proofs; shuffle/shadow
modes remain future work. Profiling dispatch replay is attribution only.
F indexes at most 256 new tokens per update and caps both maps at 4096 keys
(estimated conservative 2 MiB per eligible lane). Skipping/eviction can
reduce hits. Graft eligibility can freeze head acceptance even when no graft
is applied; whole lookup saves head work, graft fallback retains it.

B1 greedy token identity is mandatory, including C/F drafting changes.
Known B4 nondeterminism makes free-running B4 comparison informational.
Real model identity, streaming and speed remain pending the exclusive M5
gates in `~/Developer/max-stack`; those scripts never build or download.
M4 validation commands and receipts are recorded in that directory's STATUS.

The native transcript tool also reads D's environment flag through the same
validated parser as `serve-native`; without that integration its bootstrap
would silently leave the grid off. It uses fp32 state by default.

## Reserved follow-up slots

Leave 2–4 subsequent topics for explicit items in
`~/Developer/overnight-1014/DECODE-GAP-REVIEW.md` and
`~/Developer/gpu-arch-research/FOLD-IN-PLAN.md`. Neither existed at initial
assembly; no speculative implementation is included.
