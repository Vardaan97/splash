# Flash-Next stack

[Back to Splash](../README.md)

This branch, `flashnext-stack`, combines the open Qwen3.8-Flash-Next changes
into one build. It is for testing them together. It is not a pull request.
Each change is reviewed in its own pull request, listed below.

## What it holds

The base is #253 (Qwen3.8-Flash-Next GGUF support) at `e5ce29f`. On top of it,
in order:

| Commits on this branch | Upstream | Change |
| --- | --- | --- |
| `806033e`..`a619802` (13) | #410 | Stream the PLE table from the GGUF with `--ple-ssd-cache` (fixes #372) |
| `d1ddd7e` (1) | Fix offered on #253 | The runtime oracle's captured-state check for a target without a DFlash draft |
| `7f7ca3e` (1) | #423 (part of #422) | Load ahead in the hyper-connection norm |
| `db39985`..`54e1cf7` (8) | PR B (part of #422) | Draft MTP proposals from a draft vocabulary and the request's context |
| `ab8b14d`..`5b13f94` (5) | PR C (part of #422) | Choose each MTP lane's depth by its expected tokens per second |

PR B and PR C are not open yet. PR C is stacked on PR B.

`54e1cf7` also carries a fix to `make test-performance-real`: it buckets short
samples by `target_prefill_rows`, not `prompt_tokens - 1`. Since captured
prefills, a short prompt runs all its rows, and the old bucketing found no
short samples in a baseline at `e5ce29f`.

Two places needed merge work. Both are in this branch's own commits, not in
the upstream branches:

- `db39985`: `GgufTargetLoader` takes both the draft head's tokens (PR B) and
  the PLE streaming options (#410). The PLE table is detected before the draft
  head is added to the planned images.
- `ab8b14d` and `fab7d6c`: the streamed-PLE verify (#410) passes the host's
  encoding time to the MTP cost model (PR C), as the other decode paths do. The
  CPU staging of PLE rows between the draft's and the verify's commands counts
  as encoding time.

## Build

```sh
git clone https://github.com/Vardaan97/splash.git
cd splash
git checkout flashnext-stack
make -j8
```

## Run Qwen3.8-Flash-Next

```sh
./splash serve --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL
```

Optional: read the PLE table from the SSD through a 2 GiB RAM cache instead of
holding it in memory.

```sh
./splash serve --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL --ple-ssd-cache 2G
```

In IQ4_NL the PLE table is 28.8 GB. Streaming it leaves that memory to the KV
cache. Within a 128 GB Mac's default Metal budget, `UD-Q4_K_XL` loads only
with streaming (#372). Other variants, such as `UD-IQ4_XS`, run the same way.

To compare against the base behaviour in the same build:

| Setting | Effect |
| --- | --- |
| `SPLASH_MTP_DRAFT_VOCAB=0` | Draft from the whole head (turns off PR B's draft vocabulary) |
| `SPLASH_MTP_DEPTH=rule` | Common threshold depth (default on max-stack; C remains available with cost) |
| `SPLASH_MTP=0` | Turn MTP off |

Leaving out `--ple-ssd-cache` keeps the PLE table in memory, as #253 does.

## Results

TODO: to be filled after GPU validation. No numbers have been measured on this
combined branch yet.

| Check | Result |
| --- | --- |
| `make check` | TODO |
| `model-runtime-oracle`, resident PLE table (fp32, bf16, default) | TODO |
| `model-runtime-oracle --ple-ssd-cache` | TODO |
| `make test-http-real`, resident and streamed | TODO |
| `make test-performance-real` against #253 at `e5ce29f` | TODO |

| Decode, M5 Max, greedy | #253 (`e5ce29f`) | This branch | This branch, `--ple-ssd-cache 2G` |
| --- | ---: | ---: | ---: |
| B1 short prompt (tok/s) | TODO | TODO | TODO |
| B4 short prompt (tok/s) | TODO | TODO | TODO |
| Long context (tok/s) | TODO | TODO | TODO |
| Accepted tokens per cycle | TODO | TODO | TODO |

TODO: model variant, machine, method and links to the raw results.
