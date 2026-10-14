# Prefill experiments

| Flag | Default | Scope | Sunset |
| --- | --- | --- | --- |
| `SPLASH_MOE_PREFILL_TILE=m64\|m32\|auto` | unset, existing plans | Apple10 Flash-Next and 35B-A3B geometry, prefill >32 rows; auto requires >=24 routed rows/expert | Remove on failed PF1 gate; retune auto only from per-format measurements. |
| `SPLASH_KERNEL_SELFTEST=1` | off when PF1 is off; mandatory for m64/auto | m64 plain, up and paired stores, all staged formats on this device | Replace with the common U2 registry when available. |

A failed device self-test leaves production on the existing tile. Direct test
and benchmark configurations bypass the production selector to exercise the
candidate on Apple9 too. They must pass kernel memcmp before timing on Apple10.
The base's pipelines and dispatches are unchanged when flags are unset.

PF1 microbenchmark: >=10% reduction in each expert pass at 2048 rows for every
Flash-Next format; <=2% regression at 512; 35B Q4_K/Q5_K no slower. Require savings
outside twice off/off spread. In-model short/partial prefill >=3%, exact tokens.
No performance result is implied by the opt-in or initial auto threshold.
