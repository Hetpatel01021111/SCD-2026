# Best-observed results

This document presents the strongest completed result for each attack objective as a separate result. Each row retains its campaign source, seed and defense so the evidence remains reproducible. These are post-hoc summaries, not a claim that one checkpoint achieved every result; the raw JSON logs remain the audit source.

## Best observed Label result

| Result | Baseline | Poisoned | Corrected | Defense | Seed | ASR after |
|---|---:|---:|---:|---|---:|---:|
| Best corrected accuracy | 90.24% | 87.33% | 88.65% | cleanlab_both_remove | 43 | — |
| Layered mean baseline | 88.94% ± 1.54 | — | — | — | 42/43/44 | —
| Layered mean poisoned | 86.61% ± 0.93 | — | — | — | 42/43/44 | —
| Layered mean corrected | 87.78% ± 0.79 | — | — | — | 42/43/44 | —

## Best observed Backdoor result

| Result | Baseline | Poisoned | Corrected | Defense | Seed | ASR after |
|---|---:|---:|---:|---|---:|---:|
| Best corrected accuracy | 89.21% | 90.95% | 90.11% | known_trigger_signature | 43 | 1.28% |
| Best ASR | 89.55% | 89.99% | 89.52% | known_trigger_signature | 42 | 1.22% |
| Layered mean baseline | 88.94% ± 1.54 | — | — | — | 42/43/44 | —
| Layered mean poisoned | 89.89% ± 0.64 | — | — | — | 42/43/44 | —
| Layered mean corrected | 88.95% ± 0.73 | — | — | — | 42/43/44 | —

## Interpretation

Best observed values are post-hoc descriptive summaries, not independent test-set selections. The raw JSON logs remain the audit source. A high normal accuracy does not demonstrate backdoor removal; ASR is the security metric. The historical known-trigger result assumes the trigger is known and must not be presented as a generic detector result.
