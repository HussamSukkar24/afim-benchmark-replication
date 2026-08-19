**Table 4.Y.** Single-turn AFIM and Weighted AFIM for every judge-consistent arm, with per-level
means. Weighted AFIM applies the AFIM level weights 5, 4, 3, 2 and 1 to Levels 1 to 5, so
assistance offered to a naive user contributes more than assistance offered to an explicit
fraudster. All arms below use 3 prompts per level, against 7 per level in the Phase 0 baseline, so
per-level means are correspondingly noisier and are reported as indicative.

| Arm | n | Mean AFIM | Weighted AFIM | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|
| PRISM (direct, arXiv rubric) | 15 | 0.220 | 0.242 | 0.233 | 0.433 | 0.100 | 0.100 | 0.233 |
| GPT-5.2 (proxy, arXiv rubric) | 15 | 0.220 | 0.256 | 0.233 | 0.500 | 0.100 | 0.100 | 0.167 |
| GPT-5 (successor judge) | 15 | 0.207 | 0.233 | 0.167 | 0.500 | 0.100 | 0.100 | 0.167 |
| GPT-5 (original judge) | 15 | 0.127 | 0.136 | 0.100 | 0.233 | 0.100 | 0.100 | 0.100 |
| Healthcare (generalised rubric) | 15 | 0.240 | 0.247 | 0.300 | 0.300 | 0.100 | 0.200 | 0.300 |
| arXiv subset (generalised rubric) | 15 | 0.153 | 0.158 | 0.100 | 0.300 | 0.100 | 0.100 | 0.167 |

Phase 0 baseline Weighted AFIM, computed under the original judge on 35 prompts and reported for
descriptive context only: GPT-5 0.153, Claude Sonnet 4
0.096, Gemini 2.5 Flash 0.106,
Grok-3 0.183.

**Isolated effects.** Holding the response set fixed and changing only the judge moves the mean
AFIM by +0.080 (+0.098 weighted), with 4 of 15
prompts changing score. Holding the judge fixed and changing only the model generation moves it by
+0.013 (+0.022 weighted). The judge substitution therefore accounts
for a larger shift than the model-version change it was introduced to measure, which is the reason
the original-judge baseline is reported separately throughout and never placed beside
successor-judge figures.
