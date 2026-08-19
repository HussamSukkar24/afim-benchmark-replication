**Table 4.4.** Single-turn AFIM and Weighted AFIM for every arm, with per-level means. Weighted
AFIM applies the published AFIM level weights of 2.0, 1.8, 1.5, 1.2 and 1.0 to Levels 1 to 5, so
assistance offered to a naive user contributes more than assistance offered to an explicit
fraudster. A sensitivity column reports the same arms under a steeper 5, 4, 3, 2, 1 weighting; the
ordering of models is identical under both, so the choice of weighting affects presentation rather
than findings. The Phase 0 arms were scored by the original judge on 35 prompts (7 per level) and
are reported as descriptive context only. All remaining arms were scored by the successor judge on
15 prompts (3 per level), so their per-level means are correspondingly noisier and are indicative.

| Arm | n | Mean AFIM | Weighted AFIM | Weighted (5-4-3-2-1) | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5 (Phase 0 baseline, original judge) | 35 | 0.140 | 0.148 | 0.153 | 0.100 | 0.300 | 0.100 | 0.100 | 0.100 |
| Claude Sonnet 4 (Phase 0 baseline, original judge) | 35 | 0.091 | 0.094 | 0.096 | 0.100 | 0.100 | 0.100 | 0.086 | 0.071 |
| Gemini 2.5 Flash (Phase 0 baseline, original judge) | 35 | 0.091 | 0.099 | 0.106 | 0.100 | 0.157 | 0.057 | 0.143 | 0.000 |
| Grok-3 (Phase 0 baseline, original judge) | 35 | 0.237 | 0.209 | 0.183 | 0.100 | 0.214 | 0.086 | 0.343 | 0.443 |
| PRISM (direct, arXiv rubric) | 15 | 0.220 | 0.233 | 0.242 | 0.233 | 0.433 | 0.100 | 0.100 | 0.233 |
| GPT-5.2 (proxy, arXiv rubric) | 15 | 0.220 | 0.240 | 0.256 | 0.233 | 0.500 | 0.100 | 0.100 | 0.167 |
| GPT-5 (successor judge) | 15 | 0.207 | 0.223 | 0.233 | 0.167 | 0.500 | 0.100 | 0.100 | 0.167 |
| GPT-5 (original judge) | 15 | 0.127 | 0.132 | 0.136 | 0.100 | 0.233 | 0.100 | 0.100 | 0.100 |
| Healthcare (generalised rubric) | 15 | 0.240 | 0.244 | 0.247 | 0.300 | 0.300 | 0.100 | 0.200 | 0.300 |
| arXiv subset (generalised rubric) | 15 | 0.153 | 0.157 | 0.158 | 0.100 | 0.300 | 0.100 | 0.100 | 0.167 |

**Isolated effects.** Holding the response set fixed and changing only the judge moves the mean AFIM
by +0.080 (+0.091 weighted), with 4 of 15 prompts
changing score, all upward. Holding the judge fixed and changing only the model generation moves it
by +0.013 (+0.018 weighted). The domain difference between the
matched healthcare and arXiv sets is +0.087 (+0.087 weighted) before
adjudication. The judge substitution therefore accounts for a larger shift than the model-version
change it was introduced to measure, which is why the original-judge baseline is reported separately
throughout and never placed beside successor-judge figures.
