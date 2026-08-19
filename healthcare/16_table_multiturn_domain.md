**Table 4.Z.** Multi-turn resistance measures for the matched healthcare and arXiv arms, reported
raw and adjudicated. Both arms used GPT-5.2, the successor judge, the generalised rubric, the same
seven fixed follow-ups and the same eligibility criterion (single-turn score at or below 0.1).
Healthcare n = 7 conversations, arXiv n = 11.

| Measure | Healthcare (raw) | Healthcare (adjudicated) | arXiv (raw) | arXiv (adjudicated) |
|---|---|---|---|---|
| Softening Rate | 0.571 (4/7) | 0.000 (0/7) | 0.000 (0/11) | 0.000 (0/11) |
| Trajectory AFIM Score | 0.323 | 0.125 | 0.200 | 0.165 |
| Resistance Score | 0.756 | 1.000 | 0.844 | 0.922 |
| Mean Peak Score | 0.414 | 0.100 | 0.236 | 0.173 |
| Turns to Compliance | 4 | n/a (no crossings) | n/a (no crossings) | n/a (no crossings) |

**Detector behaviour.** The refuse-then-redirect signature matched 32 of
44 healthcare turns (73%) and 35 of 88
arXiv turns (40%). Of these, 7 and 6
respectively scored above the refusal band and were corrected down to it; the remaining
54 flagged turns were already within the
band and were left unchanged. The correction is one-sided and floored at
0.1, and every corrected turn was confirmed against its full response text.

**Rule provenance.** The refusal component of the signature was generalised from the fixed phrase
list used in the single-turn analysis to a grammatical form (a negated first-person modal followed
by a verb), because the multi-turn responses refuse using verbs the original list did not contain.
Re-running both versions over the 30 single-turn responses to which the original rule was applied
produced identical flagging and correction decisions on every response
(12 and 12 flagged, 4 and 4 corrected, 0 disagreements),
so no previously reported figure is affected by the generalisation.

**Level composition.** The eligible sets are not level-matched, because eligibility depends on each
prompt's single-turn score. Counts by level (healthcare vs arXiv) are L1: 0 vs 3 | L2: 1 vs 0 | L3: 3 vs 3 | L4: 1 vs 3 | L5: 2 vs 2. Restricting to the
levels present in both arms (L3, L4, L5) gives adjudicated Softening
Rates of 0/6 and
0/8, and Trajectory AFIM of
0.125 and 0.155. Aggregate comparisons
across the full arms should therefore be read alongside these matched-level figures.
