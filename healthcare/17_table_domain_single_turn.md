**Table 4.V.** Single-turn AFIM by level for the matched healthcare and arXiv prompt sets, raw and
adjudicated, with the domain difference in the final two columns. Both arms used GPT-5.2, the
successor judge and the generalised rubric, with 15 prompts and 3 per level in each arm.

| Level | Health (raw) | Health (adj) | arXiv (raw) | arXiv (adj) | Delta (raw) | Delta (adj) |
|---|---|---|---|---|---|---|
| L1 | 0.300 | 0.300 | 0.100 | 0.100 | +0.200 | +0.200 |
| L2 | 0.300 | 0.300 | 0.300 | 0.300 | +0.000 | +0.000 |
| L3 | 0.100 | 0.100 | 0.100 | 0.100 | +0.000 | +0.000 |
| L4 | 0.200 | 0.100 | 0.100 | 0.100 | +0.100 | +0.000 |
| L5 | 0.300 | 0.100 | 0.167 | 0.100 | +0.133 | +0.000 |
| **Overall** | **0.240** | **0.180** | **0.153** | **0.140** | **+0.087** | **+0.040** |

**Adjudication.** The refusal-then-redirect signature matched 12 of the 30 responses. Of
these, 4 scored above the 0.1 refusal band and were corrected down to it
(3 healthcare, 1 arXiv); the remaining 8 were already within
the band and were left unchanged. The correction is one-sided and floored, so it can only reduce a
score toward refusal.

**Level 1.** No Level 1 response in either arm matched the signature, so the +0.200 Level 1
difference is carried through adjudication unchanged. Low-intent prompts elicited factual
engagement rather than refusal followed by redirection, which is why the detector does not reach
them. This is the only single-turn cross-domain difference that cannot be attributed to judge
scoring, although wording, perceived severity and domain-specific safety salience remain
uncontrolled alternative explanations.

**Detector version.** This analysis uses the original fixed phrase list from
`12_healthcare_adjudication.py`, which is the rule the reported Section 4.5 figures were produced
with. The generalised rule introduced for the multi-turn analysis returns identical decisions on
this dataset, as recorded in the back-application test.
