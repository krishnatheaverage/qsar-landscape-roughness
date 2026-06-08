# Critical review memo — internal red-team of the manuscript

*Prepared as a substitute for, not a replacement of, an independent expert co-author read. I wrote the paper, so treat this as a structured self-critique: it flags the objections a careful JCIM reviewer is most likely to raise, and what each would take to address. An actual external reviewer remains necessary before submission.*

## Summary of the work being reviewed
The paper proposes per-compound, structure-only measures of local structure–activity landscape roughness, shows they predict per-compound QSAR error across 30 bioactivity targets, demonstrates the signal is distinct from the applicability domain, introduces an activity-free cliff-flagging descriptor, and validates the relationship on two physicochemical datasets. Methods and statistics are documented and reproducible.

## Strengths
- A genuine, documented gap (per-compound, model-agnostic, activity-free roughness vs. global/pairwise measures and applicability-domain methods).
- The decisive control (partial correlation + mixed-effects vs. applicability domain) is the right test and is reported transparently, including effects that strengthen under control.
- Honest treatment of negative/weak results (Hölder null; modest y-free effect sizes; hedged GNN observation; cross-domain variability).
- Reproducible pipeline with fixed splits, seeds, and versions.

## Major points (likely reviewer concerns) and recommended responses

1. **The landscape descriptors use the query's own activity, so their strong correlation (Dirichlet ρ ≈ 0.65) is partly mechanical.** The paper says this, but the Abstract/headline still lead with the predictive framing. *Response/fix:* center the headline claim explicitly on the activity-free constructs and the applicability-domain independence; present the landscape descriptors as mechanistic confirmation, not as the deployable result. (Largely done; tighten the Abstract's first result sentence.)

2. **Effect sizes for the deployable (activity-free) signal are modest** (ρ ≈ 0.16–0.28; cliff-flag AUC ≈ 0.69). A reviewer will ask whether this is operationally useful. *Recommended addition (cheap, high value):* an enrichment / decision-curve analysis — if one flags the top X% roughest compounds, what fraction of the highest-error (or labeled-cliff) compounds is captured, versus flagging by applicability domain or by ensemble uncertainty? This reframes modest correlations as a concrete triage benefit and directly answers the "so what" question.

3. **"Model-agnostic" is claimed, but per-compound error is measured almost entirely on a random forest** (plus the GNN comparison). *Recommended addition (cheap):* replicate the roughness→error correlation for at least one additional descriptor model (gradient boosting, SVR, or kNN). If the relationship holds across model classes, the model-agnostic claim is earned; if it varies, that is itself worth reporting.

4. **The graph-network comparison rests on an uncompetitive, unoptimized GIN.** Even hedged, a reviewer may object that conclusions about "deep learning" cannot be drawn from one weak model. *Response/fix:* keep the observation explicitly secondary (done), and either (a) add a standard strong baseline (D-MPNN/Chemprop) or (b) reframe the subsection purely as "two training regimens of a standard GIN behave identically," removing any implication about deep learning broadly.

5. **Activity-cliff labels depend on the benchmark's specific thresholds** (>10-fold; >0.9 on one of three similarities). Sensitivity is untested. *Recommended addition:* vary the potency and similarity thresholds and confirm the cliff-flagging AUC and the cliff-vs-non-cliff error gap are not threshold artifacts.

6. **The ROGI contrast risks overstatement.** Reporting that a validated index "does not correlate with mean error here" invites pushback. *Response/fix:* frame it strictly as limited dynamic range across these curated, cliff-rich datasets (narrow ROGI band) plus the structural point that a per-dataset scalar cannot resolve per-compound error — not as a failure of ROGI. (Partly done; soften any remaining strong phrasing.)

7. **External validation is two datasets, and the y-free signal is weak on lipophilicity.** *Response/fix:* add at least one more endpoint (FreeSolv hydration free energy; or a QM9 quantum property) and state the cross-domain variability plainly rather than implying uniform transfer. (Variability is already stated; an extra dataset would strengthen it.)

## Minor points
- Report uncertainty on the headline statistics (bootstrap CIs on the aggregated Spearman medians and on the cliff-flag AUCs), not only IQRs and Wilcoxon p-values.
- Mixed-effects model: report the random-intercept variance and consider a random slope for roughness; confirm residual behavior.
- Make the leakage control for the a-priori descriptor explicit in the Results, not only in Methods (state that the query's activity is withheld and only training neighbors contribute).
- Hölder null: keep in SI; optionally test an alternative regularity estimator (e.g., a robust local slope) so the negative result is clearly methodological, not a single bad estimator.
- Fill the code/data repository URL and deposit the cached per-compound table as supporting information.
- A table-of-contents (TOC) graphic is required by JCIM.

## Overall assessment
A solid, honest contribution with a clear gap, the correct decisive control, and transparent limitations. If submitted as-is, the most probable outcome is **major revision**, with reviewers requesting items 2, 3, and 5 in particular. All major points except an independent expert read are addressable in silico at low cost; items 2 and 3 would most improve the paper's reception and are recommended before submission.
