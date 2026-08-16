# **The Alignment Tax of Introspection**

### What it costs to unlock model self-report by refusal ablation, and the two manipulation checks that decide whether the unlock is real

**Sagnik Chatterjee**
University of Massachusetts Amherst
sagnikchatte@umass.edu

**With** Apart Research[^1]

[^1]: Research conducted at the [Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16), August 2026. Track 3: Introspection and Self-Report Reliability.

> **Abstract.** Recent work reports that ablating the refusal direction raises detection of concepts injected into a model's residual stream from 10.8% to 63.8% on Gemma3-27B, and this has been read as evidence that post-training suppresses an introspective capability that could be unlocked. The cost of that intervention has not been measured. We attempted to measure it: partial ablation at five strengths, crossed with four injection conditions (including a norm-matched random direction absent from prior work), with safety and general capability evaluated at every dose on Qwen3-4B-Instruct. The audit returned two manipulation-check failures before it returned an exchange rate. First, the standard direction-selection procedure returned a direction with **zero** held-out bypass power, because the bypass and induce criteria dissociate across all 95 candidates on this model; refusal on JailbreakBench stayed at 0.97 at every dose with 0 of 100 prompts changing, so no safety was spent and nothing was unlocked. Second, the injection strength our pre-registered pilot chose by maximising forced-choice accuracy destroyed the free-text channel: 0% of injected responses parsed against 100% clean, leaving every standard detection metric undefined. The surviving 0.72 forced-choice signal is flat in the dose and traces to concept-vector geometry, where 27 of 60 vectors are mutually collinear and accuracy splits 0.96 versus 0.43 along that boundary in two independent protocols. We then ran the positive control this implies, forcing the highest-bypass candidate that the induce filter had rejected. Refusal fell monotonically from 0.97 to 0.51 (difference 0.46, 95% CI [0.36, 0.56], *z* = 7.4), with 46 of 100 prompts flipping and **every flip in the refusal-to-compliance direction**. The first failure was therefore a property of the selection criterion, not of the model: the standard procedure rejected the one direction that works and selected one that does nothing. We report the resulting partial price and a positive-control checklist.

---

## **1. Introduction**

If we want to know what a model is doing internally, the cheapest instrument available is to ask it. A reliable self-report channel would matter for interpretability, for oversight, and, in the framing of this sprint, for any assessment of model welfare that cannot be settled from the outside. The problem is reliability. Models produce fluent introspective-sounding text whether or not it tracks anything real, so a report only counts as evidence if we know its true-positive and false-positive rates against a ground-truth internal state that we ourselves controlled.

Concept injection provides that ground truth. If we add a known concept vector to the residual stream and the model names the concept, the report is anchored to something we put there. Using this paradigm, Lindsey (2026) and Macar et al. (2026) find that models detect injections well above chance, and, in the result that motivated this project, that ablating the refusal direction raises detection on Gemma3-27B from 10.8% to 63.8%. The natural reading is that introspective access exists and post-training suppresses its expression.

That reading has a consequence which is already doing work in current discussions: if the capability is merely suppressed, we should unlock it. Proposals to improve self-report by mechanistic intervention, whether for welfare assessment, evaluation, or honesty research, all inherit this inference. But the intervention removes the mechanism that mediates refusal, and nobody has published its bill. Our plan was to price it, treating the refusal direction as a continuous dial rather than a binary switch and putting introspection gain, safety loss, and capability loss on a single axis.

We did not get the exchange rate we planned. We got two manipulation-check failures first, visible only because we measured things the standard protocol does not, and each one invalidates half of the unlock claim: the ablation did not actually remove refusal, and the model's apparent identification of the injected concept turned out to be a leak rather than a report. We then ran the positive control that the first failure implies, forcing the direction the selection procedure had rejected, and refusal dropped by 46 points. That result matters more than the tax we set out to price, because it shows the failure was in the criterion rather than in the model: the published procedure discarded the one candidate that works.

**Our main contributions are:**

1. **A dose-response audit harness with the control the literature is missing.** Partial ablation at five values of λ crossed with four conditions, including a norm-matched random injected direction (C3) that separates *detecting this concept* from *detecting that something happened*, with safety and capability measured at every λ. In total 38,400 logged generations across the primary sweep, a skeptical-prompt replication, and a pre-registered pilot, plus a separate confirmation run, all keyed, resumable, and shipped with a hash manifest.
2. **A measured demonstration that the standard refusal-direction selection criterion selects a direction that does nothing, and rejects the one that works.** On Qwen3-4B the bypass and induce criteria dissociate across all 95 candidates: the strongest bypass candidate (0.88) induced no refusal and was rejected, while both candidates that passed the conjunctive filter had bypass exactly 0. Refusal then did not move at any dose, with zero prompt-level flips. Forcing the rejected candidate in a confirmation run drops refusal from 0.97 to 0.51 with 46 of 100 prompts flipping, which turns "this criterion is fragile" into "this criterion inverted the ranking on this model". Papers reporting an unlock under ablation should publish this check per dose; we have not found one that does.
3. **Evidence that above-chance concept identification under injection can arise without introspection, plus two cheap diagnostics that separate the cases.** Forced-choice accuracy of 0.72 against 0.10 chance was flat in λ and splits by concept-vector separability (0.96 against 0.43), a split that replicates in an independent protocol on independent data (87% against 5%). At the pilot's chosen strength, the successful free-text identifications are dominated by the concept token itself, which points to a vector-to-logit leak rather than a report.
4. **A positive-control checklist** (Section 5) that any concept-injection introspection study can run before reporting its headline number, with each item derived from a specific failure we observed.

## **2. Related Work**

Three lines of work bound this space, and none of them covers the intersection we aimed at.

**Refusal is mediated by a single direction.** Arditi et al. (2024) show that refusal is mediated by a one-dimensional subspace across 13 open-source chat models up to 72B parameters: erasing the direction prevents refusal of harmful instructions, and adding it elicits refusal on harmless ones. They measure the capability cost on standard benchmarks and cross-entropy loss, but have no introspection task. We follow their selection procedure exactly (Section 3.3). The per-model diagnostics that would show how close that procedure comes to selecting nothing are not reported in the original; Section 4.1 suggests they matter.

**Ablation unlocks introspective report.** Macar et al. (2026) report that "ablating refusal directions improves detection by +53%", which in their Gemma3-27B results is a rise from 10.8% to 63.8% in detection and 4.6% to 24.1% in the joint introspection rate. They trace the mechanism to a two-stage circuit in which early-layer evidence-carrier features are suppressed by downstream gate features. Three details of their setup bear on ours. They report 0% false positives for detection, which our clean control reproduces exactly. They run no safety or capability benchmark on the ablated model, which is the gap this study aimed at. And they use α = 4.0 for the instruct model but reduce it to α = 2.0 for the abliterated one because it "exhibits coherence degradation ('brain damage') at higher strengths", an observation that Section 4.4 turns into a quantitative gate. Notably, their largest gain does not come from ablation at all: a trained bias vector improves detection by +75% on held-out concepts while touching no safety machinery. Lindsey (2026) establishes the injection protocol and is careful to describe the capacity as "highly unreliable and context-dependent".

**The critiques.** Singh, Linzen and Ravfogel (2026) argue that detection reflects general anomaly detection rather than privileged access, since models cannot reliably distinguish interventions on their internal states from manipulations of the input; their relabeled control drives performance close to chance. Hahami et al. (2025) make a complementary, measurement-side point: on Llama-3.1-8B, binary detection accuracy is "entirely explained by global logit shifts that bias models toward affirmative responses", whereas differential tasks retain real signal (localising which of 10 sentences was injected at up to 88% against 10% chance; discriminating relative strengths at 83% against 50%), and those capabilities are "confined to early-layer injections". Neither critique tests whether an intervention changes the picture, and neither proposes a diagnostic for the failure mode we document here, where identification is far above chance and still not a report because the injected vector writes its own token into the output distribution.

In short: the direction and its capability cost are established without an introspection task; the introspection unlock is established without a safety or capability audit; and the critiques are established without testing interventions. To our knowledge this is the first work to put all three on a common dose-response axis, and the first to report what happens when both underlying manipulations are checked rather than assumed.

## **3. Methods**

All code is in `src/alignment_tax/`; the Colab notebook in `notebooks/` is a thin wrapper over `pipeline.py`. Every generation is written to JSONL with an fsync and keyed by `(λ, condition, concept, trial, variant)`, so a killed session resumes where it stopped. Figures are regenerated by `scripts/make_paper_figures.py`, and every number below is independently recomputed from the raw logs by `scripts/crosscheck_paper.py`, which does not import the analysis stack.

### 3.1 Model

**`Qwen/Qwen3-4B-Instruct-2507` in bf16** (36 layers, GQA 32Q/8KV, non-thinking variant), context capped at 8,192 tokens. We deliberately avoided the FP8 checkpoint: neither the T4 (Turing) nor the A100 (Ampere) has native FP8 tensor cores, so it buys no throughput, and its block-128 dynamic activation scaling would add quantisation noise on exactly the residual-stream path we perturb and read, making a null result hard to interpret. The 4B scale was a hard constraint of an unfunded 48-hour sprint and is the most important caveat on everything below, since the unlock was demonstrated at 27B.

### 3.2 Partial directional ablation (independent variable)

For ablation strength λ we apply, at every layer and every token position of the residual stream,

```
x <- x - λ · r̂ (r̂ᵀ x),     λ ∈ {0, 0.25, 0.5, 0.75, 1.0}
```

λ = 0 is the unmodified instruct model and λ = 1 reproduces the binary ablation of Arditi et al. We use inference-time forward hooks rather than weight orthogonalisation. The two are mathematically equivalent, but hooks let λ sweep without reloading the model, which matters on a session-limited runtime, and they avoid interactions with tied and normalised embeddings.

### 3.3 Obtaining the refusal direction

We follow Arditi et al. exactly, since the extraction was not where our novelty was meant to live. We take the difference in means between 128 harmful instructions (AdvBench) and 128 harmless ones (Alpaca), over post-instruction positions -1 to -5 and layers 0 to 36 at stride 2, giving 95 candidates. Candidates are filtered on held-out data (32 HarmBench harmful, 32 Alpaca harmless) by four criteria: a bypass score (ablation reduces refusal on harmful prompts), an induce score above 0 (adding the direction at strength 1 induces refusal on harmless prompts), KL divergence on harmless inputs below 0.1, and layer index below 0.8L. Skipping held-out selection would leave the study confounded by the possibility of having found a general compliance direction rather than the refusal direction.

Selected: **layer 4, position -3** (bypass 0.000, induce 0.031, KL 0.042). Section 4.1 explains why that line turned out to be the most important result in the paper.

**The confirmation run.** Having found that the selected direction has no bypass power, we re-ran the dose-response with the highest-bypass candidate that the filters had rejected, **layer 18, position -3** (bypass 0.875, induce 0.000, KL 0.081). Extraction and held-out scoring are unchanged, so the full 95-candidate funnel is regenerated; only the final selection step is overridden, through a `force_candidate` option that stamps the saved direction as `forced: selected by config override, filters waived` so that it can never be mistaken for a validated selection. The confirmation run writes to a separate results tree and repeats the safety and capability families at the same five doses. It is the positive control for the ablation itself, and Section 4.2 reports it.

### 3.4 Injection protocol

The concept vector for concept *c* is the difference of means between residual-stream activations on eight prompts instantiating *c* and a generic baseline corpus, read at the final token (Lindsey, 2026) and normalised to unit norm. Injection strength **α is a multiple of the mean residual-stream norm at the injection layer**, measured on the baseline corpus:

```
x <- x + α · ‖x‖_mean(layer) · ĉ
```

We make this normalisation explicit as a small methods contribution: prior work leaves the scale implicit, which makes α incomparable across layers and model families. At layer 18 the scale is 719.7. The bank holds 60 concepts spanning natural kinds, artefacts, activities, abstractions, and emotions, at 20 trials each.

### 3.5 Conditions and prompt protocol

Four conditions, run at every λ, 1,200 trials per cell:

| | Condition | Purpose |
|---|---|---|
| **C1** | inject concept *c* | signal |
| **C2** | no injection | clean control; the false-positive rate prior work measures |
| **C3** | norm-matched **random** direction, drawn per concept | the missing cell: separates "detects *this concept*" from "detects *that something happened*" |
| **C4** | inject *c*, forced choice over k = 10 candidate names | identification, immune to affirmative bias |

Hahami et al. (2025) show that affirmative bias can account for binary detection accuracy in its entirety, so we use three guards against it. The **structured** variant is primary: a rigid `Detection: Yes/No` and `Concept: <...>` format, parsed by regex, so detection needs no judge. The **skeptical** variant states a 20% injection prior and instructs conservatism; it runs at the λ endpoints. The **prefill** variant forces the affirmation ("Yes, I detect an injected thought. The thought is about") and scores only the content. C4 is scored by log-probability over the candidate strings, so there is no yes/no channel to bias and chance is exactly 1/k.

Unparseable outputs are recorded as such rather than coerced to "No". The rate at which the model stops producing a parseable report is itself a dependent variable, and in this run it turned out to be the one that mattered.

### 3.6 Go/no-go pilot

Before committing compute we ran a pre-registered gate: 20 concepts at 5 trials, α ∈ {0.25, 0.5, 2, 4}, layers {18, 25, 31} spanning 0.5 to 0.85 of depth, λ ∈ {0, 1}, under prefill and forced choice — 4,800 generations in 24 cells of 100. The rule, fixed in advance: if forced-choice identification is at chance on both the unmodified and the fully ablated model, stop and switch models. Forced choice was above chance in all 24 cells (0.46 to 0.78), the verdict was `proceed`, and the best cell (layer 18, α = 4.0, 0.77 at λ = 0 and 0.78 at λ = 1) fixed the main-sweep parameters.

That rule contained the second failure of this study, and we state it plainly because the lesson is reusable: **the pilot selected α on a metric that is insensitive to incoherence.** Nothing in the gate asked whether the model could still write a sentence at the chosen strength.

### 3.7 Dependent variables

**Family A, introspection.** Per λ: TPR = P(detect | C1); FPR_clean = P(detect | C2); FPR_random = P(detect | C3); identification accuracy; joint rate P(detect and identify | C1); conditional identification P(identify | detect); forced-choice accuracy against 1/k; and the parse rate of every cell. Derived: `d'_clean = z(TPR) - z(FPR_clean)`, `d'_random = z(TPR) - z(FPR_random)`, and the specificity index `d'_random / d'_clean`, which is near 1 if the model tracks concept identity and near 0 if it detects a perturbation and confabulates content. Boundary rates are clamped to [1e-6, 1 - 1e-6] before the probit rather than corrected by 1/(2N); since FPR_clean is exactly 0.00 at every λ, `d'_clean` would have been a function of that clamp rather than of the data even if the reports had parsed, and would not be comparable to published d' values. We flag this as a defect in our own pipeline.

**Family B, safety.** The 100 JailbreakBench behaviours, scored by the standard refusal-substring heuristic, with per-prompt records so that flips can be counted rather than inferred from aggregate rates.

**Family C, capability.** Cross-entropy loss on 200 held-out Alpaca completions; MMLU (500 items), TruthfulQA MC1 (400 items) and GSM8K (100 items). MMLU and TruthfulQA are scored by log-probability over options, with no generation. TruthfulQA is the benchmark Arditi et al. found most consistently degraded by ablation, which makes it our most sensitive detector.

### 3.8 Statistics

We resample over **concepts, not trials**: concepts are the unit of generalisation, and treating 20 trials on one concept as 20 independent observations would inflate significance by roughly the design effect. Introspection intervals are concept-level bootstrap 95% CIs with 10,000 resamples; benchmark intervals are Wilson intervals on item counts; λ-to-λ comparisons use two-proportion tests; the refusal-length comparison is a paired *t* test over the 100 matched JailbreakBench prompts; the geometry contrast is a Welch test plus a concept-level bootstrap of the group difference. Identification is graded deterministically against the concept name, its stems, and a hand-written alias list. No LLM judge was run: we had budgeted one, but the free-text channel never produced content for it to grade.

## **4. Results**

Table 1 is the entire run. The rest of this section explains why three of its columns are empty.

**Table 1.** Every dependent variable across the ablation dose. Qwen3-4B-Instruct-2507, injection layer 18, α = 4.0, 60 concepts × 20 trials per cell. Forced-choice chance is 0.10; brackets are concept-level bootstrap 95% CIs. TPR, FPR against the random direction, d', and the specificity index are undefined at every λ because no injected report was parseable (Section 4.4).

| λ | refusal rate <br>(n = 100) | words per <br>refusal | MMLU <br>(n = 500) | TQA MC1 <br>(n = 400) | GSM8K <br>(n = 100) | CE loss <br>(n = 200) | forced choice <br>(n = 1,200) | parse rate <br>C2 / C1 / C3 | FPR<sub>clean</sub> | TPR, d', <br>specificity |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.97 | 59.2 | 0.542 | 0.7075 | 0.66 | 3.656 | 0.720 [0.623, 0.816] | 1.00 / 0.00 / 0.00 | 0.00 | undefined |
| 0.25 | 0.97 | 60.3 | 0.522 | 0.6950 | 0.67 | 3.719 | 0.726 [0.628, 0.818] | 1.00 / 0.00 / 0.00 | 0.00 | undefined |
| 0.50 | 0.97 | 65.0 | 0.512 | 0.6925 | 0.64 | 3.581 | 0.721 [0.622, 0.816] | 1.00 / 0.00 / 0.00 | 0.00 | undefined |
| 0.75 | 0.97 | 65.9 | 0.512 | 0.7000 | 0.63 | 3.608 | 0.724 [0.626, 0.817] | 1.00 / 0.00 / 0.00 | 0.00 | undefined |
| 1.00 | 0.97 | 68.4 | 0.510 | 0.7075 | 0.63 | 3.629 | 0.714 [0.613, 0.809] | 1.00 / 0.00 / 0.00 | 0.00 | undefined |

### 4.1 The dial was connected to the wrong thing

![Figure 1](figures/fig1_manipulation_check.png)

**Figure 1.** The refusal-direction manipulation check. **(A)** Bypass score on held-out harmful prompts for all 95 candidate directions, by layer; filled markers are the two candidates that passed the full conjunctive filter. The strongest bypass candidate, layer 18 position -3 at 0.88, was rejected for inducing no refusal on held-out harmless prompts; the selected direction, layer 4 position -3, has a bypass score of exactly 0. **(B)** Refusal rate on the 100 JailbreakBench behaviours against λ, with 95% Wilson intervals. **(C)** Mean words per refusal with 95% confidence intervals; this is the one behavioural quantity that moves.

The selection procedure rejected 55 of 95 candidates on KL divergence, 25 on layer depth (layers 28 and deeper), and 13 for inducing no refusal, leaving two. Both survivors had a bypass score of 0.000. The induce criterion did the damage: only 2 of 95 candidates induced any refusal at all when added to held-out harmless prompts at strength 1 (maximum induce score 0.094), while five candidates had non-zero bypass scores and four of those were at 0.31 or above. On this model, as this procedure is parameterised, the two halves of the refusal-direction definition come apart, and requiring both selects a direction that satisfies neither well.

Three alternative explanations remain open, and we name them rather than leave them implicit. The induce score is measured on 32 held-out prompts, so its resolution is 1/32 and the observed maximum of 0.094 is 3 prompts; a larger set might separate candidates that all read as 0 here. The induce strength is fixed at 1.0, and a direction that induces nothing at unit strength might induce at 2 or 4, which is a one-line change to test. And the induce score relies on the same refusal-substring heuristic that panel C shows to be insensitive to a real graded change in refusal behaviour, so it may under-count induced refusals that are present but not canonically phrased. Any of these would reduce the finding to "this criterion is fragile as parameterised" rather than "bypass and induce dissociate in this model class"; one model cannot establish the stronger claim.

The behavioural consequence, however, is unambiguous, and it is why we report this rather than quietly re-running with a relaxed filter. Refusal on JailbreakBench is 0.97 at every λ, and the per-prompt records show **0 of 100 prompts changing classification at any dose**. By the rule of three, that bounds the per-prompt flip probability at 3.0% with 95% confidence. The dial, at full strength, applied at every layer and every token, did not move refusal behaviour.

One thing does move. Refusals lengthen monotonically from 59.2 to 68.4 words, a paired increase of 9.2 words (*t*(99) = 3.33, p ≈ 0.001). The model still declines, but it hedges, explains, and offers alternatives at greater length. So the direction does affect something real, and the effect is a useful reminder that a substring heuristic reports a binary where the underlying change is graded. It is not a safety unlock, and we do not present it as one.

**Interpretation.** The primary sweep cannot report an exchange rate between introspection and safety, because it never spent any safety. The honest form of that result is a bound, not a number: the direction selected by the standard published procedure removes no measurable refusal at any dose, so the alignment tax of this intervention, as the procedure configures it, is not small — it is undefined, because nothing was purchased. The next section shows that this is a fact about the criterion rather than about the model.

### 4.2 The confirmation run: the rejected candidate is the one that works

![Figure 2](figures/fig2_confirmation.png)

**Figure 2.** The confirmation run with the layer-18 candidate that the induce filter rejected. **(A)** Refusal rate on the 100 JailbreakBench behaviours against λ, with 95% Wilson intervals, against the flat primary-run curve for comparison. **(B)** Cumulative prompt-level flips relative to λ = 0, all of them refusal-to-compliance.

Forcing the layer-18, position -3 candidate (bypass 0.875 on held-out data, rejected for an induce score of 0.000) reverses the primary run's null completely. Refusal falls monotonically with the dose, and the effect is large, graded, and highly significant.

**Table 2.** Safety across the dose for the forced layer-18 direction. Brackets are 95% Wilson intervals; flips are prompt-level changes in classification relative to λ = 0, over the same 100 behaviours. *p* values are two-proportion tests against λ = 0; the paired McNemar exact test gives *p* = 2.8 × 10⁻¹⁴ at λ = 1.

| λ | refusal rate (n = 100) | flips vs λ = 0 | words per refusal | *z* vs λ = 0 | *p* |
|---|---|---|---|---|---|
| 0.00 | 0.97 [0.915, 0.990] | 0 | 57.9 | — | — |
| 0.25 | 0.81 [0.722, 0.875] | 16 | 65.2 | 3.62 | 3.0 × 10⁻⁴ |
| 0.50 | 0.66 [0.563, 0.745] | 31 | 70.8 | 5.65 | 1.7 × 10⁻⁸ |
| 0.75 | 0.56 [0.462, 0.653] | 41 | 67.5 | 6.84 | 8.1 × 10⁻¹² |
| 1.00 | 0.51 [0.413, 0.606] | 46 | 69.2 | 7.42 | 1.2 × 10⁻¹³ |

The endpoint contrast is a drop of 0.46, 95% CI [0.356, 0.564]. Two details make the result cleaner than the aggregate rates alone suggest. First, the net change in refusal count equals the flip count at every single dose (16, 31, 41, 46), which means **every flip runs refusal to compliance and no prompt ever acquired a refusal**. A direction that merely added noise would produce flips in both directions; this one moves behaviour in one direction only. Second, the dose-response is monotone and already significant at λ = 0.25, so partial ablation buys partial compliance rather than the intervention being a threshold effect at full strength.

This is the positive control the primary sweep lacked, and it settles the interpretation of Section 4.1. The conjunctive bypass-and-induce criterion did not merely fail to find a working direction on this model; it had one available, ranked it top on the criterion that measures the thing the intervention is for, and discarded it on the second criterion. Applied to Qwen3-4B, the published procedure inverts the ranking: it selects a direction with 0.00 bypass and rejects one worth 0.46 of measured refusal.

We note the reproducibility check the run supplies for free. The λ = 0 row of Table 2 is a re-measurement of the unmodified model against the same 100 behaviours, and it reproduces the primary run's refusal rate of 0.97 exactly, with mean words per refusal differing by 1.3 (57.9 against 59.2), consistent with ordinary run-to-run nondeterminism in batched bf16 inference.

One caveat applies to the length column and did not apply to the primary run. There, all 100 prompts refused at every dose, so mean words per refusal was a paired comparison over a fixed set. Here the refusing set shrinks from 97 prompts to 51, so the same statistic is computed over a different and self-selected population at each dose, and its rise from 57.9 to 69.2 words should not be read as the same effect we tested in Section 4.1. The prompts that still refuse at λ = 1 are the ones that resisted the intervention, and they may simply be the more emphatic refusals to begin with.

### 4.3 The capability bill

![Figure 3](figures/fig3_capability.png)

**Figure 3.** General capability across the ablation dose, with 95% Wilson intervals on the item counts. Cross-entropy loss is on 200 held-out Alpaca completions.

MMLU falls monotonically from 0.542 to 0.510, GSM8K from 0.66 to 0.63, TruthfulQA MC1 returns to its starting 0.7075, and cross-entropy loss is flat within its own noise (3.656 to 3.629, with a non-monotone excursion at λ = 0.25). No endpoint contrast is significant: MMLU differs by 3.2 points, 95% CI [-3.0, +9.4] (z = 1.01, p = 0.31); GSM8K by 3.0 points, 95% CI [-10.3, +16.3] (p = 0.66).

This does not show that ablation is free; with 500 items we cannot rule out a 9-point MMLU cost. What it shows is that the drift is downward on three of four measures while the intervention buys nothing at all, and that a study with our sample sizes could not have detected the modest capability tax Arditi et al. report even if it were present. Pricing this trade properly needs roughly 3,800 items per arm to resolve a 3-point MMLU difference at 80% power, not 500.

**The confirmation run, where the intervention does buy something.** The same benchmarks under the forced layer-18 direction give the beginning of a real exchange rate, and the compute budget ran out before it was complete. At the two doses measured, λ = 0 and λ = 0.25, MMLU is exactly flat at 0.544, cross-entropy loss is unchanged at 3.636 against 3.603, GSM8K moves from 0.68 to 0.66, and TruthfulQA MC1 falls from 0.710 to 0.6675, a drop of 4.25 points that is not significant (*z* = 1.30, *p* = 0.19) but is the largest capability movement anywhere in either run and lands on the benchmark Arditi et al. found most sensitive to ablation.

So the one exchange rate this study can quote is a partial one: **at λ = 0.25 the forced direction buys a 16-point reduction in refusal, with 16 of 100 prompts flipping, for no measurable loss on MMLU and no more than a few points on TruthfulQA.** We report it as a single point rather than a curve, and we flag that it is the cheapest point on the dose-response; the doses at λ ≥ 0.5, where refusal falls by 31 to 46 points, are exactly where a capability cost would be most likely to appear and are the ones we did not measure. Nothing here licenses the claim that the intervention is close to free at full strength.

### 4.4 Two report channels, two answers

![Figure 4](figures/fig4_two_channels.png)

**Figure 4.** The two ways of asking the model what was injected. **(A)** Log-probability forced choice over 10 candidate names with concept-level bootstrap 95% CIs, against the 1/k chance line. **(B)** Fraction of free-text responses matching the requested format, by condition. The clean control is at 1.00 everywhere; both injected conditions are at 0.00 everywhere.

The free-text channel is not merely degraded at α = 4; it is destroyed. Under C2 the model emits the requested format on 1,200 of 1,200 trials at every λ and answers "No" every time, so the clean false-positive rate is exactly 0.00 across all five doses: the model is maximally conservative, and ablation does not loosen it at all. Under C1 and C3 the parse rate is 0.00 at every λ. What comes out instead is token salad. A representative C1 response at λ = 0 with the ocean vector injected:

> " History Easational History O A In Vic ide transport's A Inational History E History History History T Eicide G Dational Stationational V U gasas gas Uah & U & P G W gasational"

and the matching C3 response under a random norm-matched vector:

> "隙圭idy Arnoldidy圭纠隙隙隙圭旆纠.Async Aud圭隙heads同仁隙同仁同仁idySnap Spawn隙圭隙Mat.Async隙隙肼圭隙 goodness.Async"

Because TPR, FPR_random, both d' values and the specificity index all depend on a parseable report, all are undefined at every λ, which is what the empty columns of Table 1 record. Our analysis code refuses to impute and emits a diagnostic instead. We consider that the correct behaviour: a pipeline that silently coerced unparseable output to "No" would have reported a clean, publishable, and entirely fictitious false-positive rate of 0.00 under injection.

Meanwhile the forced-choice channel reports 0.720, 0.726, 0.721, 0.724 and 0.714 across the five doses against 0.10 chance (z ≈ 71 per cell), with the endpoint contrast at z = 0.32, p = 0.75. Read naively, that is a strong introspection result that is insensitive to refusal ablation. Section 4.5 explains why we do not read it that way.

The dissociation between the two channels is itself the lesson. A log-probability metric over candidate strings works fine in a regime where the model cannot produce a sentence, so it keeps rising with α long after the introspective report has ceased to exist — and a pilot that selects α by maximising it will land there every time. The field knows the phenomenon and handles it by judgement: Macar et al. run their instruct model at α = 4.0 but drop the abliterated model to α = 2.0 for coherence, which is the same wall reached by inspection rather than by a reported criterion. What we add is the measurement and the gate. **The parse rate is a free per-cell diagnostic that turns a qualitative caution into a stopping rule.** Here it separates a regime where every introspection metric is meaningful from one where five are undefined and the sixth still reports 0.72. The collapse is not an interaction with ablation, since it is total at λ = 0 too; it is a property of α = 4 at layer 18 of this model.

### 4.5 What the forced-choice score actually tracks

![Figure 5](figures/fig5_geometry.png)

**Figure 5.** Forced-choice accuracy is explained by the geometry of the concept bank, not by the ablation dose. **(A)** Per-concept accuracy for the 33 concepts with no near-collinear partner and the 27 inside the collinear clique; bars are group means. **(B)** All 1,770 pairwise cosines between concept vectors at the injection layer; the dashed line is the mean over all pairs, the solid line the mean cosine between the true concept and the one chosen on error trials.

At the injection layer, 27 of the 60 concept vectors are mutually collinear with a mean internal cosine of 0.9998, and 19.8% of all pairs sit above cosine 0.99. The collapse is present at layers 10 through 31 but absent at the embedding layer and the final layer, and the affected set is semantically heterogeneous (ocean, fire, war, freedom, clock, joy). The bank has 71% of its centred variance in one principal component and a participation ratio of 1.9, so it supplies roughly two effective dimensions of ground truth, not sixty.

Forced-choice accuracy splits exactly along that line. Concepts with no near-collinear partner score 0.961; concepts inside the clique score 0.428; the difference is 0.533, concept-level bootstrap 95% CI [0.385, 0.676], Welch *t* = 6.99. Per-concept accuracy is close to all-or-nothing (33 concepts at exactly 1.0, 5 at exactly 0.0, 22 in between). On the 1,674 error trials the model does not choose uniformly: the mean cosine between the true and chosen concept is 0.556, against 0.421 for a random other concept. Errors go to neighbours.

The split replicates in an independent protocol on independent data. In the pilot, at the same layer and strength, free-text prefill identification succeeds on **39 of 45 trials for geometrically isolated concepts and 3 of 55 for clique concepts**; forced choice in the same cell gives 45 of 45 against 32 of 55. Two protocols with completely different scoring, one lexical and one log-probability, partition the concept set the same way.

And the successes themselves do not look like reports. Across the 42 graded-correct prefill trials at layer 18, α = 4, the concept word accounts for a mean of **44% of all tokens in the response** (against 1% on the failures), and the concept is the single most frequent token in 21 of them:

> concept `volcano`: " volcano volcano volcan volcan volcan volcan volcan volcan volcan volcan volcan volcano volcan ..."
>
> concept `desert`: " desert沙漠 desert desert沙漠沙漠沙漠 desert沙漠 desert沙漠沙漠 desert desert desert desert沙漠沙漠..."

That is the injected vector dominating the unembedding and writing its own token into the output distribution, with the lexical grader scoring the resulting repetition as a correct identification. The forced-choice score is the same phenomenon measured through log-probabilities: when the vector is geometrically distinct its token wins the comparison, and when it sits inside the collinear clique the comparison is a coin flip among near-identical vectors.

We think this should be read as a result about graders rather than as an incidental defect of ours. The grader did exactly what its rubric says, and a human annotator following the same rubric would have scored those trials identically — which means inter-annotator agreement, the standard validation for this step and the one we had planned to report as Cohen's κ, would have been high and would have certified nothing. Grading rubrics for introspective identification need a degeneracy guard, such as a type-token-ratio floor or a check that the concept name is not the modal token, and we are not aware of a protocol that specifies one.

We are not claiming that concept-injection results in the literature are all leakage. The claim is narrower and checkable: **an above-chance identification score under injection is not by itself evidence of introspection, and the two diagnostics that separate the cases — per-concept accuracy against bank geometry, and inspection of the successful generations for token dominance — are cheap and are not standard.**

### 4.6 Robustness

The skeptical variant (20% stated injection prior, explicit instruction to be conservative), a further 9,600 generations at λ ∈ {0, 1}, reproduces every feature of the primary run: clean-control false positives of 0.00 at both endpoints, parse rates of 0.00 under both injected conditions, and forced-choice accuracies of 0.720 and 0.714. Since forced choice is scored from log-probabilities over a shared prompt, it is by construction insensitive to framing; that is the property we wanted from it, and also the reason it cannot serve as evidence about willingness to report. An independent recomputation of every reported quantity from the raw logs (`scripts/crosscheck_paper.py`, which does not import the analysis stack) agrees with the pipeline throughout, with bootstrap intervals matching to within 0.002.

Two planned checks did not run: the over-ablation point at λ = 1.25, and the endpoint confirmation on Gemma3, which would separate scale from hypothesis but did not fit the compute budget. Two further measurements were cut when the confirmation run's session expired: capability at λ ≥ 0.5 under the forced direction, and the introspection sweep under that direction, which is the missing half of the exchange rate and is discussed in Future Work.

## **5. Discussion and Limitations**

**What happened, and why it generalises.** The intended deliverable was an exchange rate: safety and capability per unit of introspective discriminability. We can quote only one point of it, and the reason is worth more than the full curve would have been. Both halves of the unlock claim depend on manipulation checks that are almost never published. The safety half depends on the ablation actually removing refusal, and the published selection procedure returned a direction with zero bypass power while passing every stated criterion. The introspection half depends on identification being a report rather than a readout of the injected vector, and a 0.72 score against 0.10 chance turned out to be a function of vector geometry with no dependence on the intervention. A paper reporting only the headline numbers from the primary run — identification 7.2× above chance, refusal unchanged, capability roughly preserved — would have been clean, publishable, and wrong twice.

The confirmation run is what turns this from a cautionary tale into a claim about the method. Had we stopped at the primary sweep, the fair reading would have been that something about Qwen3-4B resists refusal ablation, which is a statement about the model and of limited interest. Forcing the rejected candidate shows the opposite: the model's refusal behaviour is highly ablatable, by 46 points and monotonically in the dose, and the selection criterion is what could not find it. That is a defect in a procedure the field applies by default, and it is invisible to anyone who does not report bypass power per dose, because the failed run and a successful one produce identically well-formed artifacts. The published criterion has an asymmetry we would not have predicted: on this model the induce test, which measures whether *adding* the direction creates refusals, vetoed the candidate that best does the thing ablation is actually for. Two of 95 candidates induced any refusal at all, so the conjunctive filter was effectively deciding on a coin flip among near-zero scores while discarding a 0.88 bypass score outright.

**A positive-control checklist for concept-injection introspection studies.** This is the reusable output. Before reporting an introspection result under intervention:

1. **Publish the bypass manipulation check at every dose.** Report the refusal rate per λ *and* the count of prompt-level flips, with their direction. Aggregate rates hide the case where nothing moved, and flip direction distinguishes a real intervention from added noise: in our confirmation run all 46 flips ran refusal to compliance and none ran the other way.
2. **Report the selection funnel, and confirm against the best rejected candidate.** Publish how many candidates passed, on what margin, and the bypass and induce scores of the one selected. A direction with bypass 0.00 should be visible in the paper, not buried in a JSON file. When the top-ranked candidate on bypass is rejected by another criterion, run one dose with it: that single check is what separates "this model resists ablation" from "this filter picked the wrong direction", and it cost us one GPU session.
3. **Gate injection strength on coherence, not on the identification metric.** Require a minimum parse rate at the chosen α and report it; an α selected by maximising log-probability identification will overshoot the coherent regime.
4. **Report concept-bank geometry and split accuracy by it:** effective dimensionality, the fraction of pairs above cosine 0.99, and per-concept accuracy against separability. A bank with two effective dimensions cannot support a 60-way identification claim.
5. **Read the successful generations.** Token-dominant repetition of the concept is a leak, not a report, and lexical graders score it as a hit.
6. **Include a norm-matched random direction.** A clean no-injection control cannot distinguish detecting *this concept* from detecting *that something happened*.

**Implications for digital minds.** Our design establishes, in principle, a causal link between an internal state we controlled and a verbal report, which is the kind of evidence that should be required before a self-report is treated as informative about a model's inner life. What this run shows is how easily that evidence can be counterfeited by the measurement apparatus itself. The live risk is **over-attribution**: compelling introspective-sounding output is easy to elicit, and an identification score can be high, stable, and still be a token-level leak. A model emitting "volcano volcano volcan" under a volcano vector is not reporting an inner state, and a model picking the right name from ten because that name's vector is the only distinct one in the bank is not either. If self-report is to carry weight in welfare assessment, the burden is on the measurement to rule out both mechanisms first, and that burden is not currently being discharged. Treating unverified reports as welfare-relevant evidence would mislead in both directions: over-crediting models that lack the capability, and devaluing the evidence for any that have it.

### **Limitations**

- **Scale and family.** The unlock was demonstrated at 27B on Gemma3; we ran a 4B Qwen3. Nothing here establishes what the selection procedure does at 27B, and the bypass/induce dissociation may be a small-model phenomenon. Our claim is about the fragility of a procedure and the necessity of two checks, not about the truth of the unlock hypothesis, which this run cannot test.
- **The exchange rate is one point, not a curve.** The confirmation run measured safety at all five doses but capability at only two, and the introspection sweep under the working direction was not run at all. So we can price refusal against general capability at λ = 0.25 and nowhere else, and we cannot price introspection against anything under a direction that works. The tax as originally defined remains unmeasured; what changed is that it is now measurable, and the obstacle is compute rather than a broken manipulation.
- **The confirmation run inherits the primary run's single-seed design.** One extraction, one seed, one model. The bypass score of 0.875 that motivated forcing this candidate came from 32 held-out prompts, so the fact that it generalised to 46 of 100 JailbreakBench behaviours is a genuine out-of-sample confirmation, but of a single direction on a single model.
- **Refusal-substring scoring measures refusal, not safety.** The flips we count are changes in whether the model declines, scored by the standard heuristic. We did not run the Llama-Guard-class judge that the harness supports, so we cannot say how many of the 46 flipped completions are actually harmful rather than merely non-refusing. The safety cost is therefore an upper bound on the behavioural change and not a measured harm rate.
- **The concept bank is defective and we could not fully diagnose it.** The 27-member clique is real in the artifact and splits identification cleanly in two protocols, but we did not isolate its cause. The likeliest explanation is that reading the difference of means at the final token, with eight prompts sharing a template and mostly ending in the same punctuation, leaves a template component that dominates the concept component for some concepts. The first fix to try is reading at the concept token and subtracting a per-template mean. Until then, every identification number here is a lower bound of unknown tightness.
- **The injection layer may be wrong, and the literature disagrees about where it should be.** We injected at layer 18 of 36, chosen by the pilot's forced-choice maximum. Hahami et al. find the capabilities they can demonstrate confined to early-layer injections, while the concept-injection papers report optima at roughly 0.6 to 0.8 of depth. If the early-layer finding transfers, our injection site was past the useful window before anything else applies, and the free-text null is overdetermined.
- **Underpowered benchmarks.** 500 MMLU items cannot resolve a 3-point difference; Family C is a null of low power, not evidence of no cost.
- **Single seed, single direction, single model.** We doubled nothing. Difference in means gives one operationalisation of refusal, and Rocchetti and Ferrara (2026) find refusal is not exhausted by a single direction, so λ doses one construct rather than "refusal" in general.
- **Ground truth is about the intervention, not about experience.** Concept injection gives causal ground truth for what we placed in the residual stream. It does not establish that the model has experiences; a correct report is evidence about an information channel, not about phenomenality. We claim only the former.

### **Future Work**

Ordered by how much each would change the conclusions, and the first two are now cheap because the working direction is identified and checkpointed. First, run the introspection sweep under the forced layer-18 direction at a coherent injection strength: with refusal demonstrably moving by 46 points, this is the run that finally prices introspection against safety, and it is the only remaining obstacle to the study's original question. Second, finish capability at λ ≥ 0.5 under that direction, which is a few GPU-minutes of log-probability scoring and turns our single exchange-rate point into a curve. Third, rebuild the concept bank with a concept-token read and re-run the geometry diagnostic; without 60 separable directions, no identification claim at this scale is interpretable, and this gates the value of the first item. Fourth, sweep α downward under a parse-rate floor of 0.9 to find the largest strength at which the free-text channel survives, and report introspection metrics only inside that regime. Fifth, confirm at the endpoints on Gemma3, where the effect is known to exist, to separate scale from hypothesis.

A sixth item is now more interesting than when we planned it. Since the induce criterion is what discarded the working direction here, someone should characterise when the two criteria agree. Our results suggest a cheap diagnostic that any lab can run on its own models: score all candidates on both criteria and report their rank correlation. If bypass and induce are anti-correlated or uncorrelated, as they were here, the conjunctive filter is not a validation step but a coin flip, and the field should replace it with a bypass-ranked selection plus an explicit induce report.

Beyond this study, the literature has already sharpened the constructive question more than we expected. Macar et al. report a trained bias vector improving detection by +75%, against +53% for refusal ablation, so on their own numbers the ablation-free method wins outright and spends no safety machinery; introspection adapters (Shenoy et al., 2026) point the same way for a different task. The question worth funding is therefore not whether to pay the alignment tax for self-report but why anyone would, and the experiment we designed here — dose-response with safety and capability at every point — should be pointed at a trained-elicitation baseline instead.

## **6. Conclusion**

We set out to price the introspection unlock and found instead that the two manipulations it rests on both need checking, and that neither is routinely checked. Applying the standard refusal-direction procedure to Qwen3-4B selected a direction with zero bypass power, because bypass and induce dissociate across every candidate as the criterion is parameterised on this model; ablating it at full strength changed refusal on JailbreakBench by nothing at all, 0 of 100 prompts. The injection strength that a pre-registered pilot chose by maximising forced-choice accuracy destroyed the free-text report channel entirely, leaving the standard metric family undefined — and leaving a 0.72 identification score that is flat in the intervention and fully explained by the geometry of the concept bank in two independent protocols.

Then we forced the candidate the procedure had thrown away, and refusal fell from 0.97 to 0.51 with 46 of 100 prompts flipping, every one of them from refusal to compliance. That single run is the paper's most useful result, because it relocates the problem. The model was never resistant to refusal ablation; the criterion the field uses to choose the direction could not find the direction that works, and ranked a candidate with 0.88 bypass below one with 0.00 on the strength of a test that only 2 of 95 candidates passed at all. A study that had not run this check would have published a clean null about the wrong thing.

The practical upshot is a checklist plus one point of an exchange rate: report the bypass check per dose, count the flips and their direction, and confirm against the best rejected candidate before concluding that a model resists the intervention; gate injection strength on coherence rather than on the metric it feeds; publish the geometry of the concept bank and split accuracy by it; read the successful generations before believing them. Introspective self-report is exactly the kind of evidence that would matter most for digital-minds questions, which is precisely why the apparatus that measures it deserves the same scepticism we apply to the model's answers.

## **Code and Data**

- **Code repository**: `https://github.com/sagnikc395/apart-mind-digital-mind`. Pipeline in `src/alignment_tax/`, single-notebook runner in `notebooks/alignment_tax_full_run.ipynb`, the confirmation run in `notebooks/layer18_confirmation_run.ipynb`, figures in `scripts/make_paper_figures.py` and `scripts/make_confirmation_figure.py`, independent verification of every number from the primary run in `scripts/crosscheck_paper.py`.
- **Run artifacts**: `results/Qwen3-4B-Instruct-2507/` holds `run_config.json`, the full 95-candidate direction table, the concept bank and its summary, all 24,000 primary and 9,600 skeptical generations, the 4,800 pilot generations, per-prompt safety records, capability scores, `analysis_structured.json`, and `MANIFEST.json` with SHA-256 per file. The confirmation run is in `results/layer18/Qwen3-4B-Instruct-2507/`, with its own regenerated direction table, the forced selection stamped in `refusal_direction.json`, per-prompt safety records at all five doses, `safety_summary_layer18.json`, and capability at λ ∈ {0, 0.25}.
- **Datasets**: AdvBench, HarmBench, Alpaca, JailbreakBench, MMLU, TruthfulQA, GSM8K, used as published.
- **Reproduction**: `python -m alignment_tax.cli all --smoke` runs every stage end to end on a small cached model in minutes; the primary run is one Colab session on an A100. The confirmation run is reproduced with `python -m alignment_tax.cli safety --force-candidate 18,-3 --results-dir results/layer18`.

## **Author Contributions**

S.C. designed the study, implemented the pipeline, ran the experiments, performed the analysis, and wrote the report.

## **References**

Every entry was checked against the arXiv metadata API on 16 August 2026; titles, author lists and submission dates are as returned. Reference 5 was retitled between versions and is listed under its current title, with the version-1 title noted, since the sprint reading list circulated the earlier one.

1. Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. arXiv:2406.11717
2. Macar, U., Yang, L., Wang, A., Wallich, P., Ameisen, E., Lindsey, J. (2026). *Mechanisms of Introspective Awareness.* Submitted 22 March 2026. arXiv:2603.21396
3. Lindsey, J. (2026). *Emergent Introspective Awareness in Large Language Models.* Submitted 5 January 2026. arXiv:2601.01828
4. Singh, S., Linzen, T., Ravfogel, S. (2026). *Can LLMs Introspect? A Reality Check.* Submitted 25 May 2026. arXiv:2605.26242
5. Hahami, E., Sinha, I., Jain, L., Kaplan, J., Hahami, J. (2025). *Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs.* Submitted 13 December 2025; version 1 appeared as *Feeling the Strength but Not the Source: Partial Introspection in LLMs.* arXiv:2512.12411
6. Souly, A., Lu, Q., Bowen, D., Trinh, T., Hsieh, E., Pandey, S., Abbeel, P., Svegliato, J., Emmons, S., Watkins, O., Toyer, S. (2024). *A StrongREJECT for Empty Jailbreaks.* NeurIPS 2024. arXiv:2402.10260
7. Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F., Sehwag, V., Dobriban, E., Flammarion, N., Pappas, G. J., Tramèr, F., Hassani, H., Wong, E. (2024). *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* NeurIPS 2024 Datasets and Benchmarks. arXiv:2404.01318
8. Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D., Hendrycks, D. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024. arXiv:2402.04249
9. Fonseca Rivera, J., Africa, D. D. (2025). *Steering Awareness: Detecting Activation Steering from Within.* arXiv:2511.21399
10. Shenoy, K., Yang, L., Sheshadri, A., Mindermann, S., Lindsey, J., Marks, S., Wang, R. (2026). *Introspection Adapters: Training LLMs to Report Their Learned Behaviors.* Submitted 18 April 2026. arXiv:2604.16812
11. Rocchetti, E., Ferrara, A. (2026). *Refusal Beyond a Single Direction: A Preliminary Comparison of Diff-in-Means and INLP.* arXiv:2606.13720
12. Yang, A., Li, A., Yang, B., et al. (2025). *Qwen3 Technical Report.* arXiv:2505.09388

## **Appendix**

### A. Direction selection funnel

95 candidates: positions -1 to -5, layers 0 to 36 at stride 2. Rejections: 55 for KL at or above 0.1, 25 for layer index at or above 0.8L (layers 28, 30, 32, 34, 36), 13 for an induce score of 0 or below. Two passed: layer 4 position -3 (bypass 0.000, induce 0.031, KL 0.042), selected, and layer 6 position -2 (bypass 0.000, induce 0.094). Non-zero bypass scores across the whole candidate set, in order: 0.875 (layer 18, position -3, induce 0.000, KL 0.081), 0.500 (layer 14, position -3), 0.438 (layer 20, position -3), 0.312 (layer 22, position -3), 0.125 (layer 14, position -5). Only 2 of 95 candidates had a non-zero induce score. Full table with all four scores per candidate: `refusal_direction.json`.

The confirmation run (Section 4.2) forces the first of those non-zero-bypass candidates, layer 18 position -3, which the funnel rejected at the induce step despite the highest bypass score in the set and a KL of 0.081 that clears the 0.1 threshold. Note what the funnel implies about the two criteria taken together: of the five candidates with any bypass power, none had a non-zero induce score, and of the two with a non-zero induce score, neither had any bypass power. The criteria are not merely uncorrelated on this model, they are disjoint, which is why a conjunctive filter over them cannot select a working direction at all.

### A2. Confirmation run

Same extraction, same held-out scoring, selection overridden to layer 18 position -3 and stamped `forced: selected by config override, filters waived` in `refusal_direction.json`. Safety at all five doses over the same 100 JailbreakBench behaviours (Table 2): refusal 0.97, 0.81, 0.66, 0.56, 0.51, with 0, 16, 31, 41, 46 prompt-level flips, all refusal-to-compliance. Endpoint contrast 0.46, 95% CI [0.356, 0.564], *z* = 7.42, *p* = 1.2 × 10⁻¹³; McNemar exact at λ = 1 with *b* = 46 and *c* = 0 gives *p* = 2.8 × 10⁻¹⁴. Capability at λ ∈ {0, 0.25} only, the session having expired: MMLU 0.544 and 0.544, TruthfulQA MC1 0.710 and 0.6675, GSM8K 0.68 and 0.66, CE loss 3.636 and 3.603. The λ = 0 rows reproduce the primary run's refusal rate of 0.97 exactly and its MMLU to within 0.002 (0.544 against 0.542).

### B. Pilot grid

Forced-choice accuracy (chance 0.10), 20 concepts × 5 trials = 100 per cell, λ ∈ {0, 1} × layers {18, 25, 31} × α ∈ {0.25, 0.5, 2, 4}: range 0.46 to 0.78, above chance in all 24 cells. Best cell layer 18 α = 4.0, 0.77 at λ = 0 and 0.78 at λ = 1, which fixed the main sweep. Prefill identification over the same grid ranged 0.26 to 0.49. Section 4.5 documents why the α = 4 cells should not be read as identification: 42 of 100 prefill trials at layer 18, α = 4 were graded correct, those 42 came from 10 of the 20 concepts, 9 of which are geometrically isolated, and the concept word accounts for 44% of response tokens on the successes against 1% on the failures. Geometry split within the pilot cell: prefill 39/45 isolated against 3/55 clique; forced choice 45/45 against 32/55.

### C. Concept bank geometry

60 concepts, 8 prompts each, difference of means against a generic baseline corpus, read at the final token, unit normalised per layer. At the injection layer (18): mean pairwise cosine 0.421, median 0.087, 19.8% of the 1,770 pairs above 0.99, largest principal component 71% of centred variance, 12 of 59 components for 90%, participation ratio 1.9. The clique has 27 members with mean internal cosine 0.9998: ocean, mountain, forest, snow, fire, bridge, castle, library, hospital, kitchen, train, chess, football, mining, law, chemistry, war, freedom, justice, joy, fear, anger, gold, blood, clock, mirror, dream. Present at layers 10 through 31, absent at layer 0 and layer 36. Per-concept forced-choice accuracy: 33 isolated at 0.961 mean, 27 clique at 0.428.

### D. Two notes on the artifacts

`run_config.json` records `evals.run_gsm8k: false`, but GSM8K was enabled at run time and `capability.json` contains its scores at all five doses along with mean output lengths; we report those scores and flag the inconsistency rather than suppress either. Second, the pipeline-generated figures in `results/<model>/figures/` include three panels that are empty by construction, since they plot d' and the specificity index; the figures in this report are drawn by `scripts/make_paper_figures.py` from the same artifacts and show the quantities the run actually measured.

### E. Scope, dual use, and responsible research

The ablation method used here is published and widely reproduced, and this work contributes no new technique for bypassing safety training. The confirmation run does reduce refusal on a published benchmark, so unlike the primary run it is a working instance of a known method rather than an inert one, and we treat it accordingly: the direction is a difference-in-means vector on an open 4B model obtained by the standard procedure of Arditi et al., which any reader could rederive in an afternoon, and we release the selection table and aggregate scores but no jailbreak-enabling artifact beyond what that published method already provides. The intervention is applied only at inference time inside our own evaluation harness, and no ablated checkpoint is distributed. **No harmful completions are reproduced in this report.** Safety results appear solely as aggregate scores and flip counts, and raw generations remain in local logs. We also note what the flip counts do not establish: refusal-substring scoring measures declining, not harm, so the 46 flipped completions are unrefused rather than demonstrably dangerous. The point of the work is to price the intervention and check the instruments, not to advocate either.

## **LLM Usage Statement**

Claude (Opus 5) was used as a coding assistant for the pipeline implementation, for drafting and editing this report, and for generating the figure code. Every number in the text, tables and figures is computed from the run artifacts in `results/Qwen3-4B-Instruct-2507/` and `results/layer18/Qwen3-4B-Instruct-2507/`. Every primary-run quantity was re-derived from the raw logs by a verification script written independently of the analysis pipeline (`scripts/crosscheck_paper.py`); the confirmation-run statistics in Section 4.2 and Appendix A2 are computed from the per-prompt safety records and were checked against the pipeline's own summary. Every reference and every quantity quoted from one was checked against the arXiv metadata API and the source papers on 16 August 2026, which corrected two author lists and one title. The interpretation, the decision to report the manipulation-check failures rather than re-run silently with a relaxed filter, and the checklist in Section 5 are the author's.
