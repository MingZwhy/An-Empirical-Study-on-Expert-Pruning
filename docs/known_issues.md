# Known issues / open questions

Items verified to be real but **deliberately not changed**, because they are experimental
design decisions rather than defects. Revisit before publishing cross-model comparisons.

## DiEP: the k=2 skipping rule does not carry over to top-8 routing

Status: **open, deferred until the Qwen3-30B sweep finishes.** Found 2026-08-03.

Paper: *DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning*
(NeurIPS 2025, [arXiv:2509.16105](https://arxiv.org/abs/2509.16105)).

DiEP scores 8 to 21 points below plain Fixed-K at a matched average expert count on
Qwen3-30B, and the gap widens monotonically with beta:

| configuration | avg experts | suite mean | Fixed-K at the same budget | gap |
| --- | --- | --- | --- | --- |
| DiEP beta=0.5 | 5.95 | 65.87 | 74.09 (k=6) | −8.22 |
| DiEP beta=0.7 | 4.84 | 59.17 | 72.81 (k=5) | −13.64 |
| DiEP beta=1.0 | 3.80 | 47.51 | 68.26 (k=4) | −20.75 |

The Fixed-K column is the reference the report uses, which is the lower of the old table's value
and the re-run here: k=6 and k=5 re-ran higher (74.79 and 73.67) and keep the old value, k=4
re-ran lower and uses it.

Ban and Dynamic Routing stay within about 2 points of Fixed-K at the same budgets and are often
slightly above it, so the budget itself is not what breaks DiEP.

**The collapse is DiEP's `gamma_2`, not thresholding in general.** NAEE has the same threshold
form — every rank compared against `w_e0 * beta` — and differs from DiEP only by the `gamma_2`
factor. Two of its re-runs have now landed, and it costs a fraction of what DiEP does:

| configuration | avg experts | suite mean | Fixed-K at the same budget | gap |
| --- | --- | --- | --- | --- |
| NAEE beta=0.45 | 5.13 | 71.29 | 72.81 (k=5) | −1.52 |
| NAEE beta=0.55 | 4.08 | 65.67 | 68.26 (k=4) | −2.59 |

So thresholding against the top-1 gate weight is worth 1.5 to 2.6 points at these budgets, a
mild and stable penalty. Multiplying that threshold by `gamma_2` turns the same rule into a
14–21 point loss, and the damage grows with beta while NAEE's does not. That isolates the
`gamma` generalisation as the cause and makes the two corrections below worth running.

**That conclusion is for `k_min=2`. At `k_min=1` on Ling-lite both rules collapse together**, so
the Ling avg-2 row says nothing about `gamma_2`:

| configuration | avg routed experts | suite mean | note |
| --- | --- | --- | --- |
| Fixed-K k=2 | 2.00 | 33.47 | reference, the old table's value: the re-run here scored 34.50 |
| Ban lambda=0.35, k_min=1 | 1.99 | 32.37 | dynamic k from sensitivity, not a threshold |
| Dynamic Routing thr=0.4 | 2.04 | 31.07 | cumulative mass, no k_min |
| NAEE beta=0.75, k_min=1 | 2.09 | **9.97** | threshold against `w_e0` |
| DiEP beta=0.8, k_min=1 | 2.26 | **9.34** | same, times `gamma_2` |

The two threshold rules lose two thirds of the suite at a budget the other three survive, and
they are the only two configured with `k_min=1` — the compact table's own choice for this row.
Ban is also at `k_min=1` but does not set its count by comparing weights, so it does not
concentrate mass on single-expert tokens the way a relative-weight bar does at beta=0.75 with
only 6 routed experts.

**Measured: the floor is the whole story.** Both rows were repeated with beta untouched and only
`k_min` moved from 1 to 2, so the single-expert token is the only thing that changes:

| configuration | avg routed experts | suite mean | Fixed-K at the same average |
| --- | --- | --- | --- |
| NAEE beta=0.75, k_min=1 | 2.09 | 9.97 | 35.1 |
| NAEE beta=0.75, k_min=2 | 2.43 | **39.85** | 41.2 |
| DiEP beta=0.8, k_min=1 | 2.26 | 9.34 | 38.2 |
| DiEP beta=0.8, k_min=2 | 2.57 | **41.82** | 43.8 |

The Fixed-K column interpolates between the k=2 and k=3 re-runs here (33.47 at 2.00 and 51.49 at
3.00). Raising the floor is worth about 30 points for roughly a third of an expert, which no
budget argument can explain: at `k_min=2` both rules land 1 to 2 points under the Fixed-K line,
which is where the other dynamic methods sit, while at `k_min=1` they are 25 to 29 points under
it. So on Ling-lite nothing was wrong with comparing weights against `w_e0`, and `gamma_2` makes
no difference either — NAEE and DiEP move together at both floors. A router with only six experts
cannot afford to drop to one, and the compact table's choice of `k_min=1` for this row is what
produced the collapse.

This does not carry over to the Qwen3-30B finding above, which was measured entirely at
`k_min=2`: there DiEP loses 8 to 27 points where NAEE at the same budget loses 2 to 3, and that
gap is still `gamma_2`.

### Why the old table never showed this

The concern that the old table did not show DiEP as unusually weak is right, and the reason is
that the datasets DiEP fails on were not being evaluated. On Ling-lite the columns aime24,
aime25, lcb, hellaswag and livebench are byte-identical across every beta setting of both NAEE
and DiEP — five of nine datasets that do not move at all as the pruning strength doubles. On
Qwen3-30B the same columns take only two or three distinct values across the whole sweep: from
beta=0.6 to beta=1.2, DiEP's aime24 is pinned at 72.60, aime25 at 52.81 and lcb at 41.14.

Comparing the old row against our re-run of the identical configuration (DiEP beta=0.7,
k_min=2) isolates it:

| dataset | old table | re-run | delta |
| --- | --- | --- | --- |
| mmlu_pro | 69.71 | 70.2 | +0.5 |
| gpqa | 47.47 | 46.0 | −1.5 |
| math_500 | 80.40 | 81.8 | +1.4 |
| gsm8k | 93.03 | 92.7 | −0.3 |
| aime24 | 72.60 | 35.3 | **−37.3** |
| aime25 | 52.81 | 26.5 | **−26.3** |
| lcb | 41.14 | 14.9 | **−26.2** |
| hellaswag | 85.34 | 83.3 | −2.0 |
| livebench | 90.67 | 82.0 | −8.7 |

The four short-answer datasets agree to within 1.5 points, so those were genuinely evaluated
back then; the entire 11-point drop in the mean comes from the long-generation ones that were
replayed from cache. This also says something about the method rather than the bug: DiEP's
damage is concentrated in long chain-of-thought tasks, where over-pruning makes the model
ramble into the 32768-token cap and score zero, while short-answer accuracy holds up
comparatively well (mmlu_pro 70.2 against Fixed-K k=5's 74.2). Ban reaches aime24 42.9 at
avg 3.14, so the budget itself is not what makes those tasks impossible.

### What the paper specifies

Section 4.3 skips expert `e1` when `w_e1 < gamma * w_e0`, with `gamma = gamma_1 * gamma_2`
(eq. 12), where `gamma_1` is the per-layer median of `w_e1/w_e0` over calibration data and
`gamma_2 = CKA(y_e0, y_e1)` divided by the layer's mean pairwise CKA. It states "we assume
k = 2 for simplicity" and never gives a rule for k > 2.

This skipping is a secondary mechanism in the paper; the contribution is the differentiable
selection of experts to remove permanently (sections 4.1 and 4.2, intra-layer logits and
inter-layer importance scores). On the one fine-grained model the paper reports it on,
Deepseek-MoE-16B at top-6 of 64, skipping buys a **1.04x speedup for 0.4 to 0.5 accuracy
points** (Table 5) — a mild efficiency tweak, not a way to halve the active expert count.

### How our version differs

1. **Only section 4.3 is implemented.** There is no differentiable pruning stage anywhere in
   the repository, and the calibration artifact holds only `sim_matrix`, `mean_sim`,
   `gamma_1` and `token_counts`. So what the tables call DiEP is the paper's auxiliary
   skipping heuristic, evaluated on its own.
2. **k=2 is generalised by comparing every selected expert against the top-1**, giving
   `threshold_i = w_e0 * beta * gamma_2(e0, e_i)` for ranks >= k_min. That choice is ours.
3. **`gamma_1` is replaced by the swept `--naee_beta`**, because `gamma_1` is a fixed
   constant and would pin the method to a single operating point. At `beta = 0.81`, the mean
   of `gamma_1` over the 48 layers, our threshold numerically reproduces the paper's.

### Why it collapses

**`gamma_1` is a rank-1 statistic used as a rank-independent threshold.** It is the median of
`w_e1/w_e0`, so at k=2 it skips the second expert about half the time by construction. Held
against `w_e0` for every rank of a top-8 router it prunes the whole tail: at a
rank-to-rank decay near the measured 0.81, rank 3 already sits at 0.81³ = 0.53, below the
paper's median gamma of 0.59. A rank-aware generalisation would calibrate `gamma_1_i` as the
median of `w_ei/w_e0` per rank.

**It is `gamma_2`'s spread, not its tail, that does the damage.** Weighting each expert pair by
how often the router picks it (from the artifact's token counts, rather than counting all
off-diagonal pairs equally) puts `gamma_2` at:

| model | p10 | p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| Qwen3-30B | 0.31 | 0.60 | 0.94 | 1.77 | 28.6 |
| Ling-lite | 0.66 | 1.01 | 1.31 | 1.81 | 21.6 |

Because gate weights are sorted descending, any pair with `beta * gamma_2 >= 1` is dropped
whatever weight the router gave it. That share is small at the settings where DiEP already
collapses: on Qwen3-30B 2.1% at beta=0.7, 7.8% at beta=1.0 and 17.3% at beta=1.2. Ling-lite is
the opposite, 4.4% at beta=0.7 but 52% at beta=1.0 and 76% at beta=1.2.

So the earlier reading of this as an unboundedness problem was wrong. At beta=0.7 on Qwen3-30B,
where DiEP is 14.5 points below Fixed-K, 98% of decisions are nowhere near the bound. What hurts
is the dispersion: p10 to p90 spans a factor of three, so the effective bar wanders between 0.22
and 0.66 of `w_e0` while NAEE holds a flat 0.45 for the same average budget. DiEP is not pruning
more, it is pruning worse — dropping an expert with a healthy gate weight because it happens to
resemble the top-1, and keeping a negligible one because it does not.

Worth recording so it is not attempted again: **bounding `beta * gamma_2` at 1 is a no-op.** With
the weights sorted, `w_ei < w_e0` holds for every rank past the first, so a multiplier of 3 and a
multiplier of 1 produce the same decision, prune everything past `k_min`. A cap only changes
anything strictly below 1. This is checked in `scripts/test_diep_gamma_alpha.py`.

Note that the paper's own setting sits at 21.6%, between our beta=0.7 and beta=0.85 rows, so
the transfer failure is not an artefact of pushing beta too far. Feeding `gamma_1 * gamma_2`
into the same top-k generalisation puts it at an estimated average of **3.63 of 8 experts** on
Qwen3-30B and **2.29 of 6** on Ling-lite (`scripts/estimate_diep_paper_point.py`, k_min=2).
That is worth stating plainly: on a top-2 router the rule can remove at most one expert per
token, which is the regime the paper validates at 1.04x to 1.07x speedup; on a fine-grained
router the identical rule removes more than half of the activated experts, and our measurements
at that budget put it 14 to 21 points below Fixed-K.

### Plan

Nothing here is a coding slip, so this is a methodological choice. Agreed 2026-08-03:

1. ~~**Wait for the NAEE re-runs before attributing the collapse.**~~ **Settled 2026-08-03**:
   NAEE lands 2.4 to 2.6 points below Fixed-K at matched budgets (beta 0.45 and 0.55, table
   above), so thresholding against the top-1 gate weight is not what breaks DiEP — `gamma_2`
   is. Step 3 is therefore live. The third setting (beta 0.35) is still in flight and only
   extends the curve.
2. **Measure the paper's own operating point**, one run per model: `--diep_use_gamma1` with
   `--naee_beta 1.0` makes the threshold `w_e0 * gamma_1 * gamma_2`, which is eq. 12 exactly.
   This is the only setting the paper specifies, it needs no swept hyperparameter, and it is
   the most defensible DiEP number we can report. Estimated to land at avg 3.6 (Qwen3-30B)
   and 2.3 (Ling-lite), both inside the sweep range.
3. **Damp `gamma_2` instead of clipping it**, and report the result as a separate variant
   alongside the plain generalisation, since the contrast is the useful part. Done 2026-08-03,
   `--diep_gamma_alpha` and `--diep_threshold_cap`:

       bar = min(beta * gamma_2**alpha, cap)

   `alpha=1` is the plain generalisation and `alpha=0` is NAEE exactly, verified token for token
   at four betas in `scripts/test_diep_gamma_alpha.py`, so alpha is a single knob that
   interpolates between them and measures what the similarity signal is worth. The runs use
   `alpha=0.5`, `cap=0.9`. The cap is secondary — see above, it has to be below 1 to do anything
   and only matters at Ling-lite's aggressive settings — but it also stops the rule degenerating
   into "prune to `k_min` regardless of weight" at avg 2 and 3.

   Rejected: calibrating `gamma_1` per rank. It would be defensible on its own, but NAEE uses the
   same flat bar across all eight ranks and loses only 2.5 points, so a flat bar is demonstrably
   not the problem, and adding rank calibration to DiEP alone would leave it differing from NAEE
   and MC-MoE by two things at once, which is exactly the confound the NAEE re-runs were spent to
   remove. `gamma_1` stays available for the paper-faithful point in step 2 and nowhere else.

   beta per target is solved offline (`scripts/solve_diep_beta.py`) so no lane is spent
   searching. The NAEE runs measure the survival function of the weight ratios directly, the
   artifact gives the `gamma_2` distribution, and the two combine into a predicted average. The
   assumption that they are independent is checked against the four DiEP runs already measured,
   which the fit does not see: the prediction is within 0.15 experts at beta=1.0 and 1.2 and
   overshoots by 0.49 at beta=0.5 and 0.7, and that residual is corrected for, scaled by alpha
   since at alpha=0 the prediction is exact by construction. The first run to use a solved beta
   confirms it: `alpha=0.5, cap=0.9, beta=1.394` was predicted to average 2.20 experts and
   measures 2.189.

   **First result, at avg 6 on Qwen3-30B**: damping recovers part of the loss but does not remove
   it. Reading the three points as a sweep in alpha at a matched budget of about six experts:

   | alpha | configuration | avg | suite mean |
   | --- | --- | --- | --- |
   | 0 (NAEE) | beta=0.35 | 6.41 | 73.56 |
   | 0.5 | beta=0.442, cap=0.9 | 6.03 | 68.76 |
   | 1 (plain) | beta=0.5 | 5.95 | 65.87 |

   The cost is close to linear in alpha, about 5 points per half unit, with no threshold below
   which the similarity signal is free. That is the cleanest statement available about what
   `gamma_2` is worth in a top-8 router: nothing, and it is priced in proportion to how much of it
   is used. Whether the remaining four damped runs at lower budgets keep that linearity is what
   they are for.

### On Qwen3-Next-80B the beta must be probed, not solved, and k_min=1 was not worth its runtime

Two things came up when the variant reached the top-10 model, both decided 2026-08-09.

**The offline solver is off by enough to matter here.** It put `beta=1.786` (plain, `k_min=2`) at an
average of 3.00; the run measures 2.62. The residual correction it applies is fitted on the other two
models, and on a top-10 router the `gamma_2` distribution sits low enough that the cap binds for a large
share of tokens, which the independence assumption does not capture. So on this model the operating
point is read off short probes instead — one dataset, 100 questions, 256 new tokens, about three
minutes each on four otherwise idle cards, against the thirty to fifty hours a full suite costs here.
Three of them bracket the curve tightly, and a fourth confirms the interpolation:

| alpha=0.25, cap=0.9 | measured avg |
| --- | --- |
| beta=0.57 | 4.06 |
| beta=0.60 | 3.81 |
| beta=0.72 | 3.03 |
| beta=0.812 | 2.62 |

Worth noting from the last row: plain `alpha=1` at `beta=1.786` and damped `alpha=0.25` at `beta=0.812`
both average 2.62. That is not a coincidence but it is also not degeneracy — both happen to put the mean
threshold at about 0.74 of the top weight, which is where NAEE sits at `beta=0.74`.

**The two `k_min=1` runs were stopped part-way and dropped from the plan.** They were the old table's
settings for targets 4 and 3 (`beta=0.95` and `beta=1.1`) and they were generating 5570 tokens per
question — the verbosity that follows from routing some tokens to a single expert, the same collapse
already recorded for Ling-lite and gpt-oss. Measured against completed runs at comparable budgets
(`NAEE beta=0.75` took 52.6 h, `Fixed-K k=2` 51.5 h) they had 25 to 35 hours left each, on eight cards,
and they were the only thing keeping the sweep from finishing a day and a half earlier. What they would
have added is also the least comparable row in their tiers: at target 4 NAEE uses `k_min=2` and Ban
`k_min=3`, so a DiEP row with the floor at 1 differs from its neighbours by two things at once. The
damped rows at `beta=0.57` and `beta=0.72`, both `k_min=2`, take their place. Plain DiEP keeps its rows
at avg 6 and 5, and the `beta=1.786` run at 2.62 gives it a `k_min=2` point at the tight end, so the
before-and-after contrast the variant exists to show is still on the table.

Explicitly not doing: implementing sections 4.1 and 4.2, because that stage is static pruning
for memory and its output cannot be placed on a per-token average-experts axis; switching
`--diep_pruning_mode` to `prefix`, which is strictly worse here since one failing rank
truncates every rank behind it; and reinterpreting the rule as the soft down-weighting the
paper's introduction hints at, which saves no computation and so cannot produce a budget point.

The rows currently labelled DiEP should be relabelled to show they are our top-k
generalisation rather than the paper's method; deferred until the method is settled. No other
method is affected.

### The avg-2 tier was written off on one word from the solver

Decided 2026-08-10. The solver printed `unreachable, capped` for any target the average never reached,
and that reads as "no setting exists". On the strength of it this file and the 80B report said the DiEP
line could have no same-budget row at avg 2, and the tier was skipped on Qwen3-30B as well. What the
message actually meant is narrower: with a threshold cap the average approaches a floor instead of
crossing the target, and when that floor sits near the target, the beta where it settles *is* the
operating point. Three probes, one per model:

| alpha=0.25, cap=0.9 | measured avg |
| --- | --- |
| Qwen3-Next-80B, beta=1.2 | 2.075 |
| Ling-lite, beta=1.1 | 2.083 |
| Qwen3-30B, beta=1.285 | 2.087 |

The solver now reports the saturating beta and how far the floor is from the target; re-solving the 80B
tier returns `beta=1.221` for an average of 2.00, so it finds the row it had previously talked us out of.

Of the three, only Qwen3-30B's is held rather than run, and for cost rather than reachability: at the 32k
limit that model's completed rows in this tier took 40 h (`Fixed-K k=2`), 84 h (damped `alpha=0.5` at
2.308) and 97 h (`Ban` at 2.013), because the fewer experts it keeps the longer it rambles. Four days of
two cards buys little here — every method in that tier scores between 13.9 and 27.2 against a 74.4
baseline, so the tier says "two experts destroys this model" no matter who is asked. The other two models
still have signal at avg 2 (Ling-lite 33.5 to 41.8 against 58.8; the 80B 39.9 to 52.6 against 79.1), which
is why their rows are worth their runtime. The measured 2.087 above is the whole of what the held run
would have contributed to the average-experts axis.

## Ling-lite-1.5-2507: shared experts are not counted in the average

Status: **open, needs a decision on methodology.** Found 2026-07-31.

Renormalization was also suspected here, but it is a non-issue: `bailing_moe.py` sets
`renormalize` from `config.norm_topk_prob`, which is `True` in this checkpoint, so
Ling-lite renormalizes surviving gate weights exactly like the Qwen3 models do.

### Shared experts are excluded from `average_selected_experts`

Ling-lite has 2 always-on shared experts on top of the routed top-k. Our statistics only
count routed experts, so the reported average understates the compute actually spent, and
an "avg ≈ 3" Ling-lite point is not comparable to an "avg ≈ 3" Qwen3 point (Qwen3-30B has
no shared expert). Decide whether the target-average axis should count shared experts.

## Qwen3-Next-80B: the aggressive budgets generate under a 16k limit, not 32k

Status: **applied, one control run in flight.** Changed 2026-08-07.

At two or three routed experts out of ten this model rambles: nearly every answer runs to the
generation limit, which is why `NAEE-kmin2-beta0.75` needed about nineteen hours for the 873
livebench prompts alone and Fixed-K k=2 spent nineteen hours on mmlu_pro. The premise of the
change is that an answer that is not correct within 16k tokens is almost never correct at 32k,
so the second half of the window buys time and nothing else. The measured effect on cost is
larger than the halving of the limit suggests — the restarted Fixed-K k=2 projects about six
hours for mmlu_pro against the nineteen it had just spent — because a shorter window also
shortens the tail of prompts that would have run to the end.

Which runs: every non-Baseline configuration whose lowest target is 3 or below, seven in all.
The unpruned baseline stays at 32k because it is not verbose and was already measured there,
and the avg-4 group stays at 32k because its Fixed-K reference was measured there. A table
whose rows use different limits cannot separate a method's loss from a truncated answer, so the
limit is chosen per table, not per run. Nothing changes for the other three models.

The premise is not verified, which is the open part. `Ban-kmin2-lambda0.3` had already reached
its last task at 32k when the change was made, so it was left alone and a second run of the
same configuration, `Ban-kmin2-lambda0.3-cap16k`, was queued instead. Those two rows are the
control: identical routing, the two limits, so their difference measures what the shorter
window costs at this budget. If it is small the change is free; if it is not, the seven capped
rows each owe a correction of roughly that size, and the avg-2 and avg-3 tables would need to
be re-run at one limit.

Mechanically the limit is appended to a configuration's flags as `--max_new_tokens 16384`,
which overrides the worker's own `32768` because argparse takes the last occurrence. It is part
of lighteval's model hash, so a capped run gets a fresh cache namespace and cannot replay
samples generated under the wider limit — the two restarted runs began from scratch. Reports
read the limit back from each run's `config_general.model_config.generation_parameters` rather
than from its configuration, and mark any row that used a shorter one, so the record stays
truthful even if a configuration file is edited afterwards.

## gpt-oss-20b: forcing eager mode to count experts cost a factor of 36

Status: **fixed, runs restarted.** Measured 2026-08-07.

gpt-oss is mxfp4-quantized, so its expert selection is inlined into the compiled region instead of
reaching the experts through an opaque vLLM custom op. `main.py` reacted to that by forcing
`--enforce_eager` for every routed run on this model, because the router's Python-side counters stop
running once Dynamo traces them, and `average_selected_experts` is the x-axis of every table here.

The price of that decision was never measured. It is large. Same configuration, same 100 mmlu_pro
prompts, one card:

| mode | wall clock | mmlu_pro | average experts reported |
| --- | --- | --- | --- |
| eager, router | **182 min** | 59.0 | 2.028 |
| CUDA graphs, router | **5 min** | 54.0 | 1.954 |
| CUDA graphs, no router | 3 min | 72.0 | — |

Two things follow. The pruning does apply under compile — the graph arm sits next to the eager arm
and far from the unpruned one, which is the failure that mattered: had the router been silently
bypassed the score would have climbed toward 72 with no statistics to notice it. And eager mode is
not a small tax on a 20B model. Its per-step Python and launch overhead cannot be amortized when the
batch is small, which is exactly the state a collapsed model's tail is in: the last six of those 100
prompts took 2.6 minutes each at 2% GPU utilization, each running to the 32k limit alone. In
production, with thousands of prompts in flight, the same comparison is milder but still decisive —
166 tok/s against the 1138 tok/s the Fixed-K runs sustain with graphs.

At that rate the eight routed gpt-oss runs were not going to finish. `DiEP-kmin1-beta1.95` had spent
26 hours to reach 54% of mmlu_pro, the first of nine tasks, and Fixed-K needed 15.4 hours for aime24
*with* graphs. One to two weeks per run, on eight cards that the 80B sweep could otherwise use.

The fix is `--allow_compiled_router`: keep CUDA graphs, accept that the run cannot count its own
experts, and measure the average in a separate short eager probe of the same configuration,
`<name>-avgprobe`. All eight runs were restarted this way; switching mode changes lighteval's model
hash, so none of their earlier work could be replayed.

The probe has to run eager — that is its whole purpose — so the first version of it inherited exactly
what made eager unaffordable. With no limit on generation, a collapsed model wrote single answers out
to the 32k window at 6 tok/s: **45 minutes per prompt**, four hours for 38 of them, on a card the last
unclaimed run was waiting for. An average over tokens does not need long answers, so probes are now
100 mmlu_pro prompts with generation limited to 256 tokens — around a million routing decisions, five
minutes. What the limit changes in principle is which tokens are averaged, since a truncated answer
never reaches the repetitive tail where routing might drift; the prompts and count are therefore the
same 100 the eager arm of the A/B used, where the full-length average is known to be 2.0277 for NAEE
beta=0.75, so the first probe to land is a check on the shortcut and not only a measurement.

One subtlety worth keeping. A graph run *does* report an average, but a wrong one: vLLM pads each
batch up to a captured graph size and those padding rows are counted as tokens, which is the 1.954
against 2.028 above, low by 3.6%. That number is therefore recorded as
`average_selected_experts_padded` and the field the reports read stays empty, so a row can only get
its operating point from the probe. Rows measured this way are marked ᵍ.

## lm_eval: a finished evaluation exited 134, and would have been run again

The zero-shot QA suites (`--harness lm_eval`) reach the same vLLM engine through a different wrapper,
and that wrapper has no equivalent of lighteval's `cleanup()`. Left alone, the process wrote every
result correctly and then died with `terminate called without an active exception` and SIGABRT: torch
warns that `destroy_process_group()` was never called, and the NCCL process group's watchdog thread is
still joinable when the libraries unload, which is a std::terminate. Nothing about the measurement was
wrong — only the exit code, which is enough for a queue worker to treat a completed evaluation as a
failure and spend another card-hour repeating it.

Three things were needed, and only the first is the one lighteval already does:

* Destroy the process group explicitly. vLLM's `destroy_distributed_environment()` did not cover the
  default group in this version, and it is the group's watchdog that aborts.
* Reap exited children. An exited child stays listed until its parent reaps it, and a zombie cannot be
  killed, so the engine process looked like something that refused both to exit and to die.
* Wait for the engine process to be gone, then leave a small margin. Adding the child-process check
  alone — tens of milliseconds — already turned one full-scale run from 134 into 0, which says the race
  window is that narrow; waiting on purpose is the honest version of that accident. The wait also
  covers the worse failure it would otherwise hide: an engine that outlives the run keeps its share of
  the card, and the next configuration scheduled onto it cannot allocate.

`multiprocessing.resource_tracker` is excluded from that wait. It is Python's own helper for catching
leaked shared memory, it holds no GPU, and it is meant to outlive everything else until the parent
goes — waiting for it meant waiting the full grace period on every run and then killing it with a
message about GPU memory it never held.

## The QA harness is exactly reproducible, and that is how we found out it is not batch-invariant

Four of the 80B recalibration rows failed three times each and were given up on, which is a state the
loop could not report: a row whose run has been abandoned looks identical, from the solver's side, to a
row whose run has not started — both are "still under budget, nothing new to queue". It said `入队 +0`
eight rounds in a row while the node sat idle, because `ensure_supervisor` only restarts a supervisor
when there is *unclaimed* work and all 19 claims were taken (fifteen finished, four abandoned). Eight
cards idled for an hour and a half. The supervisor guard is right — two supervisors on one host hand the
same card to two runs — so the fix is that the loop now names the abandoned configurations each round
instead of reporting the absence of proposals. It deliberately does not resurrect them: the retry cap
exists so a genuinely broken configuration cannot burn cards forever, and telling that apart from an
environmental failure means reading the log.

The failures themselves were `CUDA out of memory occurred when warming up sampler with 256 dummy
requests`, and the reason only DiEP and Ban hit it is arithmetic rather than anything about those
methods being wrong. On a 47.37 GiB card at TP=4 this model places 37.22 GiB of weights, 1.42 of
activation peak, 2.79 of CUDA graphs and 3.84 of KV cache — 45.4 GiB, above vLLM's own 0.9 target,
because the graphs are captured after the cache has been sized. Under 2 GiB is left, the sampler
warm-up wants a few hundred MiB of it for 256 sequences at once, and the two methods whose routers peak
slightly higher do not fit. NAEE and Dynamic Routing did, which is why fifteen rows had already landed
and the failure looked method-specific.

Capping vLLM's concurrent sequences at 64 fixes it, and the interesting part is what that cap does and
does not change. Measuring one configuration all three ways:

* **Repeating a run at unchanged settings reproduces every metric exactly.** All seven numbers, to the
  digit. So this harness has no run-to-run noise to hide behind, even at TP=4 — the expert average
  moved in its fifth digit (6.067201 against 6.067415), which is statistics accumulation, not routing.
* **The cap does not touch expert selection**: 6.067415 experts at 64, 6.067415 at 256.
* **It does move the scores**, by 0.92 points on average and 2.20 at most, in both directions
  (openbookqa −2.20, arc_easy +0.97). Different batch shapes give different reduction orders, the last
  bits of the logits move, and multiple-choice calls that were near ties flip.

So the scores are deterministic per setting but not comparable across settings at the one-point level,
which is roughly the standard error of the four-task mean anyway (0.74 points). The marked rows are
labelled ᶜ in the tables with that number attached, rather than re-running the 47 rows that already
landed at 256 to chase an effect the size of the error bar. What would not have been acceptable is
leaving it unlabelled, because a point of difference between two rows is exactly the size of claim the
QA tables are used to make.

## A threshold is not a budget: the QA sweep was comparing rows that spent less than Fixed-K

The 0-shot QA sweep reused the thresholds from the generative tables, on the reasoning that the same
router at the same setting is the same configuration. It is — but the number of experts it spends is
not a property of the setting, it is a property of the setting *and its input*. Every one of these
methods keeps experts until the router's own confidence falls past a bar, and a router reading a
one-sentence multiple-choice question is more decided than one in the middle of a derivation. So the
same threshold that averaged 6.16 experts on the generative suite averaged 4.85 on QA.

That is not a small imprecision, because it breaks the comparison the tables exist for. Fixed-K spends
exactly its k on any input, so a dynamic row that came out below Fixed-K *and* cheaper than it says
nothing about the method: the lower score is explained by the smaller budget. Across the four models 48
of the QA rows were in that position, worst at −1.15 experts (Qwen3-Next-80B, NAEE at the avg-6 tier)
and −1.13 (Qwen3-30B, Ban at avg 6).

The fix is a separate calibration for this harness — `scripts/solve_qa_hyper.py`, which inverts each
method's measured QA averages at each tier's k plus a small margin, since spending slightly more than
Fixed-K is honest and spending less is not. The re-solved configurations are their own operating points,
so each has its own directory and its own `qa_only` entry in the master list; the generative tables and
their finished-run counts leave those out, and the QA tables prefer the cheapest row that still spends
at least what Fixed-K does. Runtime made this affordable to iterate rather than derive: one QA run is
two minutes on a 48 GB card for Ling-lite and twelve for the 80B, so the loop measures, re-solves against
what it measured, and queues again. The first seven landed all within [k, k+0.35].

Two things were worth knowing before trusting the arithmetic:

* **Ban has to be solved from its own rule, not from a slope.** Ban does not threshold a weight, it
  picks a count: `k = k_min + (topk - k_min) * clamp(lambda * s)`. Its lambda is bounded at 1, and on
  Qwen3-30B no lambda reaches the avg-6 tier at the floor the generative row used — 0.95 already gave
  4.87, and 1.0 would give about 4.97. One measured average pins down the mean sensitivity `s` this
  model shows on these questions, and the floor that does reach the tier follows in closed form:
  `k_min=5, lambda=0.931`, which measured 6.032. Guessing the floor instead would have spent runs on
  `k_min=6`, where Ban cannot be anything but fixed top-6.
* **The slope has to be read in the direction of the correction.** These curves steepen as the
  threshold falls, so taking the segment nearest the *current* average sends beta far past where it
  needs to go when the nearest segment lies on the flat side: 0.636 to 0.314 for four tenths of an
  expert, on Ling-lite's damped DiEP. The segment that brackets the target is the one that describes
  getting there.

The same principle is broken in two of the generative rows — Qwen3-30B's Ban at avg 6 spends 5.05 and
Qwen3-Next-80B's DiEP at avg 5 spends 4.35 — and eight more sit 0.05 to 0.2 under. Those runs cost a
day each rather than eight minutes, so they are marked in the tables rather than re-run.

## VL models: the harness is ready, but neither checkpoint can be *pruned* yet

The multimodal harness (`--harness lmms_eval`) is wired up and the datasets are in place, but the two
checkpoints it is meant for are each blocked by something below the harness. Both blocks are in the
engine layer, not in the evaluation, so unpruned and Fixed-K runs are unaffected and are the right way
to establish reference numbers.

**Qwen3-VL-30B-A3B: the pinned vLLM does not know the architecture.** The checkpoint declares
`Qwen3VLMoeForConditionalGeneration`; the model registry of the commit this repository pins (01efc7e,
2025-09-15) has no `qwen3_vl` implementation at all, only `qwen2_5_vl` — the model is newer than the
pin. `transformers` 4.57.6 does support it, which is what makes the gap easy to miss: nothing fails
until the engine looks the architecture up.

The resolution is not to raise the pin: the router patch attaches to a specific `fused_topk` and the
whole 40-card sweep runs on this combination. The multimodal harness therefore lives in the `qwen35`
environment (`install_qwen35.sh`), which already exists for the same reason — models too new for the
pinned engine — and whose vLLM 0.17.0 ships `qwen3_vl_moe.py`. Its text tower is
`Qwen3MoeSparseMoeBlock`, i.e. the same router we already prune on Qwen3-30B-A3B, so nothing about
the method has to change for this model.

What does have to change before that environment can prune anything is the patch's attachment point.
Between the pinned commit and 0.17.0, `fused_topk` moved from
`fused_moe/fused_moe.py` to `fused_moe/router/fused_topk_router.py`, and its signature grew a
`scoring_func: str = "softmax"` argument. So the port is small and mechanical, but until it is done,
`qwen35` runs unpruned baselines and Fixed-K only — the latter works because
`--num_experts_per_tok` rewrites the config and never touches the router.

That baseline path is now measured rather than assumed: on 2026-08-10 the unpruned Qwen3-VL-30B-A3B
answered 16 `chartqa_lite` questions in 4.7 minutes on two cards, `relaxed_overall` 0.5625, images and
all. Two further things had to be fixed before the full nine-dataset suite would finish, both of them
in our own call into the harness rather than in the model: `apply_chat_template` below, and a CLI
namespace too thin for the tasks that read settings off it — `ocrbench` asks for `output_path` when it
writes its submission file, and it asks during aggregation, so the AttributeError arrived 4.5 hours in
with all 9584 answers already generated and nothing saved. Unset attributes now read as None, which is
what a task would see from the upstream CLI for an option nobody passed, and a 16-question pass over
all nine datasets (13 minutes) is the cheap way to exercise every aggregation function before
committing to the long run.

One measurement caveat from that pass: `ocrbench` normalises its score by fixed per-category totals
rather than by how many questions were asked, so under `--max_samples` it reports near zero (0.015 for
16 questions) and only a full run of that task means anything.

The chat-template one is worth spelling out because it looks like a setting rather than a bug.
`simple_evaluate` was being passed `apply_chat_template=True`, which reads like the setting that decides whether an instruct
checkpoint sees a chat format — but the wrapper `get_model("vllm")` returns builds its own OpenAI-style
messages from each task's `doc_to_messages`, so that is not where templating happens, and the flag
cannot be turned on at all: `evaluate()` does `getattr(lm, "apply_chat_template")` and no model class
in lmms-eval v0.7.2 defines it. It is now off, and the `--lmms_eval_no_chat_template` switch that
implied a choice existed is gone.

**Kimi-VL-A3B, first blocker: the pinned engine cannot construct it at all.** Measured 2026-08-09 by
running the unpruned baseline as a smoke test in the `expertpruning` environment, where it failed in
thirty seconds, three times over. vLLM's registry inspects the architecture in a subprocess and that
subprocess dies importing the model:

    ImportError: cannot import name 'PytorchGELUTanh' from 'transformers.activations'

The pinned commit's `kimi_vl.py` reaches for a `transformers` symbol that 4.57.6, the version this
environment pins for everything else, no longer exports. So this is not about pruning: the unpruned
baseline cannot run here either, and Kimi-VL belongs on the `qwen35` line for the same reason Qwen3-VL
does. The `lmms_eval` half of that smoke test is therefore untested by this run; the `lm_eval`
regression that ran beside it passed (Ling-lite, four tasks, 361 s, average 2.803 experts).

**Kimi-VL-A3B, second blocker: its router does not go through the function we patch.** The language tower is
DeepSeek-V2-shaped and its config asks for `topk_method: noaux_tc`, and vLLM's `DeepseekV2MoE` passes
`use_grouped_topk=True` unconditionally for that method — so expert selection happens in
`grouped_topk`, while the patch replaces `fused_topk`. This does not silently produce unpruned
numbers: `_assert_layer_routes_through_patch` was added for exactly this case and stops the run. What
it costs is that every dynamic method needs a `grouped_topk` counterpart before Kimi-VL can be pruned
at all.

Worth noting for when that is written: with `n_group: 1` and `topk_group: 1` the grouping is a no-op,
so the arithmetic reduces to a plain top-k — but over *sigmoid* scores with an added correction bias
(`e_score_correction_bias`), not the softmax probabilities every threshold in this repository is
calibrated against. A ratio-to-the-top rule like NAEE's does not mean the same thing on a sigmoid
score as on a normalised probability, so the sensible reading is to apply the rule to the normalised
routing weights vLLM computes after selection, and to treat any Kimi-VL budget as separately
calibrated rather than transferable from the text models. Kimi-VL also keeps 2 always-on shared
experts on top of its 6 routed ones, the same caveat already recorded for Ling-lite.

**DeepSeek-V2-Lite-Chat, staged as a standby model, lands on the cheap side of that same wall.** It is
blocked by the identical mechanism — `use_grouped_topk=True` is passed for every `DeepseekV2MoE`, not
only for `noaux_tc` layers, so the guard stops a pruning run here too — but what the guard is standing
in front of is different. Its `topk_method` is `greedy` and its `scoring_func` is `softmax`, which
means `e_score_correction_bias` is `None`, and with `n_group: 1` and `topk_group: 1` the grouping
selects nothing away. The selection is therefore the same arithmetic on the same normalised
probabilities that every threshold here is calibrated against, so this model needs the hook extended
to a second entry point and not a score-space rethink: the argument that made Kimi-VL expensive does
not apply. Budgets still have to be solved per model, as always. Its shape is close to Ling-lite's —
64 routed experts, top-6, plus 2 always-on shared ones (same avg-counting caveat), 27 layers with the
first dense — and unlike Kimi-VL it loads in `expertpruning` as it stands: config, tokenizer and
vLLM's registry all accept it, with only the usual benign `rope_scaling` field warnings from
transformers on the YaRN entries.

## Qwen3-30B-A3B-Thinking-2507: at two experts it stops being able to stop

Status: **8k generation limit applied to the k=2 row only.** Added 2026-08-11.

This model is in the study to be compared against `Qwen3-30B-A3B-Instruct-2507`, which is the same
architecture down to the layer count and the router — 48 layers, 128 experts, top-8 — so any
difference in how far pruning degrades it is a difference between reasoning in the output and
answering directly. That is the whole reason it runs under settings shaped for the other model, and
two of those settings are worth stating plainly.

**Two of the nine tasks cap generation below what a chain of thought needs.** gsm8k allows 256 tokens
and livebench 4096, against 32768 for the rest of the suite. A thinking trace does not reach
`</think>` inside 256 tokens, so those columns read near zero for the *unpruned* model too, and
averaging them in would credit pruning with preserving a score the model never got to produce. They
are shown, marked ˣ, and left out of the mean. Raising them was the obvious alternative and is worse:
the comparison against Instruct's existing numbers is the only reason to run this model, and those
numbers were produced under exactly these caps.

**At k=2 the run was not going to finish.** It took 63 seconds per prompt, which projects to 210
hours for mmlu_pro alone, against the 40.2 hours Instruct's whole nine-task suite took at the same
budget. Two things were going on and only one of them is about the model. The mechanical one is the
KV cache: this router keeps 4 KV heads of 128 dimensions across 48 layers, which is 96 KB per token,
so one sequence at the full 32k window holds 3.0 GB. Two 48 GB cards give 94 GB, weights take 61, and
what is left fits nine concurrent sequences — the run was not compute-bound, it was concurrency-bound.
Moving it to four cards raised the ceiling to about forty and cut the cost per prompt from 63 seconds
to 18.75. The other one is the finding: pruned to two experts this model essentially never emits a
stop token, so nearly every answer runs to whatever the window is, which is why the window is what
sets the price. An 8k budget for that row brings it to 2.32 seconds per prompt and 7.6 hours for
mmlu_pro, a factor of 27 against where it started, and the report reads the limit back out of the run
and marks the row ᵏ. The same trade was already made for Qwen3-Next-80B's deepest budgets, and it has
the same open end: a truncated answer scores zero however good it was going to be, so the k=2 row is
a lower bound on that tier rather than a measurement of it.

**Two operational notes from setting it up.** The standby models were downloaded to
the slower shared filesystem, which reads at 37.8 MB/s uncached against the faster one's 1.5 GB/s — 96 MB/s with four
readers, so it parallelises a little — and 57 GB took 45 minutes to reach two hosts instead of the
three minutes the earlier pushes took. Anything staged from there should be budgeted an order of
magnitude higher, which for the 235B checkpoint is hours. And `pgrep -f VLLM::EngineCore` matches
every run on the host, not the one being cleaned up: using it to clear a killed run's orphans took
down the engine cores of the two healthy runs sharing the node. They kept logging for five minutes
while their workers spun in NCCL on cards that read 0% utilisation with the memory still held, which
is what that failure looks like from outside. Orphans have to be identified by walking `/proc` for
processes holding an nvidia device and matching them to the dead run's process group, and it is worth
remembering that `nvidia-smi` inside a container reports host PIDs that cannot be signalled from
inside it.

## A claim is not evidence of a run, and three probes were dead for half an hour before anyone looked

Status: **fixed; a reconciling watchdog now runs.** Added 2026-08-12.

Three DiEP probes for the Thinking model were believed to be in flight on one host. They had died
within seconds of starting, all three on the same line: the calibration artifact was passed as the
shared-filesystem path it was produced on, and that host has no such mount. The artifact is 3.2 MB, so
the fix was to stage it on all three hosts and rewrite the flag, but the interesting part is why
half an hour passed with two cards idle and the scheduling looking healthy.

The queue marks work by creating `claim_<i>`. A worker claims, runs, and on failure bumps `tries_<i>`
and moves on — but if the *worker* is gone, the claim stays behind with no results and no process, and
every status check that counts unclaimed work reports the queue as fully claimed and therefore busy.
That is the same shape as the earlier accident where eight cards sat idle for an hour and a half with
nineteen claims held and four of them abandoned. Counting claims answers "has this been picked up",
never "is this running".

`/root/thinking_watch.sh` now reconciles the three facts every five minutes — a claim exists, no
results exist, no process is generating that directory — and releases the claim, then starts any lane
that is missing while its cards are free and its queue has work. It found the three dead probes on its
first pass.

It also has to know about one deliberate exception. Three configurations were moved to faster hosts
and their claims were **left behind on purpose**, so the original host would not pick them up again. To
a claim-reconciling watchdog those look exactly like the failure it exists to fix, and acting on them
would run the same configuration on two machines writing into one directory. Rewriting the queue to
remove them is not an option either: claims are keyed by index, so deleting an entry re-points every
claim after it. They are marked with a `moved_<index>` file that the watchdog skips.

## The Thinking model's router is more concentrated, so no threshold could be borrowed from Instruct

Status: **resolved by measurement; thirteen probes, both curves.** Added 2026-08-12.

The two models are the same network with different post-training, which made it tempting to reuse the
betas that landed on target for Instruct. At equal threshold the Thinking model consistently spends
fewer experts:

| method | beta | Instruct avg | Thinking avg |
| --- | --- | --- | --- |
| NAEE | 0.35 | 6.408 | 6.114 |
| NAEE | 0.45 | 5.134 | 4.963 |
| DiEP-damped α=0.25 | 0.413 | 6.110 | 5.927 |
| DiEP-damped α=0.25 | 0.79 | 3.008 | 2.963 |

The gap is small at a single point and compounds across the curve: Instruct reaches an average of 4.08
at NAEE beta 0.55, the Thinking model reaches it at 0.531, and by the middle of the range the same
threshold separates the two by more than half an expert. Borrowing the table would have put most tiers
under their budget, which is the one thing that makes a row incomparable to Fixed-K.

These are three-minute probes — 100 mmlu_pro prompts at a 256-token limit, far too short to score
anything, long enough to measure what a threshold spends — so thirteen of them cost less than one hour
of one card, against ten hours for a full run at the wrong operating point.
`/root/solve_thinking_beta.py` inverts the measured curves; the tier is targeted with a small margin
above it rather than exactly, because a run that spends slightly more than Fixed-K is comparable to it
and one that spends less is not, and three tiers that came in a shade under (5.927, 4.963, 2.963) were
re-probed rather than accepted.

## Ban's two knobs: why its top tier under-spends, and how the probes were pooled

Status: **solved per tier for the Thinking model.** Added 2026-08-12.

Ban keeps `k = k_min + (topk - k_min) * clamp(lambda * s)` experts. Two knobs, and the slope's factor
clamps at 1, which is the whole story behind the Instruct model's worst budget violation: at the 6 tier
with the floor at 3 and lambda already 0.95 it spent 5.054, and no larger lambda would have helped,
because the term it multiplies is already saturating. Reaching that tier needs the floor raised, and on
the Thinking model it takes a floor of 5 — which lands at 6.014 measured.

Rearranging the rule gives a way to spend fewer probes. The mean clamped activation

    E(lambda) = (avg - k_min) / (topk - k_min)

does not contain `k_min`, so in principle a measurement at any floor constrains every floor. Two of the
seven probes deliberately share lambda 0.9 at different floors to test that, and they disagree by 0.039
in E — small, but it maps to 0.2 experts at a floor of 3. The likely cause is that the per-token count
is an integer: a floor of 5 quantises three remaining slots more coarsely than a floor of 3 quantises
five. So `solve_thinking_ban.py` reads a floor's own points where it has two or more and pools only as
a fallback, and that distinction decides a real case — pooled, the 5 tier appeared to need lambda above
1 and a raised floor; on the floor's own two points it solves at 0.952 with the floor left at 3, which
is what keeps the row comparable to Instruct's.

One ordering mistake is worth recording because it looked like an unreachable tier. The solver tested
lambda feasibility before checking whether a probe had already landed in the target band, so the 6 tier
was reported as unsolvable at every floor: solving for 6.08 from a floor of 5 needs lambda 1.12, while
the floor-5 probe at lambda 0.7 was sitting at 6.014 all along. A measured point in the band needs no
solving, so that question has to be asked first.

## Multimodal pruning: the knee is where the text model's is

Status: **Fixed-K sweep complete; dynamic methods still blocked.** Added 2026-08-12.

`Qwen3-VL-30B-A3B-Instruct`'s six-configuration budget sweep is in `results/`, generated by
`scripts/report_vl.py`. Read against `Qwen3-30B-A3B-Instruct-2507` — same size, same native top-k,
nine multimodal datasets instead of nine reasoning ones — the two decline curves stay within 3.0
points of each other at every budget and break at the same place: nearly free to k=5, about six points
at k=4, collapsed at k=2 (14.20 against an unpruned 71.80). On this evidence the expert budget governs
the damage, not the modality; attaching a vision tower does not make the mixture measurably more
fragile. Only Fixed-K is available, for the reason recorded above: the router patch has not been
ported to the newer vLLM this checkpoint requires, and Fixed-K needs no patch because it is the
config-level top-k.

## The sigmoid router: DeepSeek-V3.2 cannot run here, and it is not the contrast it looks like

Status: **V3.2 ruled out on three independent counts; cheaper substitutes identified.** Added
2026-08-12.

Every model in the study scores its experts the same way — softmax over the gate logits, then the
selected weights divided by their sum. `expert_pruning/routing.py` bakes that in: each method's score
comes from `_topk_softmax`, and every threshold is stated as cumulative *probability* mass. A
sigmoid-scored router gives independent per-expert gates that sum to nothing in particular, so the same
β does not mean the same thing, which is why a sigmoid checkpoint is worth having. The one that was
proposed, `nvidia/DeepSeek-V3.2-NVFP4`, is not the one to use.

**It will not start on these cards.** The checkpoint carries `index_topk: 2048`, and vLLM keys DeepSeek
Sparse Attention off exactly that field (`platforms/cuda.py`): when it is present the platform *forces*
a sparse MLA backend rather than falling back to a dense one. Both sparse backends refuse our GPUs —
`FlashMLASparse` accepts compute capability major 9 or 10, `FlashInferMLASparse` only 10 — and the RTX
Blackwell reports 12.0. The dense backends are irrelevant here even though `TritonMLA` accepts
any capability, because the forcing happens before backend selection. Note that neither memory nor the
FP4 format is the obstacle: 386.6 GiB of weights over eight 71.7 GiB cards leaves about 23 GiB a card,
and NVFP4 weights would have found a home in the Marlin MoE path, which needs only capability 7.5.

**It would need the grouped router we have not written.** `n_group: 8`, `topk_group: 4` — real
group-limited selection, the case that genuinely needs a grouped version of each method, unlike
DeepSeek-V2-Lite and the two candidates below where a single group makes the grouping a no-op.

**And it does not isolate renormalization.** Its config says `norm_topk_prob: true`, so the selected
sigmoid gates *are* divided by their sum, and then scaled by `routed_scaling_factor: 2.5`. It is a
sigmoid-with-renormalization router, one axis away from ours rather than two.

### Nobody ships a sigmoid router that skips renormalization

Fourteen checkpoints' configs were read for this, and every sigmoid-scored router among them
renormalizes the selected weights: DeepSeek-V3/V3.2, Kimi-K2 (384 experts), Kimi-VL, Moonlight,
GLM-4.5-Air/4.6/4.7-Flash, Ling-1T, Ling-mini-2.0. MiniMax-M2 is the one whose config leaves
`norm_topk_prob` out, and it is not an exception either — `minimax_m2.py` passes `renormalize=True`
as a literal. Two of these cannot be read off the config at all: GLM never declares `scoring_func`,
`glm4_moe.py` hardcodes sigmoid, so a missing field here means "read the implementation", not "softmax".

| scoring | renormalized | checkpoint | weights | runs in the pinned engine? |
| --- | --- | --- | --- | --- |
| softmax | yes | the five models already in the study | on disk | yes |
| softmax | **no** | `DeepSeek-V2-Lite-Chat` (`norm_topk_prob: false`) | on disk, 29.3 GiB | yes |
| sigmoid | yes | `Moonlight-16B-A3B-Instruct` | 29.7 GiB | yes (`DeepseekV3ForCausalLM`) |
| sigmoid | yes | `GLM-4.5-Air` | 205.8 GiB | yes (`Glm4MoeForCausalLM`) |
| sigmoid | yes | `GLM-4.7-Flash` | 58.2 GiB | no — `qwen35` only, see below |
| sigmoid | yes | `Ling-mini-2.0` (V2 of the family whose V1 we already prune) | 30.3 GiB | no — `BailingMoeV2` unregistered |
| sigmoid | no | none exists | — | — |

So the un-renormalized arm is already downloaded and it is the *softmax* one, which is the opposite
pairing to what one would guess. The sigmoid-without-renormalization cell has no checkpoint because
nobody trains that way; it can only be produced as our own ablation, by turning renormalization off on
a model trained with it — which measures our methods' sensitivity to the score shape rather than
anything a model ships with.

For the sigmoid arm the cheapest choice is also the best-matched one. Moonlight-16B-A3B is the
text-only sibling of Kimi-VL-A3B and its routing shape is *identical to Ling-lite-1.5-2507's* — 64
experts, top-6, two shared experts — so the pair differs in the scoring function and almost nothing
else. It has one group, dense MLA, and `DeepseekV3ForCausalLM` is in the pinned registry, so it can be
pruned rather than only budget-swept. Its limits are an 8k context and February-2025-era quality, which
will show up as low absolute scores on the harder tasks.

### GLM-4.7-Flash needs a four-line config class, and Fixed-K is all it can do

`Glm4MoeLiteForCausalLM` is registered in vLLM 0.17.0 (`qwen35`) but not in the pinned engine, so this
model cannot be pruned until the router patch is ported. Its config is also newer than the environment:
`model_type: glm4_moe_lite` is unknown to transformers 4.57.6, there is no 4.x release that knows it
(the version list goes 4.57.6 → 5.0.0, and the checkpoint was written by 5.0.0rc0), and vLLM 0.17.0
requires `transformers<5`. `expert_pruning/config_shims.py` closes that gap without touching the
environment: `PretrainedConfig` keeps unknown keyword arguments as attributes, so a subclass carrying
only the right `model_type` hands vLLM every field the checkpoint declares. Verified on CPU —
`ModelConfig` builds, `use_mla` is True, the tokenizer loads, the chat template renders.

Three details worth knowing before reading its numbers. It has no `index_topk`, so it takes the dense
MLA path and `TritonMLA` on our cards rather than the sparse backend that rules out V3.2. Its chat
template appends `<think>`, so it reasons by default and will need the same treatment as
Qwen3-30B-A3B-Thinking: gsm8k's 256-token and livebench's 4096-token caps are below what a chain of
thought needs. And its native top-k is 4, which makes `Baseline-k4 / FixedK-k3 / FixedK-k2` the
comparable ladder — the same three rows gpt-oss-20b has.

## The suite's cost is not where the request counts say it is: aime was half of it

Asked when the Thinking model's sixteen configurations would finish, the honest answer needed a model of
what a run has left, and the tool in use answered a different question. lighteval evaluates the nine
datasets as six groups and prints one tqdm bar per group, so `thinking_eta.sh`, which reads the last
bar, reports progress through the *current* group. Read as an ETA that is badly wrong, because the
groups are nowhere near equal and the most expensive one is last.

Measured on Qwen3-30B-A3B-Thinking-2507, relative to the mmlu_pro stage:

| group | requests | cost |
| --- | --- | --- |
| mmlu_pro | 12032 | 1.00 |
| livebench_20241125 | 150 | 0.045 |
| math_500 + gpqa:diamond + lcb:codegeneration_v6 | 873 | 0.225 |
| hellaswag_fixed | 10042 | 0.35 |
| gsm8k | 1319 | 0.035 |
| aime24_avg + aime25_avg | 3840 | 2.34 |

Sixty questions cost more than twelve thousand, because avg@64 samples each of them 64 times and this
model reasons at length before answering, while mmlu_pro is multiple choice and hellaswag is scored by
likelihood. The whole suite is about four times its mmlu_pro stage, so a run that has just finished
mmlu_pro is a quarter done, not five-sixths. `thinking_eta2.py` prices runs this way.

The fix was to sample 16 times instead of 64 (`tasks/aime.py`, commit 9bda568), which cuts the suite to
about 2.25x mmlu_pro. avg@n's reported standard error is computed across the 30 questions, not across
samples, so it barely moves: the Instruct baseline's aime24 stderr was 0.0616 at avg@64, an interval far
wider than anything the sample count contributes.

Two things about applying it to runs already in flight. A run reads the task definitions once, at
startup, so only a new process picks up the change — and restarting one discards everything it has
generated, which made the arithmetic easy for the three runs still inside mmlu_pro: 1.4–2.6 hours lost
each against 22–36 saved.

That second point deserves stating precisely, because it was first written here as "the cache lands when
a group finishes", which is wrong and would make a restart look cheap for a run that has crossed a group
boundary. `@cached` decorates the model's generate call, and lighteval issues *one* such call for the
whole generative pass — the log line naming eleven tasks and 24476 samples is that single call, and the
six progress bars are inside it. The parquet is written when it returns. A run that has finished
mmlu_pro, livebench and math/gpqa/lcb therefore has an empty cache directory, containing only the
task-hash folders, which is exactly what Dynamic Routing thr0.33 looked like after 27 hours. The probe
runs do have cached parquet files, and that is the tell: a probe evaluates one task, so its pass ends.

There is no configuration under which a restart replays anything, either. The cache namespace is a hash
of the whole model config dump, and `tensor_parallel_size` is one of its fields, so the very thing these
restarts are for — moving a run from two cards to four — gives it a fresh namespace.

### Two ways the restart went wrong

`pgrep -f` matched the script doing the searching. With the stop logic inlined into an ssh command, the
remote shell's own cmdline contained every pattern in the script, so it matched itself, killed itself
halfway through, and left the run alive with its lane gone. The stop logic now lives in a file on the
host (`stop_thinking_run.sh`), whose cmdline is a path and four short arguments. Related: `tr -d '\0'`
on `/proc/<pid>/cmdline` deletes the separators and runs the arguments together, so a pattern with a
trailing space never matches; it has to be `tr '\0' ' '`.

The watcher reconciled the half-finished restart. `thinking_watch.sh` sweeps every five minutes for
exactly the state a restart passes through — lane not running, cards free, configuration unclaimed — and
started the lane itself. The relaunch then started a second one, and two runs of thr0.61 ended up on the
same two cards writing to the same directory and the same log file; the second failed on GPU memory,
bumped the retry counter, and re-claimed. Both were cleaned up and `restart_for_avg16.sh` now checks for
a lane of that name before starting one. The general lesson is that any manual intervention on these
lanes has to assume the watcher will act during the window.

## Lowering max_model_len to cap a run's budget feeds the model an empty prompt

The Thinking sweep's two-expert row is capped at 8192 generated tokens, because at that budget the model
stops emitting stop tokens and every answer runs to the limit. Applying the same cap to GLM-4.7-Flash's
k=2 row, it went in as `--max_model_len 8192 --max_new_tokens 8192`, and the log said:

    context_size + max_new_tokens=9292 which is greater than self.max_length=8192.
    Truncating context to 0 tokens.

lighteval compares prompt length plus `max_new_tokens` against the model's context length and
left-truncates the prompt by the difference. When the two are equal the prompt is truncated to nothing
and the run scores the model on empty inputs. The row was stopped four minutes in and restarted with
`--max_new_tokens 8192` alone, leaving the window at 32768 — which is what the Thinking rows had all
along (`max_model_length: 32768`, `max_new_tokens: 8192` in their results JSON, verified before touching
anything). A budget cap belongs on the generation limit only.

## Idle cards are not free cards

A queue lane holds a fixed set of cards for its whole life and sits in a wait loop between
configurations, so between two runs — and for as long as a dying engine takes to release memory —
`nvidia-smi` reports its cards as idle. `glm47_watch.sh` read that as free capacity and started
GLM-4.7-Flash on the four cards the Ban tier was two minutes from reusing. Its gate had been "the host
has idle cards and no unclaimed work in its queues", which misses this case exactly: the tier was
claimed, by the lane that was restarting it. Both this watcher and `upgrade_thr033.sh` now subtract the
cards named in every lane's cmdline from what nvidia-smi calls idle.

The same day, the reverse: `thinking_watch.sh` would not start a lane because one of that name was
already running. Lane names repeat across queues — every host calls its four-card lane `L0_3` — and the
`L0_3` it saw belonged to a different queue, had already released its cards, and was only scoring in the
background. Four cards on .207 waited on a process that was not using them. A lane's identity is its
name *and* its queue, and the check now matches both.

### Three self-inflicted process kills, and what actually works

Killing a run by anything less than its own process tree has now cost time three separate ways:

  * `pgrep -f 'VLLM::EngineCore'` matches every run on the host. Used to clean up after a stopped run, it
    took down the healthy Ban tier next door and discarded the 4.7 hours it had generated, none of which
    was cached.
  * `pkill -f 'sleep 300'` matched the shell that was running it, which ended that shell mid-script and
    left the work after it undone.
  * `pgrep -f <pattern>` inside a command that mentions the pattern matches that command. Bracketing one
    character (`[m]ain.py`) is enough when the check has to stay inline.

What works is collecting the target's children *before* killing the parent — once the parent is gone its
engine processes are indistinguishable from any other run's — and matching cmdlines whole rather than by
substring.

## A watcher that could never have fired

`upgrade_tier2_tp4.sh` waited for two cards to free before restarting the tier-2 row on four. It would
have waited forever: the free-card query was written `nvidia-smi ... || awk '$2+0<1000{...}'`, with a
logical-or where a pipe was meant, so `awk` never ran and the test looked for `" 4 "` inside raw
nvidia-smi output that only ever contains `4,`. Nothing about the failure is visible from outside — the
log says it is waiting, which is what waiting looks like. It is replaced by `upgrade_thr033.sh`, which
also waits on *both* hosts that can host the row rather than only the one next to it, since the config
carries a weights path both have locally.
## Two ways lm_eval refuses a task the nine-dataset QA ladder needs

Adding boolq, piqa, social_iqa, mmlu and hellaswag to the four QA suites already measured turned up two
blockers, both worth knowing before adding a tenth.

`social_iqa` as the harness ships it reads the `social_i_qa` repository, which serves its data through a
loading script. datasets 5.0.1 refuses those outright — "Dataset scripts are no longer supported" — and
`allenai/social_i_qa` is the same repository renamed, so it fails identically. The fix is a task
definition pointing at the parquet mirror `lighteval/siqa`, whose columns (`context`, `question`,
`answerA/B/C`, and a label of "1", "2" or "3") are exactly what the shipped definition already expects,
so nothing about the scoring changes. The other eight load from the Hub as they are; boolq's `super_glue`
path and `winogrande`, both also once script-based, now resolve to converted parquet.

Repo-local task definitions cannot be reached through `TaskManager(include_path=...)`. `get_task_dict`
logs every selected task as a path relative to `lm_eval/tasks`, via `Path.relative_to`, which raises for
a file that came from anywhere else — so the run dies at task-dict construction, after the engine is up
and the datasets are built, about five minutes in. Nothing in the message points at the include path.
`tasks_lm_eval/` therefore stays the source of truth and `lm_eval_runner.py` mirrors it into a directory
*inside* the harness's task tree, where the harness's own recursive scan finds it and its path arithmetic
holds. Mirroring happens per run, so hosts that keep their own checkout stay in step without a
provisioning step.
