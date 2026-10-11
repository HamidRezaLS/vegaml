# Vega benchmark results, 2026-10-11

The long form of the table at the top of the repository README. The run is
[`../notebooks/vega_benchmark_2xT4.ipynb`](../notebooks/vega_benchmark_2xT4.ipynb) and the figure
is drawn from the same JSON by [`make_table_png.py`](make_table_png.py).

One run of the whole menu, finished 2026-10-11 on two Tesla T4s in a Kaggle session.
43 public tasks at up to 60 evaluation rows each (45 to 60, 33 of the 43 at a full 60),
231 JevBench items, 1075 DecisionBench items.
Totals: about 5 h 40 m of GPU time across three sittings, 3 h 55 m of which ended in a
host memory kill, 1502 readouts carried forward into the last sitting, which measured the
4B backbone in 99.4 minutes.
Environment: python 3.13.15, torch 2.11.0+cu128, transformers 5.19.0, vegaml 0.8.0.

Skill is chance corrected, `(accuracy - chance) / (1 - chance)`, so a 77 label task and a
two label task can sit in the same column. Jev is the hosted `jev-latest` endpoint, called
once per row. Vega 0.8B and Vega 4B are `engine` with both public adapters loaded.

## The final table

| metric | Vega 0.8B | Vega 4B | Jev |
| --- | ---: | ---: | ---: |
| **Public menu, 43 tasks, macro average** |  |  |  |
| accuracy | 0.556 | 0.717 | 0.836 |
| skill (chance corrected) | 0.388 | 0.594 | 0.755 |
| macro F1 | 0.486 | 0.649 | 0.796 |
| coverage | 1.000 | 1.000 | 1.000 |
| ECE | 0.151 | 0.148 | 0.115 |
| NLL | 1.099 | 0.768 | 0.878 |
| Brier | 0.550 | 0.384 | 0.242 |
| AURC (lower is better) | 0.362 | 0.169 | 0.077 |
| latency p50, ms | 74 | 106 | 162 |
| **JevBench, 231 items** |  |  |  |
| accuracy | 0.545 | 0.662 | 0.866 |
| skill | 0.334 | 0.505 | 0.803 |
| ECE | 0.178 | 0.169 | 0.067 |
| tier easy (48) | 0.979 | 1.000 | 1.000 |
| tier original (72) | 0.458 | 0.667 | 0.986 |
| tier hard (111) | 0.414 | 0.513 | 0.730 |
| paraphrase consistency (36 groups) | 0.694 | 0.722 | 0.972 |
| **DecisionBench, 1075 items** |  |  |  |
| accuracy | 0.509 | 0.633 | 0.752 |
| skill | 0.362 | 0.524 | 0.677 |
| coverage | 1.000 | 1.000 | 0.997 |
| ECE | 0.103 | 0.098 | 0.122 |
| NLL | 1.628 | 1.420 | 3.196 |
| Brier | 0.638 | 0.496 | 0.398 |
| AURC (lower is better) | 0.291 | 0.157 | 0.140 |
| soft gold KL, 10 rows (lower is better) | 1.012 | 0.854 | 6.099 |

### Abstention and set prediction, where the engines report it

| metric | Vega 0.8B | Vega 4B | Jev |
| --- | ---: | ---: | ---: |
| public abstain rate | 0.555 | 0.255 | - |
| public accuracy when answering | 0.667 | 0.785 | - |
| public conformal coverage (0.9 target) | 0.705 | 0.771 | - |
| public conformal set size | 1.467 | 1.175 | - |
| DecisionBench abstain rate | 0.564 | 0.298 | - |
| DecisionBench accuracy when answering | 0.719 | 0.746 | - |
| DecisionBench conformal coverage | 0.600 | 0.679 | - |

### Per task wins against Jev, public menu

- Vega 0.8B: wins 4, loses 38, ties 1 of 43.
- Vega 4B: wins 12, loses 31, ties 0 of 43.

The backbones were not run on JevBench, so that suite has no backbone column.

### The two backbones, same prompts, no engine

| metric | Qwen3.5-0.8B gen | Qwen3.5-4B gen | Qwen3.5-0.8B logprob | Qwen3.5-4B logprob |
| --- | ---: | ---: | ---: | ---: |
| public accuracy | 0.523 | 0.700 | 0.317 | 0.303 |
| public skill | 0.326 | 0.578 | 0.105 | 0.099 |
| public coverage | 0.983 | 0.951 | 0.905 | 0.834 |
| public parse rate | 0.983 | 0.951 | - | - |
| public latency p50, ms | 201 | 338 | 178 | 547 |
| DecisionBench accuracy | 0.464 | 0.634 | 0.292 | 0.342 |
| DecisionBench coverage | 0.994 | 0.947 | 0.724 | 0.619 |

### Per task accuracy, public menu

| task | labels | chance | majority | Vega 0.8B | Vega 4B | Jev | 0.8B base gen | 4B base gen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aegis_prompt_safety | 2 | 0.500 | 0.600 | 0.500 | 0.600 | 0.900 | 0.733 | 0.717 |
| ag_news | 4 | 0.250 | 0.367 | 0.967 | 0.950 | 0.967 | 0.800 | 0.967 |
| anli_r3 | 3 | 0.333 | 0.400 | 0.267 | 0.350 | 0.683 | 0.383 | 0.450 |
| arc_challenge | 4 | 0.250 | 0.283 | 0.483 | 0.800 | 0.983 | 0.683 | 0.900 |
| arc_easy | 4 | 0.250 | 0.283 | 0.600 | 0.917 | 1.000 | 0.800 | 0.967 |
| banking77 | 77 | 0.013 | 0.067 | 0.683 | 0.717 | 0.817 | 0.233 | 0.350 |
| boolq | 2 | 0.500 | 0.767 | 0.767 | 0.817 | 0.917 | 0.783 | 0.883 |
| casehold | 5 | 0.200 | 0.267 | 0.350 | 0.550 | 0.800 | 0.583 | 0.700 |
| climate_fever | 4 | 0.250 | 0.383 | 0.383 | 0.450 | 0.533 | 0.200 | 0.400 |
| clinc150_oos | 151 | 0.007 | 0.200 | 0.683 | 0.900 | 0.883 | 0.083 | 0.100 |
| commonsenseqa | 5 | 0.200 | 0.233 | 0.350 | 0.533 | 0.900 | 0.533 | 0.767 |
| cruxeval_output | 4 | 0.250 | 0.383 | 0.650 | 0.850 | 1.000 | 0.800 | 0.983 |
| dbpedia14 | 14 | 0.071 | 0.100 | 0.600 | 0.917 | 0.983 | 0.717 | 0.983 |
| emotion | 6 | 0.167 | 0.433 | 0.550 | 0.650 | 0.617 | 0.483 | 0.567 |
| enron_spam | 2 | 0.500 | 0.517 | 1.000 | 1.000 | 0.950 | 0.483 | 0.917 |
| glue_rte | 2 | 0.500 | 0.550 | 0.633 | 0.883 | 0.850 | 0.550 | 0.733 |
| hellaswag | 4 | 0.250 | 0.283 | 0.250 | 0.650 | 0.900 | 0.433 | 0.800 |
| jailbreak_cls | 2 | 0.500 | 0.569 | 0.983 | 1.000 | 0.966 | 0.586 | 0.966 |
| ledgar | 100 | 0.010 | 0.083 | 0.333 | 0.650 | 0.700 | 0.117 | 0.183 |
| massive_intent_en | 60 | 0.017 | 0.067 | 0.383 | 0.600 | 0.850 | 0.183 | 0.367 |
| medmcqa | 4 | 0.250 | 0.317 | 0.183 | 0.467 | 0.717 | 0.350 | 0.650 |
| medqa_usmle | 4 | 0.250 | 0.356 | 0.288 | 0.491 | 0.780 | 0.491 | 0.644 |
| mmlu | 4 | 0.250 | 0.283 | 0.350 | 0.717 | 0.900 | 0.483 | 0.650 |
| mmlu_pro | 10 | 0.100 | 0.183 | 0.200 | 0.383 | 0.900 | 0.250 | 0.517 |
| openbookqa | 4 | 0.250 | 0.383 | 0.583 | 0.833 | 0.967 | 0.483 | 0.833 |
| prompt_injections | 2 | 0.500 | 0.582 | 0.746 | 0.855 | 0.764 | 0.455 | 0.564 |
| pubmedqa | 3 | 0.333 | 0.550 | 0.133 | 0.400 | 0.867 | 0.650 | 0.583 |
| race_middle | 4 | 0.250 | 0.317 | 0.600 | 0.817 | 0.983 | 0.700 | 0.900 |
| sciq | 4 | 0.250 | 0.300 | 0.733 | 0.933 | 0.983 | 0.800 | 0.983 |
| superglue_cb | 3 | 0.333 | 0.533 | 0.667 | 0.822 | 0.800 | 0.444 | 0.867 |
| toxic_chat_jailbreak | 2 | 0.500 | 1.000 | 0.983 | 0.917 | 1.000 | 1.000 | 1.000 |
| toxic_chat_toxicity | 2 | 0.500 | 0.967 | 0.950 | 1.000 | 0.983 | 0.933 | 0.983 |
| trec_coarse | 6 | 0.167 | 0.300 | 0.533 | 0.767 | 0.900 | 0.483 | 0.883 |
| truthfulqa_mc1 | 5 | 0.200 | 1.000 | 0.360 | 0.620 | 0.940 | 0.440 | 0.660 |
| tweet_hate | 2 | 0.500 | 0.633 | 0.650 | 0.683 | 0.783 | 0.417 | 0.667 |
| tweet_irony | 2 | 0.500 | 0.586 | 0.776 | 0.776 | 0.810 | 0.414 | 0.793 |
| tweet_offensive | 2 | 0.500 | 0.695 | 0.780 | 0.780 | 0.729 | 0.661 | 0.661 |
| tweet_sentiment | 3 | 0.333 | 0.525 | 0.424 | 0.593 | 0.525 | 0.508 | 0.576 |
| tweet_stance_abortion | 3 | 0.333 | 0.673 | 0.364 | 0.545 | 0.436 | 0.545 | 0.364 |
| unfair_tos | 2 | 0.500 | 0.842 | 0.737 | 0.877 | 0.667 | 0.526 | 0.807 |
| vitaminc | 3 | 0.333 | 0.450 | 0.467 | 0.617 | 0.800 | 0.333 | 0.633 |
| winogrande | 2 | 0.500 | 0.533 | 0.500 | 0.567 | 0.900 | 0.500 | 0.617 |
| yelp_stars | 5 | 0.200 | 0.283 | 0.500 | 0.567 | 0.600 | 0.467 | 0.567 |


## What the table says

**Jev wins the menu.** 0.836 accuracy against 0.717 for Vega 4B and 0.556 for Vega 0.8B, and
it wins 31 of 43 public tasks against the 4B, 38 of 43 against the 0.8B. On JevBench its
hard tier is 0.730 against 0.513, and its paraphrase consistency is 0.972 against 0.722: ask
it the same question three ways and it answers the same way. That is the single widest gap in
the run and it is not a knowledge gap, it is a stability gap.

**Vega wins calibration on DecisionBench.** ECE 0.098 for the 4B against 0.122, NLL 1.420
against 3.196, soft gold KL 0.854 against 6.099. Jev is accurate and overconfident: mean
confidence 0.874 on rows it gets right 0.752 of the time. When a row has a distribution
rather than a single gold answer, Vega is far closer to it. Jev also returns a bare
`max_tokens_exceeded` on 3 of 1075 rows, which is where its 0.997 coverage comes from.

**The twelve tasks the 4B wins are mostly short social and safety text, and almost none of
them hold up.** Nine of the twelve are worth three rows or fewer, and the two widest both
lose to their own majority class. `prompt_injections` is the only one left standing.
"Where Vega surpassed Jev" below works through all twelve with row counts. `engine_no_adapter`
matches `engine` on all of them except `clinc150_oos`, which falls from 0.900 to 0.517, so the
adapters carry exactly one task.

**The 4B backbone, prompted correctly, nearly matches the 4B engine.** 0.700 against 0.717 on
the menu, 0.634 against 0.633 on DecisionBench, and it wins 24 of 43 public tasks head to
head. It wins the reading and knowledge tasks (`commonsenseqa` 0.767 against 0.533,
`medmcqa` 0.650 against 0.467, `pubmedqa` 0.583 against 0.400) and loses the many label
classification tasks badly (`clinc150_oos` 0.100 against 0.900, `ledgar` 0.183 against 0.650,
`banking77` 0.350 against 0.717). What the engine adds over a plain generate call is not
accuracy in aggregate: it is 1.000 coverage against 0.947, a probability on every row, an
ECE, an abstention, and a third of the latency. Those matter, but the accuracy claim does not
survive this column, and this column only exists because the chat template bug was fixed.

**The 0.8B engine is worth it against its own backbone only on calibration.** 0.556 against
0.523 generate, a 21-21-1 split across tasks. Against `logprob` scoring it is 30-10-3, which
is the comparison the earlier draft used and the weaker claim.

**Two unexplained weaknesses.** Conformal coverage is 0.771 for the 4B against a 0.900 target,
0.705 for the 0.8B, so the prediction sets are too small and the calibration that produces
them is miscalibrated in the tail. And reversing the option order flips the answer on 0.43 of
`ledgar` rows for the 4B, 0.70 for the 0.8B, with `casehold` at 0.50 and `mmlu` at 0.40. Both
of these are real findings against Vega, not noise, and both sit on the same axis Jev wins by.

**Where Vega leads on a slice.** JevBench `family:temporal_numeric`, 0.467 for the 0.8B and
0.400 for the 4B against 0.200 for Jev on 15 items, and `engine_no_adapter` at 0.333 for both,
so the adapters are doing something on dates and arithmetic. Two DecisionBench domains follow,
at one and two rows. Small n, worth a look, not worth a headline.

## Where Vega surpassed Jev

Three different kinds of win, with very different weight behind them.

### 1. Calibration and cost on DecisionBench, 1075 rows

These are the wins worth making. They are measured on the whole suite, not a slice, and the
margins are wide.

| metric | Vega 0.8B | Vega 4B | Jev | winner |
| --- | ---: | ---: | ---: | --- |
| ECE | 0.103 | 0.098 | 0.122 | both Vegas |
| NLL | 1.628 | 1.420 | 3.196 | both Vegas |
| soft gold KL, 10 rows | 1.012 | 0.854 | 6.099 | both Vegas |
| coverage | 1.000 | 1.000 | 0.997 | both Vegas |
| latency p50, ms | 85 | 245 | 163 | 0.8B |

Jev is accurate and overconfident: mean confidence 0.874 against 0.752 accuracy, which is what
drives an NLL of 3.196 against 1.420. On the ten rows that carry a gold distribution rather
than a single gold answer, its KL is 6.099 against 0.854. It also returns a bare
`max_tokens_exceeded` on 3 of 1075 rows, which is the whole of its coverage shortfall.

On the public menu the only metric win is NLL for the 4B, 0.768 against 0.878, and it holds on
18 of 43 tasks. Latency: the 0.8B is faster on 40 of 43 tasks, the 4B on 28 of 43. Jev wins
public ECE, Brier and AURC, and wins every calibration metric on JevBench.

### 2. Accuracy on public tasks, and why eleven of the twelve do not hold up

`rows+` is the net number of rows the win is worth. `p floor` is the smallest p a two sided
sign test could give it, assuming the most favourable pairing possible, which is `2^(1-rows+)`.
The per row data was not kept, so the true p is higher than this for every line.

| task | rows | best Vega | maj | Vega | Jev | delta | rows+ | vs maj | p floor |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unfair_tos | 57 | 4B | 0.842 | 0.877 | 0.667 | +0.211 | 12 | +2 | 0.0005 |
| tweet_stance_abortion | 55 | 4B | 0.673 | 0.545 | 0.436 | +0.109 | 6 | -7 | 0.031 |
| prompt_injections | 55 | 4B | 0.582 | 0.855 | 0.764 | +0.091 | 5 | +15 | 0.062 |
| tweet_sentiment | 59 | 4B | 0.525 | 0.593 | 0.525 | +0.068 | 4 | +4 | 0.125 |
| tweet_offensive | 59 | both | 0.695 | 0.780 | 0.729 | +0.051 | 3 | +5 | 0.250 |
| enron_spam | 60 | both | 0.517 | 1.000 | 0.950 | +0.050 | 3 | +29 | 0.250 |
| jailbreak_cls | 58 | 4B | 0.569 | 1.000 | 0.966 | +0.034 | 2 | +25 | 0.500 |
| emotion | 60 | 4B | 0.433 | 0.650 | 0.617 | +0.033 | 2 | +13 | 0.500 |
| glue_rte | 60 | 4B | 0.550 | 0.883 | 0.850 | +0.033 | 2 | +20 | 0.500 |
| superglue_cb | 45 | 4B | 0.533 | 0.822 | 0.800 | +0.022 | 1 | +13 | 1.000 |
| clinc150_oos | 60 | 4B | 0.200 | 0.900 | 0.883 | +0.017 | 1 | +42 | 1.000 |
| toxic_chat_toxicity | 60 | 4B | 0.967 | 1.000 | 0.983 | +0.017 | 1 | +2 | 1.000 |

Reading down the `rows+` column: nine of the twelve wins are three rows or fewer, so no
pairing of the data could make them significant. The two that could are the two with a
baseline problem. `unfair_tos` is the widest win in the run at 12 rows, but Vega sits only 2
rows above its own majority class while Jev sits 10 rows below it, so the finding is that Jev
collapses on that task, not that Vega is good at it. `tweet_stance_abortion` is 7 rows below
its majority class, so both systems lose to answering the same thing every time.

That leaves **`prompt_injections` as the one honest accuracy win**: 0.855 against 0.764, 15
rows above a 0.582 majority class, 5 net rows ahead of Jev at a p floor of 0.062. Close to the
line and worth repeating at a larger n, not yet a claim.

The 0.8B beats Jev on four tasks, all of them in this list: `unfair_tos`, `tweet_offensive`,
`enron_spam`, `jailbreak_cls`. It never wins a task the 4B loses.

Everything in this list is short social or safety text except `clinc150_oos`, which is the
one task the adapters carry: `engine_no_adapter` drops it from 0.900 to 0.517.

### 3. Slices, 15 to 25 rows each

| suite | slice | n | Vega 0.8B | Vega 4B | Jev | net rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| JevBench | family:temporal_numeric | 15 | 0.467 | 0.400 | 0.200 | 4 |
| DecisionBench | domain:public_sector | 25 | 0.760 | 0.960 | 0.880 | 2 |
| DecisionBench | domain:privacy | 25 | 0.880 | 1.000 | 0.960 | 1 |

`temporal_numeric` is the interesting one: both Vegas more than double Jev on dates and
arithmetic, and `engine_no_adapter` scores 0.333 for both, so the adapters are doing something
there. Four net rows out of fifteen is a hint to chase with a bigger slice, nothing more. The
two DecisionBench domains are one and two rows and should not be quoted.

### What this adds up to

Vega does not surpass Jev on accuracy anywhere that survives a careful look, with
`prompt_injections` the single candidate. It surpasses Jev on honest probabilities, on
DecisionBench by a wide and well powered margin, on full coverage, and on latency at the 0.8B
size. That is the claim the data supports.

## Notes on reading it

- A public task is at most 60 rows, so a single row moves it by 0.017 and any per task delta
  below about 0.05 is inside the noise. Suite level macro averages are steadier, but they are
  still 43 numbers each.
- **Three tasks a constant answer wins.** `truthfulqa_mc1` has a majority accuracy of 1.000:
  upstream `mc1_targets` always puts the true answer first and this task does not shuffle its
  options, so gold is "A" on all 50 rows and always answering A scores 1.000 against 0.940 for
  Jev and 0.620 for Vega 4B. That column measures position bias, not truthfulness, and it
  should be fixed by routing the task through the same `_shuffled` helper `sciq` uses in
  the notebook's `bench_tasks.py`.
  `toxic_chat_jailbreak` drew 60 rows that are all the negative class, so a constant answer
  scores 1.000 there too, and `toxic_chat_toxicity` is at 0.967. Read all three as ceilings.
- `tweet_stance_abortion` has a 0.673 majority class that beats both systems, so the
  +0.109 the 4B takes off Jev there is two systems losing to a constant answer by different
  amounts.
- Zero truncation collisions on all 43 tasks: no task's options were cut to the point of
  being indistinguishable.
- `base-*/logprob` sits near chance because the scoring ran out of GPU memory on long rows:
  coverage 0.834 for the 4B on the menu, 0.619 on DecisionBench. Treat those two columns as
  a floor, not a measurement.
- The engines' abstention is reported but never acted on: accuracy is computed on every row.
  Vega 4B abstains on 0.255 of public rows and is right 0.785 of the time on the rest.
