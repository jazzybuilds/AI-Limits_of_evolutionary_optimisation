#set page(numbering: "1")

#set text(lang: "en", region: "GB")

#let make-venue = move(dy: -1.9cm, {
  box(rect(fill: blue.darken(30%), inset: 10pt, height: 2.5cm)[
    #set text(font: "TeX Gyre Pagella", fill: white, weight: 700, size: 20pt)
    #align(bottom)[University of Sussex]
  ])
  set text(22pt, font: "TeX Gyre Heros")
  box(pad(left: 10pt, bottom: 10pt, [Artificial Life]))
})

#let make-title(
  title,
  authors,
  abstract,
  keywords,
) = {
  set par(spacing: 1em)
  set text(font: "TeX Gyre Heros")
  
  par(
    justify: false,
    text(24pt, fill: rgb("004b71"), title, weight: "bold")
  )

  text(12pt,
    authors.enumerate()
    .map(((i, author)) => box[#author.name #super[#(i+1)]])
    .join(", ")
  )
  parbreak()

  for (i, author) in authors.enumerate() [
    #set text(8pt)
    #super[#(i+1)]
    #author.institution
    #link("mailto:" + author.mail) \
  ]

  v(8pt)
  set text(10pt)
  set par(justify: true)

  [
    #heading(outlined: false, bookmarked: false)[Abstract]
    #text(font: "TeX Gyre Pagella", abstract)
    #v(3pt)
    *Keywords:* #keywords.join(text(font: "TeX Gyre Pagella", "; "))
  ]
  v(18pt)
}

#let template(
    title: [],
    authors: (),
    date: [],
    doi: "",
    keywords: (),
    abstract: [],
    make-venue: make-venue,
    make-title: make-title,
    body,
) = {
    set page(
      paper: "a4",
      margin: (top: 1.9cm, bottom: 1in, x: 1.6cm),
      columns: 2
    )
    set par(justify: true)
    set text(10pt, font: "TeX Gyre Pagella")
    set list(indent: 8pt)
    // show link: set text(underline: false)
    show heading: set text(size: 11pt)
    show heading.where(level: 1): set text(font: "TeX Gyre Heros", fill: rgb("004b71"), size: 12pt)
    show heading: set block(below: 8pt)
    show heading.where(level: 1): set block(below: 12pt)

    place(make-venue, top, scope: "parent", float: true)
    place(
      make-title(title, authors, abstract, keywords), 
      top, 
      scope: "parent",
      float: true
    )


    show figure: align.with(center)
    show figure: set text(8pt)
    show figure.caption: pad.with(x: 10%)

    // show: columns.with(2)
    body
  }

#show: template.with(
  
  title: [From Coarse to Refined Control: Exploring the Limits of Evolutionary Optimisation],
  authors: (
    (
      name: "Jason Fang (309734)",
      department: "Informatics",
      institution: "University of Sussex",
      city: "Brighton & Hove",
      country: "UK",
      mail: "",
    ),
  ),
  date: (
    year: 2022,
    month: "May",
    day: 17,
  ),
  keywords: (
    "Neuroevolution",
    "Continuous-Time Recurrent Neural Networks",
    "Neutrality",
    "Action-Space Representation",
    "Evolutionary Optimisation",
  ),
  doi: "10.7891/120948510",
  abstract: [
   Neutral mutations — bit-flips with no phenotypic effect — are hypothesised to
promote evolvability by sustaining genetic diversity and enabling neutral drift
towards novel fitness peaks. We test this hypothesis by evolving Continuous-Time
Recurrent Neural Network (CTRNN) controllers across tasks of increasing
difficulty: Acrobot, MountainCarContinuous, LunarLander (discrete), and
LunarLander (continuous). With 75% genome neutrality, evolution solves the two
simpler tasks (fitness 0.84 and 0.94) and generalises across held-out physics
variants. Discrete LunarLander plateaus at fitness ≈0.25 across all encoding
conditions tested — baseline ($p$=0.75), no neutrality ($p$=0.0), and
real-valued encoding — confirming the failure is a property of the problem
representation, not the genome. We identify the primary barrier as a
mismatch between the smooth policy class (CTRNN tanh outputs) and the
discontinuous action interface (argmax): incremental weight improvements are
behaviourally invisible until an argmax boundary is crossed. Switching to the continuous action variant — which
accepts tanh outputs directly — and extending the evolutionary budget to 200
generations, the same GA achieves fitness *0.91*, clearing the task-solved
threshold on both training and held-out test physics. This confirms the
landscape is navigable given a representation aligned with the controller's
output structure. We discuss implications for neutrality at small population
sizes, and motivate plasticity as a route to more sample-efficient and robust
control.
  ],
)  



= 1 Introduction

A central challenge in neuroevolution is premature convergence: the tendency of
a small population to fix on the first individual that outperforms its
contemporaries, foreclosing exploration. Classical selection pressure is the
culprit — in a population of twenty, tournament selection with size three can
replicate a single winner across the entire population in fewer than ten
generations, transforming the search into a local hill-climber with no memory
of discarded alternatives.

One proposed remedy draws on the theory of neutral evolution. In biology,
the majority of observed mutations are selectively neutral; they accumulate in
the genome without altering fitness. Neutral drift allows a lineage to wander
through genotype space while its phenotype remains stable, increasing the
probability of encountering a mutation that is beneficial in the current
environment or pre-adapted to a future one @Kimura1983NeutralEvolution. Translated to
artificial evolution, embedding non-coding bits in the genome — bits that
encode no network weight — should produce an analogous effect: the population
diffuses through a larger neutral neighbourhood, maintaining diversity and
sustaining discrete lineages even under strong directional selection
@Fang2026NeutralityNK @Barnett2001NetcrawlingNetworks.

Whether this theoretical benefit materialises in practice depends critically on
the difficulty of the target task. Simple control problems with smooth fitness
landscapes require only a brief directed search; the overhead of a large neutral
genome may simply slow convergence without improving the final solution.
Harder problems, where fitness improvements are sparse and deceptive, are the
natural test-bed for the neutrality hypothesis.

This paper reports a systematic evaluation across three environments. The two
simpler problems, Acrobot and MountainCarContinuous, are used to validate that
the evolutionary framework functions correctly and generalises. LunarLander
serves as the hard case: a stochastic, multi-effector control problem where
naive evolutionary search is expected to struggle. Our aim is to characterise
*why* it struggles — distinguishing between failures of the encoding, the
search operator, the selection mechanism, and the landscape itself — and to
connect those findings to the theoretical motivation for introducing plasticity
in Stage 2.

= 2 Methods


== Controller Architecture

Controllers were CTRNN neurons with 8 recurrent hidden units @Beer1995CTRNNDynamics. Each neuron $i$
integrates according to:

$ tau_i dot(y)_i = -y_i + sum_j w_(j i) sigma(y_j + theta_j) + I_i $

where $tau_i$ is the time constant, $w_(j i)$ the recurrent weight, $theta_j$
a bias, $sigma$ the logistic function, and $I_i$ the external input at
timestep $t$. Output units were softmax-normalised for discrete actions
(Acrobot, LunarLander) or tanh-squashed for continuous force
(MountainCarContinuous, LunarLanderContinuous). Integration used Euler steps
with $Delta t = 0.2$.

== Genome Encoding

CTRNN weights, biases, and time constants were packed into a flat binary
genome using Gray coding with 16 bits per parameter, giving a resolution of
$approx 0.0001$ (e.g. $approx 0.0001$ over $[-3, 3]$ for weights; ranges
differ per parameter type — see @tbl-genome). Gray coding ensures that
single-bit mutations produce maximally small parameter perturbations, locally
preserving the neutral neighbourhood structure.

Neutral bits were appended after the active (weight-encoding) region. With
neutrality fraction $p$, the total genome length is:

$ N = N_"active" / (1 - p) $

so that at $p = 0.75$, three bits in four carry no information. Mutations in
the neutral region have zero effect on the decoded network. The active region
scales with the environment's observation and action dimensions; @tbl-genome
shows the resulting genome sizes across all four environments.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, center, center, center, center),
    table.header(
      [*Parameter group*], [*Range*], [*Acrobot*], [*MCC*], [*LL*], [*LLC*],
    ),
    [Obs. dim],                        [],              [6],     [2],     [8],      [8],
    [Act. dim],                        [],              [3],     [1],     [4],      [2],
    table.hline(stroke: 0.4pt),
    [$W$ (recurrent, $8 times 8$)],    [$[-3, 3]$],     [64],    [64],    [64],     [64],
    [$W_"in"$ (input)],                [$[-3, 3]$],     [48],    [16],    [64],     [64],
    [$W_"out"$ (output)],              [$[-3, 3]$],     [24],    [8],     [32],     [16],
    [bias],                            [$[-2, 2]$],     [8],     [8],     [8],      [8],
    [$tau$ (time constant)],           [$[0.1, 2.0]$],  [8],     [8],     [8],      [8],
    table.hline(stroke: 0.4pt),
    [*Active params (total)*],         [],              [*152*], [*104*], [*176*],  [*160*],
    [Active bits ($times 16$)],        [],              [2,432], [1,664], [2,816],  [2,560],
    [Total bits ($p$=0.75)],           [],              [9,728], [6,656], [11,264], [10,240],
  ),
  caption: [CTRNN parameter breakdown and genome sizes per environment. LL = LunarLander-v3, LLC = LunarLanderContinuous-v3, MCC = MountainCarContinuous-v0. Ranges are fixed across all environments.],
) <tbl-genome>

The value $p = 0.75$
was chosen based on prior NK landscape experiments demonstrating that this
level maximises population genetic diversity through an optimal balance between
neutral exploration and selective maintenance of lineage structure
@Fang2026NeutralityNK.

== Fitness Function

Fitness was a weighted combination of training and generalisation performance:

$ f = 0.45 dot f_"train" + 0.55 dot f_"gen" $

Training performance was measured on environments with held-in physics
parameters; generalisation performance on held-out variants never seen during
evolution. Each individual was evaluated over $N_"seeds" times N_"configs"
times N_"episodes" = 3 times 2 times 2 = 12$ episodes per condition to
estimate expected performance.

For Acrobot, the maximum tip height $h$ reached during the episode (range
$[-2, +1]$) was mapped via:

$ f_"episode" = "clip"((h + 2) / 4,~0,~1) $

so that the chain hanging straight down gives 0.0 and reaching the goal height
gives $approx 0.75$.

For MountainCarContinuous, the maximum car position $x$ reached during the
episode (range $[-1.2, +0.6]$) was mapped via:

$ f_"episode" = "clip"((x + 1.2) / 1.8,~0,~1) $

so that the car stuck at the left wall gives 0.0 and reaching the flag
($x = 0.45$) gives $approx 0.917$.

For LunarLander, per-episode reward $r$ was mapped to $[0,1]$ via:

$ f_"episode" = "clip"((r + 100) / 300,~0,~1) $

so that a crash corresponds to 0.0 and a bare successful landing (the +100
landing bonus alone, $r = 100$) corresponds to $approx 0.67$.



== Genetic Algorithm

The GA used here extends the NKp framework from @Fang2026NeutralityNK, which
demonstrated that intermediate neutrality sustains population diversity on
epistatic fitness landscapes. The two key properties carried over are:
neutrality (non-coding bits that buffer mutations from phenotypic effect) and
epistasis (fitness contributions that depend on interactions between loci,
captured here by the recurrent connectivity of the CTRNN) @Kauffman1993OriginsEvolution. Applied to
reinforcement learning control, a steady-state GA with tournament selection
(size 3) and bit-flip mutation (rate 0.005 per bit) was used throughout.
Population size was 20; runs lasted 50 generations for all discrete-action
tasks and 200 generations for LunarLanderContinuous, with 2 independent
replicates per configuration.

== Experimental Conditions

Six runs were conducted in total, spanning three environments and varying the
encoding and neutrality level @Towers2024Gymnasium. @tbl-conditions summarises all conditions; runs
marked as ablation studies hold all other factors constant relative to the
LunarLander discrete baseline (Test 3).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, left, center, center, left),
    table.header(
      [*Test*], [*Environment*], [*Encoding*], [*p*], [*Gens*], [*Role*],
    ),
    [1], [Acrobot-v1],                  [binary, Gray],  [0.75], [50],  [baseline],
    [2], [MountainCarContinuous-v0],    [binary, Gray],  [0.75], [50],  [baseline],
    [3], [LunarLander-v3 (discrete)],   [binary, Gray],  [0.75], [50],  [baseline],
    [4], [LunarLander-v3 (discrete)],   [binary, Gray],  [0.0],  [50],  [ablation — no neutrality],
    [5], [LunarLander-v3 (discrete)],   [real-valued],   [—],    [50],  [ablation — no Gray coding],
    [6], [LunarLanderContinuous-v3],    [binary, Gray],  [0.75], [200], [continuous action space],
  ),
  caption: [All experimental conditions. Each run used 2 independent replicates.],
) <tbl-conditions>

Tests 4 and 5 are ablations on the discrete LunarLander baseline (Test 3), each
varying a single factor:

- *Test 4 (no neutrality)*: $p = 0.0$, binary encoding. Removes the neutral
  padding while keeping the same Gray-coded representation. Genome shrinks
  from 11,264 to 2,816 bits.
- *Test 5 (real-valued encoding)*: Genome is a vector of 176 floats in $[0,1]$,
  linearly mapped to weight ranges. Gray coding and neutral padding are not
  applicable; $p$ is ignored. Mutation is additive Gaussian noise
  ($sigma = 0.05$).

Diversity was measured as mean pairwise Hamming distance across the population.
Genomes were snapshotted at generations 0, 10, 20, 30, 40, and 50 (and every
25 generations up to 200 for LunarLanderContinuous) to track active-bit drift
independently of neutral-bit drift.

Full source code is located at: https://github.com/jazzybuilds/AI-Limits_of_evolutionary_optimisation

= 3 Results

== Acrobot and MountainCarContinuous (MCC)

Both tasks were solved convincingly. Acrobot reached a best fitness of *0.843*
(@fig-acrobot-fitness). The best individual generalised to all four held-out
physics variants (modified link lengths, masses, and moments of inertia), as
shown in the state trajectories (@fig-acrobot-traj). Hamming diversity declined
gradually from ~4,870 to ~350–790 over 50 generations (@fig-acrobot-diversity),
consistent with progressive but not premature convergence. Critically, fitness
continued to improve after the initial diversity collapse — the post-sweep
population, though genetically narrow, could still locate improvements because
the fitness landscape is smooth and every weight mutation produces a
proportional fitness change.

#figure(
  image("evolution_fitness_stage1_acrobot_ctrnn_1773892874.png",
        width: 100%),
  caption: [Test 1 — Acrobot fitness curves over 50 generations. Lines show
  best-in-generation and mean fitness per replicate; shading covers the
  replicate range.],
) <fig-acrobot-fitness>

#figure(
  image("evolution_diversity_stage1_acrobot_ctrnn_1773892874.png",
        width: 100%),
  caption: [Test 1 — Mean pairwise Hamming distance over 50 generations for
  Acrobot.],
) <fig-acrobot-diversity>

#figure(
   image("evolution_trajectories_stage1_acrobot_ctrnn_1773892874.png",
        width: 100%),
  caption: [Test 1 — Acrobot state trajectories at selected generation
  snapshots for the best replicate. Tip height increases reaching the goal region (height $>= 1.0$) by the gen 10
  snapshot.],
) <fig-acrobot-traj>

MountainCarContinuous achieved best fitness *0.944* (@fig-mcc-fitness). Both
replicates were within 0.001 of each other (0.9438, 0.9433), confirming strong
reproducibility. Notably, the best individual at generation 0 already achieves
non-trivial fitness (~0.4–0.5): this is expected because the fitness function
rewards maximum car position reached during the episode, not task completion.
A random CTRNN will oscillate the car back and forth unpredictably, and by
chance a random genome may produce forward-biased thrust that moves the car
further right than the initial position — enough to score above 0 under
$f = "clip"((x + 1.2) / 1.8, 0, 1)$ even without reaching the flag. The
population then rapidly improves from this non-zero baseline as selection
locks onto oscillation strategies that sustain rightward momentum. Diversity
followed a similar smooth decline (@fig-mcc-diversity; ~3,330 to ~275–825 by
generation 10), with the same rapid sweep–recovery pattern as Acrobot. State
trajectories show the car progressively building momentum to reach the flag
(@fig-mcc-traj). Both results validate that the framework is functioning
correctly: the CTRNN has sufficient capacity, the fitness signal is
discriminative, and 50 generations at population 20 is a viable budget for
tasks with smooth fitness landscapes.

#figure(
  image("evolution_fitness_stage1_mountaincarcontinuous_ctrnn_1773893100.png",
        width: 100%),
  caption: [Test 2 — MountainCarContinuous fitness curves over 50 generations.],
) <fig-mcc-fitness>

#figure(
  image("evolution_diversity_stage1_mountaincarcontinuous_ctrnn_1773893100.png",
        width: 100%),
  caption: [Test 2 — Mean pairwise Hamming distance over 50 generations for
  MountainCarContinuous.],
) <fig-mcc-diversity>

#figure(
  image("evolution_trajectories_stage1_mountaincarcontinuous_ctrnn_1773893100.png",
        width: 100%),
  caption: [Test 2 — MountainCarContinuous state trajectories at selected
  generation snapshots.],
) <fig-mcc-traj>

== LunarLander — Discrete Baseline

The $p = 0.75$ run reached a best fitness of *0.2507* (reward $approx -25$),
as shown in @fig-ll-fitness. Mean pairwise Hamming distance collapsed from
~5,630 at generation 0 (the expected value for a random binary genome of 11,264
bits) to ~488 by generation 10 — a near-complete genetic sweep
(@fig-ll-diversity). However, Hamming then partially *recovered*, fluctuating
up to ~1,010–1,560 by generation 50. This recovery is consistent with neutral
drift: after fixation of the best individual, the 8,448 non-coding bits have no
fitness consequence and freely accumulate random mutations, causing the
population to spread through genotype space without any phenotypic change. The
post-sweep recovery is visibly larger than in the all-active cases (Acrobot,
MCC), consistent with neutral drift accumulating in the non-coding bits.
Fitness remained plateaued throughout — the population was phenotypically
trapped while genotypically drifting, as confirmed by the stationary descent
trajectories across all generation snapshots (@fig-ll-traj). Action sequences
further confirm the failure mode: the discrete controller rarely sustains main
thruster firing for consecutive timesteps — the argmax winner flips too
infrequently to achieve controlled braking, leaving the lander in an
uncontrolled descent (@fig-ll-actions).

Notably, two independent replicates with different random seeds produced
genomes with ~50% active-bit Hamming distance from each other — maximally
different genotypes — yet converged to the same fitness ceiling. This rules out
a single unlucky initialisation as the explanation.

#figure(
  image("evolution_fitness_stage1_lunarlander_ctrnn_1773897161.png",
        width: 100%),
  caption: [Test 3 — LunarLander (discrete, $p$=0.75) fitness curves over 50
  generations. Fitness plateaus near 0.25 after the initial sweep with no
  further improvement.],
) <fig-ll-fitness>

#figure(
  image("evolution_diversity_stage1_lunarlander_ctrnn_1773897161.png",
        width: 100%),
  caption: [Test 3 — Mean pairwise Hamming distance for the discrete
  LunarLander baseline ($p$=0.75). The post-sweep recovery to ~1,010–1,560 bits
  reflects neutral drift in the 8,448 non-coding bits.],
) <fig-ll-diversity>

#figure(
  image("evolution_trajectories_stage1_lunarlander_ctrnn_1773897161.png",
        width: 100%),
  caption: [Test 3 — LunarLander (discrete, $p$=0.75) state trajectories at
  selected generation snapshots. Descent behaviour shows little qualitative
  change across generations, consistent with fitness plateau.],
) <fig-ll-traj>

#figure(
  image("evolution_actions_stage1_lunarlander_ctrnn_1773897161.png",
        width: 100%),
  caption: [Test 3 — LunarLander (discrete, $p$=0.75) action sequences at
  selected generation snapshots. Each row shows the argmax action selected at
  each timestep (0 = no action, 1 = left thruster, 2 = main engine,
  3 = right thruster). Main-engine firing (action 2) maintains slow switching across all generations, preventing sustained braking.],
) <fig-ll-actions>

== Ablation Tests

*Test 4 (no neutrality, $p = 0.0$)* produced a best fitness of *0.2250*
(@fig-ll-nn-fitness). Hamming collapsed from ~1,410 (≈50% of 2,816 active bits)
to only ~75–95 by generation 5 — even faster than the baseline, because the
smaller genome means tournament selection sweeps it in fewer generations
(@fig-ll-nn-diversity). Hamming then partially recovered to ~140–180 by
generation 50. The recovery is smaller than in the baseline, consistent with
fewer freely-drifting bits: all bits encode weights, so only positionally
neutral mutations (where the bit flip produces no change to the decoded
Gray-code value) can accumulate without selection pressure. Fitness improved
slowly and continuously rather than freezing, but a quantitatively similar ~0.22 ceiling was
reached (@fig-ll-nn-traj).

#figure(
  image("evolution_fitness_stage1_lunarlander_ctrnn_test0_no_neutrality_1773943600.png",
        width: 100%),
  caption: [Test 4 — LunarLander (discrete, $p$=0.0) fitness curves over 50
  generations. A quantitatively similar ~0.22 ceiling is reached as in the baseline despite
  the absence of neutral padding.],
) <fig-ll-nn-fitness>

#figure(
  image("evolution_diversity_stage1_lunarlander_ctrnn_test0_no_neutrality_1773943600.png",
        width: 100%),
  caption: [Test 4 — Mean pairwise Hamming distance with $p$=0.0. The faster
  initial collapse and reduced post-sweep recovery (to ~140–180 bits) reflect
  the absence of freely-drifting non-coding bits.],
) <fig-ll-nn-diversity>

#figure(
  image("evolution_trajectories_stage1_lunarlander_ctrnn_test0_no_neutrality_1773943600.png",
        width: 100%),
  caption: [Test 4 — LunarLander (no neutrality) state trajectories at
  selected generation snapshots.],
) <fig-ll-nn-traj>

*Test 5 (real-valued encoding)* produced a best fitness of *0.2216*
(@fig-ll-rv-fitness). The Hamming diversity metric is inappropriate for continuous genome and thus has been omitted.

The evolution was not failing completely: best fitness improved from ~0.009 at
generation 1 to 0.2216 by generation 50 — a 25× increase above the random
baseline. However, a quantitatively similar ceiling appeared as in the baseline (Test 3) and Test 4.

#figure(
  image("evolution_fitness_stage1_lunarlander_ctrnn_test2_realvalued_1773943783.png",
        width: 100%),
  caption: [Test 5 — LunarLander (real-valued encoding) fitness curves over 50
  generations.],
) <fig-ll-rv-fitness>


#figure(
  image("evolution_trajectories_stage1_lunarlander_ctrnn_test2_realvalued_1773943783.png",
        width: 100%),
  caption: [Test 5 — LunarLander (real-valued encoding) state trajectories at
  selected generation snapshots.],
) <fig-ll-rv-traj>

The three conditions — 75% neutrality (baseline, Test 3), no neutrality (Test 4),
and continuous real-valued (Test 5) — hit ceilings of 0.2507, 0.2250, and 0.2216
respectively, similar within observed stochastic variation given only two
replicates per condition.

== LunarLander — Continuous Action Space

The convergence of all three discrete conditions to the same ≈0.22–0.25 band
suggested a structural barrier. A key feature of the discrete variant is the
action interface: the CTRNN produces two continuous output values which are
passed through softmax and reduced to a single argmax index (no action, left
thruster, main engine, right thruster). This discards all magnitude information
from the outputs — a CTRNN unit producing 0.4 for "fire main" and 0.6 for "no
action" does not fire, even though the difference is only 0.2. Incremental
improvements to the weights produce no behavioural change as long as the argmax
winner stays the same, creating flat plateaus in the fitness landscape.

LunarLanderContinuous removes this bottleneck entirely. Both actuators (main
engine, lateral thruster) are driven by the raw tanh output of the CTRNN's two
output units. Running the identical *p* = 0.75 binary GA for 200 generations,
both replicates solved the task convincingly: *best fitness 0.9099* (replicates:
0.9099, 0.9064), well above the task-solved threshold of 0.667
(@fig-llc-fitness). Training and generalisation components grew in tandem,
reaching ≈0.88 by generation 175, confirming that the high fitness reflects
genuine control across varied physics rather than a lucky test episode.

Diversity followed the same rapid collapse–recovery pattern as the discrete
baseline (~5,116 → ~376–536 by gen 10, recovering to ~423–600 by gen 50)
(@fig-llc-diversity). The sweep dynamics are identical to the discrete baseline;
what differs is that the post-sweep population can continue to improve fitness
because the continuous action interface provides a smooth gradient.

Action sequence analysis reveals a clear qualitative progression
(@fig-llc-actions). Early snapshots (generation 0–25) show sparse, erratic
main-engine firing with little lateral coordination — sufficient to slow descent
but not to land. By generation 50–100 the controller transitions to sustained
main thrust timed to altitude, with lateral corrections beginning to appear.
From generation 150 onwards, thrust patterns become smooth and modulated: the
main engine fires continuously during descent, tapering as the lander approaches
the pad, while lateral thrust is applied in brief corrective bursts to maintain
horizontal alignment. This progression from coarse on/off firing to finely
modulated regulation is exactly the multi-actuator coordination that was
inaccessible under the discrete action interface.

#figure(
  image("evolution_fitness_stage1_lunarlander_continuous_ctrnn_1773971165.png",
        width: 100%),
  caption: [Test 6 — LunarLanderContinuous ($p$=0.75) fitness curves over 200
  generations. Both replicates reach best fitness 0.9099/0.9064, well above
  the task-solved threshold of 0.667.],
) <fig-llc-fitness>

#figure(
  image("evolution_diversity_stage1_lunarlander_continuous_ctrnn_1773971165.png",
        width: 100%),
  caption: [Test 6 — Mean pairwise Hamming distance over 200 generations for
  LunarLanderContinuous. The collapse–recovery pattern mirrors the discrete
  baseline; unlike the discrete case, the post-sweep population continues to
  improve fitness under the smooth gradient.],
) <fig-llc-diversity>

#figure(
  image("evolution_trajectories_stage1_lunarlander_continuous_ctrnn_1773971165.png",
        width: 100%),
  caption: [Test 6 — LunarLanderContinuous state trajectories at selected
  generation snapshots. From early erratic descent (gen 0–25) to smooth,
  modulated landing behaviour (gen 150+).],
) <fig-llc-traj>

#figure(
  image("evolution_actions_stage1_lunarlander_continuous_ctrnn_1773971165.png",
        width: 100%),
  caption: [Test 6 — LunarLanderContinuous ($p$=0.75) action sequences at
  selected generation snapshots. Continuous thrust values for the main engine
  (upper band) and lateral thruster (lower band) are shown per timestep.
  Early generations show sparse, low-resolution activations; by generation
  200 the main engine and lateral thrusters are smoother and more frequent, reflecting fully coordinated landing behaviour.],
) <fig-llc-actions>

= 4 Discussion

The ablation results collectively rule out the genome encoding as the primary
explanation for the discrete LunarLander failure. Neither removing neutrality
nor switching to a fundamentally different encoding moved the fitness ceiling.
The continuous action experiment then identifies where the barrier actually lies:
not in the landscape, but in the representational interface between the
controller and the environment.

== The representational barrier: action-space mismatch

The clearest signal in the discrete data is the *agreement between all
conditions*. Three encoding strategies starting from very different genomes,
using different representations, and producing different diversity behaviours
all converge to the same fitness band of 0.22–0.25. Two independent replicates
of the binary run began from genomes that were ~50% different from each other
(i.e. as different as two random genomes can be) and still ended at the same
place. This pattern — a consistent ceiling reached from multiple independent
directions — is the hallmark of a structural property of the effective fitness
landscape.

The continuous action result identifies what *induces* that structure. If the
task physics were intrinsically hard, LunarLanderContinuous would also fail.
Instead, the same CTRNN architecture, the same GA, and the same neutrality
fraction solve it at 200 generations. The barrier is not an intrinsic property
of the task; it is an induced property of the effective fitness landscape
created by the *softmax/argmax interface* of the discrete variant. The
extended budget is a genuine confound, but the diversity and fitness
trajectories rule it out: the absence of any measurable improvement after
generation 10 in all discrete runs indicates that extending the budget would
not overcome the plateau. The failure is a mismatch between the smooth policy
class (CTRNN tanh outputs) and a discontinuous action interface (argmax), not
a limitation of training time.

What does fitness ≈0.25 represent in the discrete case? The lander descends in
a controlled way without crashing, but the discrete controller never fires the
main thruster continuously enough to brake and land. The argmax is the
mechanism: a CTRNN output of 0.4 for "main engine" and 0.6 for "no action"
fires nothing. Incremental weight changes that tilt the continuous output
slightly towards the main engine produce no firing — and therefore no reward
signal — until the argmax crosses the winner threshold. This creates extended
flat regions in the fitness landscape around the discrete action boundaries.
The landscape is not structurally deceptive in the sense of having misleading
gradients; it creates extended flat regions with sparse, discontinuous gradients
only at action boundaries — precisely where the argmax winner changes. The
discrete interface effectively transforms a continuous control problem into a
temporally precise switching problem, which is significantly harder for
evolutionary search. This is the mechanistic source of the plateau.

== Evaluation noise: a secondary complication

While the action-space mismatch is the primary cause of the discrete ceiling,
a second problem compounds it: noisy fitness estimates. LunarLander is
stochastic — the landing pad spawns in a random position each episode and
wind varies — so the same controller can score −80 in one episode and +50 in
the next, purely due to luck. Each individual is evaluated over only 12
episodes, which is not enough to reliably distinguish a genuinely better
controller from a luckier one. As a result, tournament selection will
sometimes promote the wrong individual, wasting generations and preventing
the GA from exploiting small but real improvements.

The real-valued encoding (Test 5) shows this effect most clearly. The GA
*did* make progress: best fitness improved from ~0.009 at generation 1 to
0.22 by generation 50 — roughly 25× above random. But mean population
fitness stayed persistently low and noisy (0.04–0.12 throughout), which
points to a second problem beyond noise: the mutation step is too large.
Test 5 applies Gaussian noise with $sigma = 0.05$ to each of the 176
parameters simultaneously. Across all parameters, this adds up to a large
total shift — an L2 norm of $approx 0.66$, meaning the child's weight vector
is on average 0.66 units away from the parent's in 176-dimensional space.
To put this concretely: a parent that had carefully tuned its main-engine
timing will produce children whose weights are sufficiently perturbed that
the timing behaviour is completely disrupted. The good parent is discovered,
but its offspring are scattered so far across parameter space that none of
them inherit the useful behaviour. So even when a high-performing individual
is found, the population mean never climbs — the fitness advantage is
immediately lost in the next generation.

In the binary encoding conditions, the Gray-coded mutation rate of 0.005
per bit produces much smaller per-parameter perturbations on average, so
offspring stay close to high-performing parents and the population mean
tracks best-individual fitness more reliably.

With the action-space barrier removed (Test 6, continuous), evaluation noise
is less damaging because the fitness landscape is smooth: every small
improvement to thrust timing produces a proportional improvement in reward,
so the signal-to-noise ratio is much better. The 200-generation budget
further allows the fitness signal to average out episode-to-episode variation,
and both replicates converged reliably.

== The role of neutrality at small population size

Neutrality ($p$=0.75) achieved the highest peak in the discrete case (0.2507
vs 0.2250 for $p$=0.0, Test 4) but via a degenerate mechanism: a rapid sweep to the
first viable individual followed by complete stasis. Neutrality is
theoretically beneficial because it slows sweep and preserves diversity for
subsequent exploration @Wagner2008NeutralismReconciliation @Wilke2001EvolutionFlattest — but this requires the sweep to be resisted. With
population size 20, tournament-3 completes a sweep in roughly ten generations
regardless of genome size, leaving no room for neutral drift to act.

In the continuous variant, neutrality performed exactly as designed: both
replicates solved the task with $p$=0.75 and identical parameter settings.
However, their near-identical final fitness (0.9099 and 0.9064) and the
reliable monotonic improvement in both replicates suggest that the fitness
landscape under the continuous action space is sufficiently smooth that
neutrality's specific contribution — maintaining diversity during exploration
— is hard to isolate. Both replicates would likely also succeed under $p$=0 with
sufficient generations. Disentangling the benefit of neutrality from the benefit
of the continuous action space would require a direct $p$=0 ablation on the
continuous variant, which is planned for Stage 2.

== Towards plasticity

The continuous action result closes one open question — the task is solvable
given the right representation and budget — while opening another: why 200
generations instead of 50? With a smooth gradient available, convergence is no
longer blocked by flat regions, but the 4× budget still points to a
coordination problem: main-engine timing, lateral correction, and final
approach must co-evolve simultaneously, each requiring many evaluations.

A plastic CTRNN that adapts weights *within* an episode — via Hebbian or
reward-modulated rules — reframes the search target. Rather than evolving
precise actuator weights (a narrow, fragile target), evolution must find a
learning rule that allows the controller to self-tune thrust timing during each
episode. Approximate initial weights become sufficient from the start,
reducing the required evolutionary budget and improving robustness across
physics variants.

Crucially, this is where neutrality plays its intended role: the population
can drift genotypically until a productive learning rule is encountered, rather
than prematurely converging on a fixed solution. Stage 2 will test whether
continuous actions, neutrality, and within-episode plasticity together achieve
faster and more reliable convergence than fixed-weight evolution on the same
tasks.









#bibliography("refs.bib", style: "harvard-cite-them-right")




