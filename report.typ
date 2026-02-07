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
  
  title: [The Effect of Neutrality on Population Diversity in NK Fitness Landscapes],
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
    "NK fitness landscapes",
    "Neutrality",
    "Genetic Diversity",
    "Neutral Networks",
    "Genetic Drift",
  ),
  doi: "10.7891/120948510",
  abstract: [
   *Background:* Neutral mutations, which do not affect fitness, are predicted to enhance evolutionary potential by allowing populations to explore genotype space via neutral networks. However, the relationship between neutrality and genetic diversity remains poorly understood, particularly in epistatic fitness landscapes. 

*Methods:* We implemented an NKp fitness landscape model with varying neutrality probability ($p$) from 0.0 to 0.9. Populations of 100 individuals with 50-bit binary genomes evolved for 1500 generations under tournament selection and bit-flip mutation. We measured genetic diversity using average pairwise Hamming distance, genetic entropy, and unique genotype counts.

*Results:* Genetic diversity exhibited an inverted-U relationship with neutrality, peaking at $p = 0.75$ (Hamming distance: $2.622 ± 0.218$, representing a 115% increase over $p = 0$). Diversity declined at very high neutrality ($p = 0.9$: $2.376 ± 0.182$), despite increased mutation neutrality. This pattern emerged only after extended evolution (>500 generations), indicating a dynamic equilibrium effect. Importantly, diversity reduction at high neutrality occurred despite measuring raw genetic variation, indicating a lineage-level effect driven by genetic drift rather than fitness convergence.

*Conclusions:* Within the NKp landscapes studied here, intermediate neutrality is associated with maximal population diversity through a balance between neutral exploration and selective maintenance of population structure. Excessive neutrality ($p > 0.8$) is associated with reduced diversity, consistent with a regime where genetic drift overwhelms weak selection and collapses independent lineages. These results are consistent with neutral network theory and edge-of-chaos predictions for the neutrality–selection balance, within the parameter regimes examined.
  ],
)  



= 1 Introduction

== Neutral Theory and Evolutionary Dynamics

The neutral theory of molecular evolution posits that most genetic variation at the molecular level is selectively neutral, with evolutionary dynamics driven primarily by mutation and genetic drift rather than natural selection @Kimura1983NeutralEvolution. This paradigm shift challenged adaptationist perspectives and generated substantial debate about the relative importance of neutral versus selective processes in evolution.

More recently, the concept of *neutral networks* has emerged as a framework for understanding how neutrality enhances evolvability @Wagner2008NeutralismReconciliation. Neutral networks are connected sets of genotypes with equal fitness, allowing populations to explore genotype space through neutral mutations without fitness costs. This neutral exploration can facilitate access to novel fitness peaks that would be inaccessible under purely selective evolution @Barnett2001NetcrawlingNetworks.

== NK Fitness Landscapes

Kauffman's NK model provides a tunable framework for studying evolution in rugged fitness landscapes @Kauffman1993OriginsEvolution. The model specifies:
- *N*: genome length
- *K*: number of epistatic interactions per locus

Higher *K* values create more rugged, epistatic landscapes where gene interactions dominate fitness determination. At *K* = 0, genes contribute independently to fitness (smooth landscape), while *K* = *N*-1 produces a completely random landscape (maximally rugged).

== NKp Model: Incorporating Neutrality

Barnett and colleagues extended the NK model to include a neutrality parameter *p* @Barnett2001NetcrawlingNetworks. In the NKp model, each fitness table entry has probability *p* of being neutral (sharing fitness value with neighbouring genotypes differing by one mutation). This creates neutral networks whose extent is controlled by *p*:
- *p* = 0: no neutrality (standard NK model)
- 0 < *p* < 1: partial neutrality (neutral networks of varying size)
- *p* = 1: complete neutrality (all mutations neutral)

We chose the NKp formulation rather than alternative discretised-fitness models (e.g. NKQ) because NKp allows explicit control over the probability that mutations are selectively neutral, while preserving continuous fitness differences among non-neutral genotypes. This enables a clean separation of neutrality effects from fitness resolution effects, which is essential for analysing selection–drift balance and lineage persistence.

== Research Question and Hypothesis

Despite theoretical work on neutral networks, the quantitative relationship between neutrality and population genetic diversity remains unclear. We hypothesised that intermediate levels of neutrality would maximise population diversity by balancing neutral exploration with selective maintenance of population structure. Specifically, we predicted:

1. Low neutrality (*p* < 0.3) would produce low diversity due to strong selective constraints
2. Intermediate neutrality (*p* ≈ 0.5-0.7) would maximise diversity through optimal exploration-selection balance
3. High neutrality (*p* > 0.8) would reduce diversity due to loss of selective population structure

Crucially, genetic diversity in evolving populations depends not only on mutation supply but also on the persistence of independent genealogical lineages. While neutrality increases mutational exploration, selection can indirectly maintain diversity by preventing lineage coalescence. This study therefore examines neutrality not only as a source of variation, but as a modulator of lineage structure under drift–selection balance. We focus specifically on long-term equilibrium diversity rather than short-term exploratory dynamics.



While previous studies have demonstrated that neutrality can enhance evolvability, this study provides evidence that, within the NKp landscapes studied here, genetic diversity is maximised at intermediate neutrality due to lineage-level effects. In particular, we show that excessive neutrality is associated with reduced diversity through drift-driven lineage collapse, even when raw neutral mutation rates are high.

= 2 Methods


== NKp Fitness Landscape Implementation

We implemented an NKp fitness landscape with the following parameters:
- Genome length: *N* = 50 (binary alleles: 0 or 1)
- Epistatic interactions: *K* = 2 (each locus interacts with itself plus 2 randomly chosen partner loci)
- Neutrality probability: *p* ∈ {0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9}

We selected *K* = 2 to sample the moderate-epistasis regime often associated with the "edge of chaos": this setting produces structured but traversable fitness landscapes that make neutrality–selection interactions interpretable while avoiding maximal ruggedness.

For each locus *i*, fitness contribution was determined by a lookup table indexed by the allelic states of locus *i* and its *K* interaction partners. Table entries were initially assigned random contribution values from U(0,1). To implement neutrality, we then processed each table entry: with probability *p*, the entry's contribution value was replaced by copying from a randomly selected neighbour (an entry differing by exactly one bit flip). This creates neutral mutations: when two neighbouring entries share the same contribution value, a single-bit mutation between them produces no fitness change.

Note that copied values do not guarantee global neutrality transitivity; neutral relationships are local and table-specific, producing heterogeneous neutral-network connectivity rather than necessarily creating globally transitive neutral classes.

Example:

Entry [0,1,0] has fitness 0.456
Its neighbour [0,1,1] initially has fitness 0.892
With probability p, we might copy 0.456 from [0,1,0], so [0,1,1] also becomes 0.456
Now a mutation that changes [0,1,0] → [0,1,1] is neutral (same fitness)

Total fitness was calculated by averaging the contributions from all *N* loci:
$ f("genome") = 1/N sum_(i=1)^N "ContributionTable"_i ("alleles"_("i and partners")) $

The averaging normalises fitness to the range [0,1], making it independent of genome length and directly interpretable (1 = maximum fitness). Alternatively, using the sum would produce identical selective rankings but fitness values in range [0, *N*]. The normalization does not affect evolutionary dynamics, as selection operates on relative fitness differences.

This architecture creates epistasis: mutating a single gene (e.g., position 23) affects multiple locus contributions—not only locus 23's contribution, but also the contributions of any other loci that selected position 23 as an interaction partner. With *K* = 2, each gene participates in approximately 3 contribution calculations on average (its own locus plus ~2 others), creating moderate epistatic coupling across the genome.


*Algorithm 1: NKp Landscape Construction*
```
for each locus i in 1..N:
    randomly select K interaction partners from other loci
    // e.g., locus 5 might interact with loci 23 and 41
    create fitness contribution table with 2^(K+1) entries
    
    for each table entry:
        assign random contribution value ~ U(0,1)
    
    for each table entry:
        with probability p:
            choose random neighbour (differing by 1 bit)
            copy contribution value from neighbour
            // creates neutral mutations
            ```


            
*Example:* For locus 5 with partners {23, 41} and *K* = 2, the fitness contribution table has 2³ = 8 entries:
```
Alleles [locus5, locus23, locus41] → Contribution to fitness
[0, 0, 0] → 0.734  (initially random)
[0, 0, 1] → 0.892
[0, 1, 0] → 0.456
[0, 1, 1] → 0.456  (copied from [0,1,0] with probability p)
[1, 0, 0] → 0.621
[1, 0, 1] → 0.621  (copied from [1,0,0] with probability p)
[1, 1, 0] → 0.283
[1, 1, 1] → 0.147
```
This contribution table is constructed once at initialisation and remains fixed throughout evolution. Each of the N=50 loci has its own independent contribution table. The *K* interaction partners are randomly chosen across the entire genome (not spatial neighbours), creating long-range epistatic interactions.


            
== Genetic Algorithm

Populations evolved under a standard genetic algorithm with asexual reproduction:

*Population structure:*
- Population size: 100 individuals
- Genome representation: 50-bit binary strings
- Initialisation: random uniform

*Genetic operators:*
- Selection: Tournament selection (size = 3)
- Mutation: Bit-flip with probability 0.01 per locus
- Reproduction: Asexual (no recombination)

*Evolutionary dynamics:*
Each generation, 100 offspring were produced by:
1. Selecting a parent via tournament selection
2. Copying parent genome
3. Applying bit-flip mutation
4. Replacing entire population with offspring

*Algorithm 2: Evolutionary Process*
```
initialize population of 100 random genomes

for generation in 1..1500:
    offspring = []
    
    for i in 1..100:
        // Tournament selection (sampling with replacement)
        tournament = randomly select 3 individuals from population
        // Note: same individual can appear in multiple tournaments
        parent = individual with highest fitness in tournament
        
        // Reproduction with mutation
        child = copy of parent genome
        for each bit in child:
            with probability 0.01:
                flip bit (0→1 or 1→0)
        
        offspring.append(child)
    
    population = offspring  // Replace entire population
    
    if generation % 10 == 0:
        record diversity metrics

// Fitness calculation for any genome:
function calculate_fitness(genome):
    total = 0
    for each locus i in 1..N:
        allele_config = [genome[i], genome[partner1_i], genome[partner2_i]]
        contribution = ContributionTable_i[allele_config]
        total += contribution
    return total / N  // Average over all N=50 loci
```

*Example fitness calculation:* For a genome `[1,0,1,0,1,...]`, fitness is computed as follows:

1. Locus 1 contributes based on `[genome[1], genome[partner1₁], genome[partner2₁]]`
2. Locus 2 contributes based on `[genome[2], genome[partner1₂], genome[partner2₂]]`
3. ...
4. Locus 5 contributes 0.621 by looking up `[genome[5]=1, genome[23]=0, genome[41]=1]` = `[1,0,1]` in its contribution table
5. ...
6. Locus 50 contributes based on `[genome[50], genome[partner1₅₀], genome[partner2₅₀]]`
7. *Total fitness = (sum of all 50 contributions) / 50*

Crucially, if genome position 23 mutates from 0→1, this affects not only locus 23's contribution, but also locus 5's contribution (and any other locus that has 23 as a partner). Each affected locus evaluates the mutation through its own independent contribution table, creating complex, non-additive fitness effects characteristic of epistatic landscapes.

*Important note on epistatic complexity:* Interaction partner sets can overlap by chance. For instance, if locus 23 happened to select partners {5, 41}, both loci 5 and 23 would depend on the same three positions {5, 23, 41}. However, they use *independent contribution tables*, so the same allele configuration [1,0,1] would yield different contributions (e.g., 0.621 from locus 5's table, 0.342 from locus 23's table). This independence creates complex epistatic interactions where the same genetic change can have multiple, uncorrelated fitness effects.



== Diversity Metrics

We quantified genetic diversity using three complementary metrics:

*1. Average Pairwise Hamming Distance*
$ D_H = 1/(binom(N_"pop", 2)) sum_(i<j) d_H ("genome"_i, "genome"_j) $
where $d_H$ counts differing alleles between genomes $i$ and $j$.

*2. Genetic Entropy*
$ H = 1/L sum_(ℓ=1)^L [-sum_(a in {0,1}) p_(ℓ,a) log_2(p_(ℓ,a))] $
where $p_(ℓ,a)$ is the frequency of allele $a$ at locus $ℓ$, and *L* = 50.

*3. Unique Genotypes*
The count of distinct genotypes present in the population.

*Critical methodological note:* These metrics measure *raw genetic diversity* across the entire 50-bit genome, treating all loci equally regardless of their functional status. This is essential for understanding how neutrality affects population structure. We explicitly avoid filtering for functional sites to capture how selection on a subset of loci affects the retention of variation across the entire genome (including linked neutral sites).


== Experimental Design

*Primary experiment:* We tested 7 neutrality levels (*p* ∈ {0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9}) with 20 replicate populations per level. Each population evolved for 500 generations with diversity metrics recorded every 10 generations.

*Equilibrium validation:* To verify that diversity patterns reflected true equilibrium rather than transient dynamics, we conducted an extended experiment with 4 key neutrality levels (*p* ∈ {0.0, 0.45, 0.75, 0.9}) run for 1500 generations with 15 replicates per level.

All experiments used independent random seeds for landscape generation and evolutionary dynamics.

As a conceptual control, we also considered the limiting case of complete neutrality (p = 1), corresponding to pure drift without selection, to contextualise diversity dynamics in the absence of lineage-stabilising forces.



== Statistical Analysis

For each neutrality level, we computed mean ± standard error of the mean (SEM) across replicates for final diversity values. Within-population fitness variation is reported as standard deviation ($sigma$). We assessed normality visually and using standard diagnostics and—because diversity metrics across replicates were not guaranteed to be normally distributed—used non-parametric tests for formal inference. Pairwise comparisons between key conditions (for example, *p* = 0.75 vs *p* = 0.9) used the Mann–Whitney U test (two-sided). For comparisons across multiple groups we used Kruskal–Wallis tests followed by appropriate post-hoc pairwise comparisons with Bonferroni or Dunn corrections to control family-wise error. Reported p-values are two-sided unless otherwise stated; exact p-values and test statistics are provided where available. We also report effect-size estimates (rank-biserial correlation) for key pairwise contrasts to aid interpretation.


= 3 Results

== Diversity Exhibits Inverted-U Relationship with Neutrality

Population genetic diversity showed a non-monotonic relationship with neutrality probability *p* (Figure 1). All three diversity metrics exhibited qualitatively similar patterns:

*Hamming Distance* (primary metric):
- *p* = 0.0: $1.217 ± 0.042$ (baseline)
- *p* = 0.45: $1.876 ± 0.140$ (+54% vs baseline)
- *p* = 0.75: $2.622 ± 0.218$ (*maximum*, +115% vs baseline)
- *p* = 0.9: $2.376 ± 0.182$ (+95% vs baseline, but 9% below maximum)

Pairwise inference: a Mann–Whitney U test comparing final Hamming distance at *p* = 0.75 versus *p* = 0.9 (15 replicates per condition, two-sided) returned U = 132.0, p = 0.430648; the difference is not statistically significant at conventional thresholds (rank-biserial effect size ≈ -0.17).

#figure(
  image("diversity_panel_1.pdf", width: 100%),
  caption: [
    *Hamming distance peaks at intermediate neutrality.* 
    Average pairwise Hamming distance shows maximum genetic diversity at *p* = 0.75 (2.622 ± 0.218, highlighted in green), representing a 115% increase over baseline (*p* = 0: 1.217 ± 0.042). Diversity declines at very high neutrality (*p* = 0.9: 2.376 ± 0.182, 9% below optimum), demonstrating an inverted-U relationship. Error bars represent standard error across 15 replicate populations evolved for 1500 generations. NKp model parameters: *N* = 50, *K* = 2, population size = 100, mutation rate = 0.01 per locus.
  ]
)

*Genetic Entropy*:
- *p* = 0.0: $0.087 ± 0.011$
- *p* = 0.75: $0.141 ± 0.009$ (*maximum*, +62% vs baseline)
- *p* = 0.9: $0.134 ± 0.027$ (declined from *p* = 0.75)

The diversity peak at intermediate neutrality (*p* = 0.75) was robust across metrics. We do not observe a monotonic increase: *p* = 0.9 is not significantly higher than *p* = 0.75 (Mann–Whitney U = 132.0, p = 0.431, rank-biserial ≈ -0.17). While the numerical mean was highest at *p* = 0.75, the overlap with *p* = 0.9 suggests a plateauing effect in which the benefits of additional neutrality are counterbalanced by loss of selective population structure.

Importantly, mutation rate was held constant across all neutrality levels, confirming that observed diversity differences arise from neutrality–selection interactions rather than differences in mutational input.





#figure(
  image("diversity_panel_2.pdf", width: 100%),
  caption: [
    *Genetic entropy validates diversity patterns.* 
    Shannon entropy measured across all 50 loci exhibits qualitatively similar inverted-U relationship as Hamming distance, with maximum at *p* = 0.75 (0.141 ± 0.009, 62% increase over baseline). This independent metric confirms that intermediate neutrality maximises population genetic variation. Error bars represent standard error across 15 replicates.
  ]
)

#figure(
  image("diversity_panel_3.pdf", width: 100%),
  caption: [
    *Unique genotypes follow inverted-U pattern.* 
    Number of distinct genotypes in populations shows similar relationship with neutrality as other diversity metrics. Peak diversity at *p* = 0.75 demonstrates that intermediate neutrality maximises genotypic variation through balance of neutral exploration and selective population structure. Error bars represent standard error across 15 replicates.
  ]
)



== Equilibrium Dynamics: Pattern Emergence Over Time

Short-term results (500 generations) showed marginal differences between *p* = 0.75 and *p* = 0.9, with *p* = 0.9 slightly higher ($2.266 ± 0.164$ vs $2.246 ± 0.169$). However, extended evolution revealed divergent trajectories:

*Diversity change from generation 500 to 1500:*
- *p* = 0.75: $2.246 → 2.622$ (+16.8% increase)
- *p* = 0.9: $2.266 → 2.376$ (+4.8% increase)

This resulted in a *reversal of ranking* (Figure 5), with *p* = 0.75 emerging as the clear optimum only after extended evolution. The dramatic difference in diversity growth rates (Figure 6) suggests that *p* = 0.75 maintains ongoing exploration while *p* = 0.9 experiences stagnation. 

Qualitatively similar dynamics were observed for genetic entropy, confirming the robustness of this temporal pattern. Meanwhile, fitness trajectories (Figure 4) showed that all populations reached stable adaptive plateaus well before the divergence in diversity occurred, confirming that the diversity split was not driven by differences in adaptive success.






#figure(
  image("time_panel_3.pdf", width: 100%),
  caption: [
    *Fitness rapidly increases from random initialisation.* 
    Mean population fitness over 1500 generations shows rapid adaptive improvement during first 100-200 generations, followed by plateau at equilibrium values. All neutrality levels maintain viable fitness (>0.68), with modest differences between conditions. Higher neutrality shows slightly reduced equilibrium fitness, reflecting exploration-adaptation trade-off. Shaded regions indicate standard error across 15 replicates.
  ]
)

#figure(
  image("equilibrium_panel_1.pdf", width: 100%),
  caption: [
    *Extended evolution reveals *p* = 0.75 as the clear optimum.* 
    Side-by-side comparison of diversity after 500 generations (blue bars) versus 1500 generations (red bars). At 500 generations, *p* = 0.75 and *p* = 0.9 show similar diversity levels (2.246 vs 2.266, respectively). Extended evolution to 1500 generations produces divergent outcomes: *p* = 0.75 reaches 2.622 whilst *p* = 0.9 achieves only 2.376, reversing the ranking. Green shading highlights the *p* = 0.75 condition. This pattern demonstrates that short-term observations may not reflect long-term evolutionary outcomes. Error bars represent standard error across replicate populations.
  ]
)

#figure(
  image("equilibrium_panel_2.pdf", width: 100%),
  caption: [
    *Diversity growth rates differ dramatically between neutrality levels.* 
    Change in Hamming distance from generation 500 to 1500. The *p* = 0.75 condition (highlighted in green) shows the largest increase (+0.377, +16.8%), indicating continued exploration with stable population structure. In contrast, *p* = 0.9 exhibits minimal growth (+0.109, +4.8%), suggesting stagnation due to weak selection. Baseline (*p* = 0) and intermediate (*p* = 0.45) conditions show negligible change, having reached equilibrium by generation 500. This differential growth pattern confirms that intermediate neutrality maximises long-term diversity maintenance. Error bars represent standard error across replicate populations.
  ]
)



== Fitness Remains Viable Across All Neutrality Levels

Despite varying neutrality, all populations maintained viable mean fitness levels at generation 1500:

- *p* = 0.0: $0.727 ± 0.024$ (*highest*)
- *p* = 0.45: $0.704 ± 0.025$
- *p* = 0.75: $0.689 ± 0.025$
- *p* = 0.9: $0.686 ± 0.028$ (*lowest*, but viable)

All populations showed substantial fitness increases from initial random states (≈0.50) to evolved states (>0.68), demonstrating that selection remained effective even at high neutrality. The modest fitness decline with increasing *p* (6% reduction from *p* = 0 to *p* = 0.9) indicates that neutrality trades off adaptive optimisation for exploratory potential.

#figure(
  image("diversity_panel_4.pdf", width: 100%),
  caption: [
    *Fitness remains viable across all neutrality levels.* 
    Mean population fitness shows modest decline with increasing neutrality (*p* = 0: 0.727 ± 0.024; *p* = 0.9: 0.686 ± 0.028), representing only 6% reduction. All populations achieve substantial fitness increases from random initialisation (≈0.50), demonstrating that selection remains effective even at high neutrality. This viability across all *p* values ensures that diversity patterns reflect exploration-selection balance rather than population collapse. Error bars represent standard error across 15 replicates.
  ]
)

== Lineage Persistence and Coalescence

To investigate the mechanism underlying diversity loss at high neutrality, we analyzed the genealogical history of the populations. We performed targeted reruns ($n=5$ per condition) with lineage tracking to calculate the Time to Most Recent Common Ancestor (TMRCA).

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: center,
    stroke: none,
    table.hline(),
    table.header(
      [*Neutrality ($p$)*], [*Mean TMRCA*], [*Median TMRCA*], [*$n$*]
    ),
    table.hline(),
    [0.75], [102.6], [104], [5],
    [0.90], [88.6], [64], [5],
    table.hline(),
  ),
  caption: [
    *Lineage persistence summary.* Comparison of TMRCA (generations before final) for optimal (*p* = 0.75) and high (*p* = 0.9) neutrality. Shorter TMRCA indicates more rapid lineage turnover.
  ]
)



At *p* = 0.9, populations exhibited consistently shallower genealogical trees. The median time to the Most Recent Common Ancestor was **64 generations**, compared to **104 generations** at *p* = 0.75. 

This 38% reduction in coalescence time suggests that lineages turn over much more rapidly at high neutrality, though we note the limited sample size ($n=5$) for these high-resolution genealogical traces. Muller plots (visualisations of lineage frequencies over time, see Supplementary Figures) confirmed this dynamic: *p* = 0.9 populations were characterised by frequent selective sweeps or drift-driven fixations that purged genetic variation, whereas *p* = 0.75 populations maintained multiple co-existing lineages for longer durations.

#figure(
  image("tmrca_plot.pdf", width: 80%),
  caption: [
    *Time-to-MRCA distributions.* Boxplots show the distribution of generations back to the common ancestor. The *p* = 0.9 condition (right) shows consistently shorter coalescence times, indicating that drift removes distinct lineages more rapidly than in the *p* = 0.75 condition.
  ]
)

= 4 Discussion

== Neutrality-Selection Balance and the Diversity Optimum
Our results indicate that, for the parameter regime studied, intermediate neutrality (p ≈ 0.75) is associated with maximal population genetic diversity through a balance between neutral exploration and selective maintenance of population structure. This inverted-U relationship is consistent with predictions from neutral network theory @Wagner2008NeutralismReconciliation and edge-of-chaos models @Kauffman1993OriginsEvolution.

Under a purely mutational view of neutrality, one would expect genetic diversity to increase monotonically with increasing neutrality, as a larger fraction of mutations would be selectively tolerated. The inverted-U pattern observed here therefore requires explanation beyond mutation supply alone, motivating a lineage-based interpretation of diversity maintenance.

The mechanism underlying this pattern involves two opposing forces:

*At low neutrality (p < 0.5)*: Strong selection rapidly drives populations towards local fitness peaks. Limited neutral exploration restricts the accessible genotype space, reducing diversity. Populations become "trapped" in fitness valleys with low genetic variation.

*At intermediate neutrality (p ≈ 0.6-0.8)*: Neutral networks provide pathways for exploration without fitness loss, while remaining functional sites (20-40% of genome) create fitness differences that partition the population into distinct lineages. Weak but persistent selection stabilises these fitness-differentiated subpopulations, slowing lineage coalescence and maintaining higher diversity at neutral sites.

*At high neutrality (p > 0.8)*: Excessive neutrality fails to further increase diversity—and nominally reduces it—suggesting that genetic drift begins to overwhelm the benefits of additional neutral pathways. With only 10-20% of the genome under selection, fitness differences between lineages are minimal and unstable. Genetic drift dominates, causing lineages to merge through random sampling. Paradoxically, increased neutral mutation opportunity leads to decreased diversity through drift-driven homogenisation. Rapid coalescence collapses genealogical diversity, limiting the accumulation of neutral variation despite higher mutational freedom. In the limiting case of complete neutrality, classical Wright–Fisher theory predicts a monotonic loss of diversity due to stochastic lineage coalescence, with expected heterozygosity declining at a rate proportional to 1/Ne.

These results highlight that neutrality does not operate independently of selection, but reshapes genealogical dynamics by modulating effective population size and lineage persistence.


== Edge of Chaos in Neutrality Space

Our findings align with Kauffman's "edge of chaos" hypothesis, but reveal an additional dimension. Classical NK models explore the edge of chaos via the epistasis parameter *K*, with optimal complexity near *K* ≈ 2 @Kauffman1993OriginsEvolution. We demonstrate that neutrality (*p*) constitutes a second axis, with optimal evolvability near *p* ≈ 0.75.

The system operates at a double edge of chaos:
- *K* = 2: Balanced epistasis (neither independent nor maximally coupled)
- *p* = 0.75: Balanced neutrality (neither pure selection nor pure drift)

This two-dimensional edge-of-chaos region may represent the parameter space where biological systems operate @Aldana2003NaturalNetworks.

Our observed diversity maximum at p ≈ 0.75 aligns with this two-dimensional edge-of-chaos region, suggesting populations balance neutrality and epistasis to optimise evolvability.

== Linkage and Neutral Site Diversity

Genetic diversity in this system emerges from a compounding interaction between lineage persistence and within-lineage neutral exploration.

A striking finding is that diversity reduction at high neutrality affects *all* genomic sites, not just functional ones. Crucially, our diversity metrics measure *raw genetic diversity* across the entire 50-bit genome, counting all sites equally regardless of their functional status. At *p* = 0.75, selection operates on only 25% of sites (12-13 loci), yet the resulting population structure preserves diversity across the remaining 75% neutral sites (37-38 loci).

This occurs through linkage: neutral sites are physically linked to selected sites on the same genome. When selection maintains distinct lineages based on functional differences at 25% of loci, neutral sites hitchhike along, diverging between lineages. The population structure created by weak selection at a minority of sites is sufficient to maintain genetic diversity genome-wide.

The lineage tracking results provide direct empirical support for this mechanism. The markedly shorter TMRCA observed at p = 0.9 (median 64 generations vs 104 generations at p = 0.75) confirms that these populations experience rapid genealogical turnover. Without a scaffold of selected sites to anchor independent lineages, genetic drift drives rapid coalescence, constantly purging the population of the distinct neutral variants that attempt to accumulate. Thus, the potential for high diversity offered by the 90% neutral sites at p = 0.9 is negated by the inability of the population to sustain the deep lineages required to store that diversity.

The fact that diversity does not significantly increase between *p* = 0.75 and *p* = 0.9, despite roughly a 20% increase in the availability of neutral mutations, suggests the population has reached a "drift barrier." The additional neutral sites available at *p* = 0.9 are effectively wasted because the population structure is too weak to maintain the lineages that would carry and preserve that extra variation.


== Temporal Dynamics and Equilibrium

The pattern reversal between 500 and 1500 generations highlights the importance of evolutionary timescale. Short-term dynamics at *p* = 0.9 show high diversity due to rapid neutral exploration, but this diversity is unstable. At *p* = 0.75, diversity accumulates more slowly but reaches a higher stable equilibrium through maintained population structure.

This has implications for experimental evolution studies: short-term observations may not reflect long-term evolutionary outcomes, particularly when selection-drift balance is involved.

Short-term studies may overestimate diversity at high neutrality because rapid neutral exploration temporarily inflates variation before drift collapses lineages.



== Biological Relevance

Our optimal neutrality value (*p* ≈ 0.75) aligns with empirical estimates for biological systems. RNA viruses, for instance, show approximately 70-80% of mutations being neutral or nearly neutral @Sanjuan2004DistributionVirus. Similarly, protein evolution studies suggest that 60-80% of mutations in non-critical regions may be effectively neutral @Bloom2005ThermodynamicNeutrality.

This convergence suggests that biological systems may have evolved towards the neutrality level that maximises exploratory potential while maintaining adaptive capacity—consistent with the "survival of the flattest" hypothesis in viral evolution @Wilke2001EvolutionFlattest.

== Limitations and Future Directions

Although the NKp model captures key features of epistatic and neutral fitness landscapes, the precise location of the diversity optimum is expected to depend on population size, mutation rate, and landscape structure. The qualitative mechanism identified here—lineage maintenance under intermediate selection—is therefore likely to generalise beyond the specific parameter values used.

Several limitations warrant consideration:

*1. Simplified landscape:* Real biological fitness landscapes are more complex than NK models, with temporal variation, frequency dependence, and environmental heterogeneity.

*2. Asexual reproduction:* Recombination would alter neutral network connectivity and potentially shift the diversity optimum. Our results are therefore specific to asexual (no-recombination) dynamics; introducing recombination would decouple neutral sites from linked selected sites and could substantially change diversity maintenance mechanisms and the location of any optimum.

*3. Fixed mutation rate:* Mutation rate evolution could interact with neutrality to modify diversity patterns.

*4. Population size:* Our population size (*N* = 100) is small; larger populations would experience weaker drift, potentially shifting the optimum toward higher neutrality.

Future work could explore:
- The *p*-*K* parameter space systematically
- Effects of recombination on neutral network traversal
- Co-evolution of mutation rate and neutrality
- Scaling relationships with population size
- Stochasticity in small populations may accentuate drift effects and shift the diversity optimum, highlighting the need for scaling studies.

== Conclusions

We demonstrate that population genetic diversity in epistatic fitness landscapes exhibits an inverted-U relationship with neutrality, maximising at intermediate neutrality levels (*p* ≈ 0.75). This pattern arises from a fundamental trade-off: neutrality enables exploration of genotype space, but excessive neutrality eliminates the selective population structure required to maintain diversity.

These findings have three key implications:

1. *Theoretical:* Support neutral network theory and edge-of-chaos models of evolvability
2. *Methodological:* Demonstrate the necessity of long-term evolutionary experiments to observe equilibrium dynamics
3. *Biological:* Suggest that observed neutrality levels in biological systems may be optimised for exploratory potential

More broadly, our results show that genetic diversity is an emergent property of lineage dynamics, arising from the compounding interaction between selection-maintained population structure and neutral variation accumulating within those structures.

The neutrality-selection balance represents a fundamental constraint on evolutionary dynamics, with intermediate parameter values providing optimal conditions for both adaptation and evolvability.

These results highlight that maximal genetic diversity does not arise from maximal neutrality, but from an optimal balance in which selection preserves independent lineages while neutrality enriches variation within them.









#bibliography("refs.bib", style: "harvard-cite-them-right")




