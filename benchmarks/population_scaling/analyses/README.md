# Population analysis records

`expected_score.json` collects the manuscript values for the grain-resolved
growth law, face law, time-rescaling controls, the 38-grain perturbed-start
predictability floor, and the virtual isothermal hold.

The 5,000-grain hold is prospective, with its protocol and decision rule fixed
before the 5,000-grain data were evaluated. It contains four targets. Across
all 34 tested case-target decisions, the model time-to-target error remains
within 0.29%, with one 0.1-time-unit premature stop in a retrospective
1,000-grain decision. In the four 5,000-grain decisions, the model stops no
earlier than the reference and at most 0.22% later.

Run `scripts/analyze_population_assets.py` on the release archive root to
recompute saved-time grain-volume error quantiles and the virtual-hold stop
times from the per-step volume tables. Face adjacency and the perturbed-start
record are distributed here as compact manuscript records; they require their
specialized source arrays rather than the population release archives.
