# Member 1 Validation Report

## Processing summary
- Training trips processed: 80 of 80 selected
- Student-test trips processed: 10 of 10 selected
- Training GPS observations: 208417
- Student-test GPS observations: 19160
- Exported candidate rows: 2030990
- Training candidate rows: 1839390
- Student-test candidate rows: 191600
- Runtime: 129.16 seconds

## Alignment and labels
- Mean alignment distance: 22.27 m
- Median alignment distance: 5.68 m
- 95th percentile alignment distance: 142.04 m
- Maximum alignment distance: 644.63 m
- Observations excluded/uncertain: 24478
- Positive labels: 183939
- Negative labels: 1655451
- Missing matched arc in candidate set: 15952
- Alignment distance > 50 m: 23255
- Route-position jump > 10: 4

## Candidate retrieval
- Exact-brute-force top-10 recall on sampled observations: 100.00%
- Candidate recall was sampled across up to five training trips; the KDTree can miss exact global neighbours.

## Data assumptions
- Timestamp differences are interpreted as seconds.
- Stationary observations have heading_diff=0.
- Zero-length arcs have heading_diff=0.
- Student-test route files are never used to construct inference features.
- The reference-route alignment is inferred supervision, not independently verified ground truth.
- Directed connectivity uses candidate.source == previous nearest candidate.target.
- Observations missing the aligned arc or exceeding the confidence policy are excluded.
