# Causal deceleration estimates across road environments

Study 36. Protocol fixed before retrieving trajectory rows or fitting models, 28 October 2026. The initial source probe inspected only schema, terms and location counts.

## Decision and limits

Can a causal estimate of acceleration from recent speed measurements transfer from one road environment to another, and can it flag strong recorded deceleration without excessive false alerts? This is an offline reference-signal reconstruction study. NGSIM acceleration is a processed reference, not independently instrumented brake actuation or crash ground truth. Speed and reference acceleration may share processing; agreement is not evidence of real sensor accuracy or road-safety benefit.

## Source and cohort

U.S. DOT/FHWA NGSIM vehicle trajectories, Socrata dataset 8ect-6jqj. The official metadata lists public rights and Creative Commons Attribution-ShareAlike 3.0. Retain only anonymous vehicle/frame identifiers, time, speed, acceleration, longitudinal position, vehicle class and location. No video, plates, faces or precise global coordinates.

Use vehicle IDs 1 through 80 at each of I-80, Lankershim and US-101. This deterministic early-ID convenience sample controls download and fitting cost; it is not random or representative of a network. IDs repeat across recording intervals and must never be joined globally. Deduplicate identical selected rows; reject conflicting location/vehicle/time keys. Split trajectories at every non-100-ms step or frame discontinuity. Drop frames lacking the full 1-second causal history. Report every exclusion.

## Split and models

Train on I-80 only; select model and alert threshold on Lankershim only; evaluate once on US-101 only. Do not refit using validation or test. No shuffled row splits. Group uncertainty by trajectory and state that trajectories share traffic conditions, leaving road-level generalization uncertain.

Features: current speed, causal differences over 0.1, 0.3, 0.5 and 1.0 seconds, and past-only speed mean/standard deviation over the preceding 1.0 seconds. Convert feet to meters with exactly 0.3048. Do not use reference acceleration as a feature, including its lags. Do not use future frames or a centered smoother.

Baselines: zero acceleration and speed differences over 0.1, 0.3, 0.5 and 1.0 seconds. Ridge with alpha 0.01, 0.1, 1, 10, 100; standardization fit on training only. Histogram gradient boosting with max depth 2 or 3, learning rate 0.05 or 0.1, and L2 0 or 1; 100 iterations, minimum leaf size 30, early stopping off, seed 20261028. Fit with equal total weight per training trajectory. Retain all 18 candidate results, runtime and settings.

Primary selection: lowest mean trajectory MSE on validation (units m²/s⁴), then model name. Report trajectory-balanced RMSE/MAE and frame-weighted MSE/RMSE/MAE separately. A boundary winner is not a global optimum. No grid expansion after test inspection.

## Alert operating point

Reference positive: recorded acceleration at or below -8 feet/s² (-2.4384 m/s²). This is a declared analytical threshold, not a regulatory safety definition. For the validation-selected model and the validation-selected simple baseline, search alert thresholds -12 to -4 feet/s² in steps of 0.5. Maximize validation recall subject to frame-level false-positive rate <=1%; break ties by lower FPR, then more conservative threshold. If no nontrivial threshold qualifies, use an always-negative alert rule and report zero recall. Lock each threshold before test inspection. Report confusion counts, prevalence, precision, recall and FPR; an FPR constraint on one road is not a test-road guarantee.

## Evidence and checks

Archive exact public CSV pages, URLs, timestamps and SHA-256 hashes; independently reconcile source row counts. Save predictions for all validation candidates, and only frozen selected model/baseline on test. Independently recompute MSE, RMSE, MAE and alert confusion counts from exports. Recheck SQL lag timestamps and row-level disjointness. Use 2,000 trajectory bootstrap draws for paired balanced-MSE differences, with caveats about shared traffic. Compare the selected-model acceleration errors by speed group and vehicle class without changing the selection. Inspect figures and private manuscript; publish code, eligible data and results, never manuscript drafts.

## Scope of contribution

The contribution tested is a reproducible evaluation of causal signal reconstruction and an alert threshold under road-domain shift. No new algorithm, globally novel research gap, deployed safety system or business savings is claimed. A deployment decision requires independently measured sensors and more roads.
