# Sources, units and reuse

Dataset: **Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data**, Federal Highway Administration, U.S. Department of Transportation.

- Official dataset: https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajectories/8ect-6jqj
- Metadata: https://data.transportation.gov/api/views/8ect-6jqj.json
- Rows: https://data.transportation.gov/resource/8ect-6jqj.csv
- License stated in the metadata: **Creative Commons Attribution-ShareAlike 3.0 Unported**, https://creativecommons.org/licenses/by-sa/3.0/
- Metadata temporal scope: 2005-04-20 through 2006-11-09. This is historical traffic data, not current conditions.

Source speed is feet/second, acceleration feet/second², position feet and time milliseconds since the Unix epoch. The dataset describes 0.1-second video-derived trajectories. Vehicle IDs repeat and do not identify a single vehicle across recordings. This release preserves those limitations.

Redistributed source excerpts and adaptations retain the source attribution and CC BY-SA 3.0 terms. Changes: selected columns and three convenience cohorts, unit conversions, duplicate/conflict rejection, segmentation, causal feature construction and derived results. No endorsement by DOT/FHWA is implied. This repository does not relicense third-party material under a software license.

Raw vehicle IDs are anonymous source identifiers. Global coordinates, video, faces, license plates and private personal information are excluded. `data/manifest.json` records exact queries, retrieval times and hashes. Downloads are sequential and cached; no credentials are required.
