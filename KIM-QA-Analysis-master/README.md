# KIM-QA-Analysis
This repository contains codes used for KIM QA analysis using the 6 DoF robot. Three QA tests are performed to validate KIM's tracking accuracy and repeatability: 

1. Static localisation test
2. Dynamic localisation test
3. Treatment Interruption test

Details of the analysis procedure can be found in the attached document: KIM QA publication.pdf.

As different linacs have different couch shift conventions, the codes have been put in respective folders. See the instructions in respective folders. 

## Python Static Localisation Refactor

The MATLAB static localisation workflow in `Elekta/App_Static_loc.mlapp` and `Elekta/Staticloc.m` now has a Python counterpart at `python/static_localization.py`. The script ingests raw centroid files (no manual cleaning required) plus the KIM trajectory folder, computes the same mean/std/percentile metrics, and optionally renders the trace plot. Example:

```bash
python3 python/static_localization.py \
  --kim-folder "Lyrebird Data/Static Test" \
  --centroid "Lyrebird Data/Centroid_248687_BeamID_3.1_3.2.txt" \
  --lr 0 --si 0 --ap 0
```

Run `python3 python/static_localization.py --help` for all options (frame averaging, output overrides, etc.).
