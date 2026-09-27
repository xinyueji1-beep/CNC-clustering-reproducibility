# Data Layout

## Asian Games Village

`asian_games_village/ayc_X.npy` is the bundled 40,000-pixel, 64-band array
used by the reproduction code. The first 54 bands are retained and
standardized before PCA. The original ENVI binary/header pair is preserved in
`asian_games_village/ayc_raw/`.

## Indian Pines

`indian_pines/Indian_pines_corrected.mat` and
`indian_pines/Indian_pines_gt.mat` are the public AVIRIS benchmark files used
by the labeled experiment. Only pixels with nonzero ground-truth labels are
used.
