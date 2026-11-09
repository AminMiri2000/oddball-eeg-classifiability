"""Subject classifiability in oddball EEG.

Reusable analysis code accompanying *Quantifying Subject Classifiability in
Oddball EEG: A Cross-Dataset Methodological Comparison of Signal Quality and
Classifier Effects*.

Modules
-------
paths          Project root, results and data directory resolution.
config         Experiment dataclasses and the experiment registry.
bids           Locating, reading, preprocessing and epoching the BIDS datasets.
loader         Chronological trial reconstruction and chunk grouping.
preprocessing  Cropping, label permutation, normalization, PyTorch datasets.
features       Raw-flat and time-binned features for the classical models.
splits         Purged, grouped chronological cross-validation.
models         Shrinkage LDA, logistic regression and EEGNet.
training       Fold-level evaluation of the classical models and EEGNet.
metrics        Binary classification metrics.
summaries      Fold -> subject -> model aggregation.
erp            ERP summaries and plots.
time_resolved  Sliding-window decoding.
stats          FDR correction and correlation helpers.
io             Saving tables, histories and metadata.
visualization  Notebook display and plotting helpers.
"""

__version__ = "1.0.0"
