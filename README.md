# FocusMirror ML System 
 
A prototype machine-learning pipeline for detecting fatigue/burnout-related states from FocusMirror behavioral signals, currently bootstrapped with synthetic/proxy training data. 
 
## Important Scientific Limitation 
This repository contains a development prototype designed to establish an end-to-end ML training and inference pipeline. Because real-world burnout sessions are limited during initial development, synthetic/proxy data is used as a bootstrap mechanism (data_source = 'synthetic'). 
 
Do NOT describe or deploy this model as a clinically validated burnout detector. 
 
## Architecture 
- src/: Core modular ML pipeline components. 
- models/: Saved joblib scikit-learn pipelines. 
- data/: Local dataset storage. 
- tests/: Automated pipeline validation tests.
