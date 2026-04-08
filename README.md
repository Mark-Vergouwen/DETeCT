# DETeCT: Data-Efficient low-carbon Technology Classification for multiclass Targets
Repository for data and code for: Rigaux, B. & Vergouwen, M., (2025). DETeCT: Data-Efficient low-carbon Technology adoption Classification for multiclass Targets

## 🚀 Project Overview
DETeCT presents a data-driven framework to classify the adoption of low-carbon technologies (LCTs), such as electric vehicles, heat pumps, and solar PV, using high-frequency smart meter data. Combining domain-informed and time-series features within a Random Forest and stacking model, the approach achieves robust predictive performance on large-scale household data from Flanders.

This repository contains Python code to replicate the results and implement DETeCT. A corresponding data repository is available on Zenodo. All data used in this project is published by Fluvius (2025), the Flemish Distribution System Operator, under an open data license that permits reproduction, redistribution and reuse (https://opendata.fluvius.be/p/licentieopendatafluvius/).

- **Notebooks:** exploratory analysis and modeling in Python (`notebooks/`)  
- **Reusable code:** functions and classes in base Python (`src/`)  
- **Data and outputs:** raw data and output live outside this repo and is publicly available via Zenodo: https://doi.org/10.5281/zenodo.19388929

## 📦 Features

🔍 **Multiclass classification of LCT adoption**
- Detects EVs, heat pumps, PV systems, and their combinations

⚙️ **Advanced feature engineering**
- Domain-agnostic features (tsfeatures, Fourier based spectral features)
- Domain-informed features (IFEEL, SmartMeterAnalytics)

🧠 **Machine learning pipeline**
- Random Forest classifiers 
- Stacking ensemble 
- Elaborate robustness and external validation 

📊 **Data-efficient detection**
- Works with aggregations to weekly, monthly, and yearly data
- Evaluates trade-off between timeliness and predictive performance

## How to work with DETeCT? 
To replicate the full model architecture with ground-truth labelled data, follow steps 1 - 5. 
To apply DETeCT to your own ground-truth labelled electricity consumption data, follow steps 1 - 5, replacing the data from Zenodo with your own data.
To apply DETeCT to your own electricity consumption data, skip step 4, and directly predict using the hyperparameter tuned models from Zenodo.

1. Download the data and/or machine learning architecture files from Zenodo. https://doi.org/10.5281/zenodo.19388929
2. Download the GitHub archive 
3. Preprocess input 15-minute electricity consumption data using the workflow in `notebooks/PRE_workflow.py`. This applies preprocessing and feature engineering to the input data.
4. Train and hyperparameter tune the Random Forest and stacking classifier models using the functionality in `notebooks/MODEL_RandomForest.py` and `notebooks/MODEL_Stacking.py`
5. Generate model-predictions and evaluate out-of-sample performance using `notebooks/MODEL_Postestimation.py`

## References 
Rigaux, B., & Vergouwen, M. (2026). DETeCT: Data-Efficient low-carbon Technology Classification for multiclass Targets [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19388929

Fluvius (2025). Verbruiksprofielen digitale elektriciteitsmeters: kwartierwaarden voor een volledig jaar. 
https://opendata.fluvius.be/explore/dataset/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/information/ 

 

