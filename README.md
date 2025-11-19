# Human Activity Recognition Using Wrist-Worn Accelerometer Data

This repository contains our implementation of a Human Activity Recognition system using the CAPTURE-24 dataset. The project evaluates classical machine learning, deep learning, and temporal modeling techniques to classify free-living human activities from wrist-worn accelerometer data sampled at 100 Hz. 



---

## Project Overview

The goal of this project is to classify everyday human activities such as sleep, sitting, standing, walking, bicycling, household tasks, and manual work using high-frequency accelerometer signals collected in a free-living setting.

### Dataset

We use the CAPTURE-24 dataset, which includes:

- 151 participants  
- 24-hour continuous recordings per participant  
- 100 Hz tri-axial accelerometer signals  
- Ground truth labels from wearable cameras and time-use diaries  
- Activity labels mapped to 8–10 higher-level categories  

---

## Pipeline

### Preprocessing  
The data is mapped from CPA labels to the WillettsSpecific2018 schema, segmented into non-overlapping 10 second windows, cleaned, and split into training and test sets based on participant IDs.  

### Feature Engineering  
Thirty-two statistical and spectral features are extracted from each window, including percentiles, spectral entropy, dominant frequencies, and peak features. These are used for classical machine learning models.

### Models  
We implemented and evaluated:

- XGBoost on engineered features  
- A 1D Convolutional Neural Network on raw windows  
- Hidden Markov Model smoothing for temporal consistency  
- An exploratory LSTM model for short-range sequence modeling  

---

## Results Summary

| Model          | Accuracy | Weighted F1 | Notes |
|----------------|----------|-------------|-------|
| XGBoost        | 71%      | 69%         | Strong on common activities |
| CNN            | 74%      | 72%         | Best independent-window performance |
| CNN + HMM      | 72%      | Improved sequence realism | Reduced rapid switching |
| LSTM           | 55.17%   | Low macro F1 | Trained on subset, needs scaling |

---


