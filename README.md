# Smart Grid Fault & Cyberattack Detection using Deep Learning

A unified temporal intrusion detection framework for smart power grids,
trained on the MSU-ORNL benchmark dataset (15 attack scenarios, 78,377 samples).

## Overview
Smart power grids are increasingly vulnerable to sophisticated cyberattacks, including False Data Injection Attacks (FDIA),relay setting changes and remote command injections.Accurate and interpretable detection of such threats is
critical for maintaining grid stability and operator trust.This project presents a unified temporal intrusion detection framework trained on all 15 merged scenarios of the
publicly available MSU-ORNL power grid dataset.In contrast to prior work that evaluates each dataset scenario independently, we train a single Long Short-Term Memory (LSTM) model on the combined 78,347-sequence
corpus, enabling cross-scenario generalisation.As a comparative baseline, Random Forest (RF) and Decision Tree (DT) models are evaluated on each of the 15 individual datasets using the top-40 features selected via RF importance ranking.A PyTorch Transformer model is additionally trained on the merged data to benchmark performance.

## Models Implemented
- Random Forest & Decision Tree — per individual dataset (15 × baseline)
- LSTM (PyTorch) — unified model on all 15 merged datasets
- Transformer (PyTorch) — unified model on all 15 merged datasets
- Saliency Map — gradient-based explainability for LSTM

## Key Results
| Model | Data Setup | Accuracy | F1 |
|---|---|---|---|
| Random Forest | Per-dataset (15 individual) | 93–97% | 0.95 |
| LSTM | All 15 merged | 98.12% | 0.97 |
| Transformer | All 15 merged | 98.19% | 0.98 |

## Dataset
MSU-ORNL Power Grid Dataset — Mississippi State University +
Oak Ridge National Laboratory. 15 sub-datasets, 128 features,
3 classes: Attack / Natural / No Event.

Download: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

## Novelty
- First unified temporal model trained on all 15 MSU-ORNL scenarios combined
- Saliency Map on LSTM reveals WHICH sensor and at which timestep attack occurs
- Direct comparison: per-dataset ML (Naeem et al. style) vs unified DL

## Tech Stack
Python • PyTorch • scikit-learn • pandas • SHAP • matplotlib

## Paper
Conference paper submitted to [venue name] — 2025
