# TikTok Claims Classification

## Project Overview
A machine learning project that classifies TikTok videos as either claims or opinions to help prioritize content moderation. The model achieves near-perfect accuracy in distinguishing between claims and opinions using video engagement metrics and content characteristics.

## Features
- Automated classification of TikTok videos
- Feature importance analysis
- Comparison of Random Forest and XGBoost models
- Evaluation metrics and visualizations

## Model Performance
- Accuracy: 1.00
- Precision: 1.00 (Opinion), 0.99 (Claim)
- Recall: 0.99 (Opinion), 1.00 (Claim)
- F1-Score: 1.00 for both classes

## Key Findings
- Video view count is the most important predictor
- Engagement metrics are highly predictive of claim vs. opinion content
- Both Random Forest and XGBoost models achieve exceptional performance
- XGBoost slightly outperforms Random Forest as the champion model

## Installation
1. Clone this repository:
```bash
git clone https://github.com/your-username/tiktok-claims-classification.git
