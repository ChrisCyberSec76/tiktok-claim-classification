# TikTok Claims Classification Project

## Overview
This project implements a machine learning solution for TikTok to automatically classify videos as either claims or opinions. Using the PACE framework (Plan, Analyze, Construct, Execute), we developed a highly accurate classification model that achieved nearly perfect performance in distinguishing between claims and opinions.

## Business Problem
TikTok receives a large volume of user reports identifying videos that potentially contain claims. Manual review of these reports creates a significant backlog. This project aims to:
- Automate the initial screening of reported content
- Reduce backlog of user reports
- Prioritize content for manual review more effectively
- Increase response time and system efficiency

## Project Structure
```
tiktok-claims-classification/
├── data/
│   └── tiktok_dataset.csv
├── notebooks/
│   └── claims_classification.ipynb
├── README.md
└── requirements.txt
```

## PACE Framework Implementation

### Plan
- Identified key stakeholders and project requirements
- Evaluated ethical implications and error consequences
- Determined success metrics (accuracy, precision, recall)
- Assessed data reliability and availability

### Analyze
- Conducted exploratory data analysis
- Handled missing values and outliers
- Analyzed class balance (50.3% claims, 49.7% opinions)
- Identified key features for model development

### Construct
- Engineered features from video metadata and text
- Built and compared two models:
  - Random Forest
  - XGBoost
- Implemented cross-validation and hyperparameter tuning

### Execute
- Evaluated model performance
- Selected champion model (XGBoost)
- Generated recommendations
- Documented findings and next steps

## Model Performance

### Random Forest Results
- Accuracy: 1.00
- Precision: 1.00 (Opinion), 0.99 (Claim)
- Recall: 0.99 (Opinion), 1.00 (Claim)
- F1-score: 1.00

### XGBoost Results (Champion Model)
- Accuracy: 1.00
- Precision: 1.00 (Opinion), 0.99 (Claim)
- Recall: 0.99 (Opinion), 1.00 (Claim)
- F1-score: 1.00

## Key Features
1. Video engagement metrics (strongest predictors)
   - View count
   - Like count
   - Comment count
   - Share count
2. Content characteristics
   - Text length
   - Video duration
3. Account status features
   - Verification status
   - Ban status

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/tiktok-claims-classification.git
cd tiktok-claims-classification
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
- Place TikTok dataset in `data/` directory
- Follow data preprocessing steps in notebook

2. Model Training:
```python
# Load required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load and preprocess data
data = pd.read_csv("data/tiktok_dataset.csv")
# Follow preprocessing steps...

# Train model
model = xgb.XGBClassifier(...)
model.fit(X_train, y_train)
```

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

## Future Improvements
- Additional feature engineering opportunities:
  - Ratio features between engagement metrics
  - Time-based features
  - Text sentiment analysis
- Model robustness enhancements
- Integration with video content analysis
- Real-time prediction capabilities

## Acknowledgments
- TikTok (synthetic dataset)
