# Predicting Car Insurance Claims Using Machine Learning
## Overview
This project explores machine learning models for predicting the number of claims (ClaimNb) in a French Motor Third-Party Liability insurance dataset. The dataset includes risk features for approximately 680,000 policies, such as driver age, vehicle charactreristics, enviromental factors.

We compare three modeling approaches:
1. **Decision Tree Regression** (from-scratch and 'scikit-learn')
2. **Feed-Forward Neural Network** (from-scratch and PyTorch)
3. **Gradient Boosting Regression (XGBoost)**

The goal is to provide interpretable, accurate models for predicting claims while accounting for the highly skewed and zero-inflated nature of insurance data.

Evaluation metrics include **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)** and **R²**.

---
## Dataset
- **Sourse:** Public French Motor Third-Party Liability dataset
- **Split**: 80% training, 20% test
- **Features:** 'IDPol', 'ClaimNb', 'Exposure', 'VehBrand', 'VehGas', 'VehPower', 'VehAge', 'DrivAge', 'Area', 'Density', 'Region', 'BonusMalus'
- **Target variable:** 'ClaimNb' - the number of claims per policy

**Preprocessing steps:**
- Log transformation for numerical features ('VehAge', 'DrivAge', 'Density')
- Categorical encoding ('VehBrand', 'VehGas', 'Area', 'Region')
- PCA on selected continuous features

## Project Structure
The repository is organized into notebooks corresponding to each stage of the analysis and each modeling approach:

```
ML-Project/
├── Project_description_and_data/
│   └── ML_Project_Proposal_2025.pdf   # Project Description
│   └── claims_train.csv               # Training dataset (80%)
│   └── claims_test.csv                # Test dataset (20%)

├── preprocessing/
│   ├── data_cleaning.ipynb            # Data cleaning and preprocessing
│   ├── density_log_scaling.ipynb      # Log-scaling of numerical features
│   ├── transform.ipynb                # Feature transformations
│   ├── transform.py                   # Helper transformation functions
│   └── exploratory_data_analysis.ipynb

├── pca_and_clustering/
│   ├── pca_clustering_scraps.ipynb    # PCA exploration
│   ├── clustering.ipynb               # Initial DBSCAN clustering
│   └── clustering_final.ipynb         # Final clustering results

├── models/
│   ├── decision_tree.ipynb            # Decision Tree regressor (from scratch + reference)
│   ├── ffnn_scratch.ipynb             # Feed-forward neural network (from scratch)
│   ├── NN.ipynb                       # Neural network (reference implementation)
│   └── gradient_boost.ipynb           # Gradient Boosting (XGBoost)

├── img/
│   ├── pc1_pc2.png
│   ├── pc1_pc3.png
│   ├── pc2_pc3.png
│   ├── pc_values.png
│   ├── shap1.png
│   └── shap2.png

├── LICENSE
└── README.md
```

## Contributors
- Anna Pólya (apol@itu.dk)
- Dimitra Filareti Tsairi (dimt@itu.dk)
- Hanna Karátson (hanka@itu.dk)
This project was developed as a group submission for a Machine Learning exam.

## Code References & Acknowledgements
Parts of this project were inspired by or adapted from the following sourses:
- **Teaching Assistant (TA) provided solutions and examples**, which were used as guidance for structuring models, preprocessing pipelines and evaluation logic
- **Artificial Intelligence tools** were used to assist with code debugging, refactoring, improving clarity of implementations.
- Standard machine learning libraries and documentation

All external inspiration was used for learning and guidance purposes.
The final implementations and analysis reflect our own understanding and decisions.





