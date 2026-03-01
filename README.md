# Twitter Sentiment Analysis Framework

## Project Overview
This project implements a comprehensive machine learning pipeline to classify Twitter messages into positive or negative sentiments. The analysis covers the entire lifecycle, from advanced feature engineering and dimensionality reduction to the evaluation of multiple classification and clustering models.

## Key Pipeline Stages
* **Exploratory Data Analysis (EDA)**: Statistical distribution analysis of user engagement metrics (likes, retweets, followers).
* **Feature Engineering**: Extraction of 26 distinct features, including textual metrics (sentiment ratios, word counts), user-based metrics (engagement rates), and temporal patterns.
* **Dimensionality Reduction**: Implementation of PCA (Principal Component Analysis) to identify core emotional and technical dimensions in the data.
* **Model Training & Optimization**:
    * **Supervised Learning**: Comparative analysis of Decision Trees, Multi-Layer Perceptron (MLP), and Support Vector Machines (SVM).
    * **Unsupervised Learning**: K-Means clustering (K=7) based on behavioral patterns.
* **Performance Enhancements**: Utilizing SMOTE for class balancing and applying Dropout/Batch Normalization to stabilize neural network training.

## Results Summary
* The **MLP (Multi-Layer Perceptron)** model achieved the best balance with a validation accuracy of ~89% and an AUC score of 0.96.
* Feature importance analysis revealed that `Engagement Rate` and `Hashtags Count` were the strongest predictors of sentiment.

## Technical Stack
* **Language**: Python (Jupyter Notebook).
* **Libraries**: Pandas, Scikit-learn, Matplotlib, Seaborn, NumPy.
