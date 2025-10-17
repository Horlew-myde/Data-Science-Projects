# Unsupervised Learning & Feature Engineering Projects

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=yellow)
![Scikit-learn](https://img.shields.io/badge/SciKit--Learn-1.1%2B-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-blue?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange?logo=jupyter)

## Overview

This repository showcases a curated collection of projects focused on **Unsupervised Learning** and **Advanced Feature Engineering**. These projects demonstrate the ability to find hidden patterns in data without pre-existing labels and to skillfully create new features that enhance model performance.

The notebooks cover everything from foundational clustering techniques to more complex applications like anomaly detection, dimensionality reduction, and association rule mining.

### Core Concepts & Algorithms Explored
* **Clustering**: K-Means, Hierarchical Clustering (Agglomerative)
* **Dimensionality Reduction**: Principal Component Analysis (PCA)
* **Anomaly Detection**: Isolation Forest
* **Association Rule Mining**: Apriori Algorithm
* **Feature Engineering**: Polynomial Features, Interaction Terms, Binning/Discretization
* **Data Preprocessing**: `StandardScaler`, One-Hot Encoding
* **Model Evaluation**: Elbow Method (WCSS), Dendrograms, $R^2$ Score Improvement

---

## Project Showcase

Here is a breakdown of the projects included in this repository.

### 11. Customer Segmentation (K-Means)
* **Objective**: To segment mall customers into distinct groups based on their spending habits.
* **Techniques**: K-Means Clustering, exploratory data analysis (EDA), and the **Elbow Method** to determine the optimal number of clusters ($k$).
* **Outcome**: Identified 5 distinct customer personas (e.g., "High Income, Low Spending" vs. "Low Income, High Spending"), providing actionable insights for targeted marketing campaigns.

### 12. Image Compression (K-Means)
* **Objective**: To reduce the color palette of an image, effectively compressing its data size.
* **Techniques**: K-Means Clustering applied to pixel RGB values.
* **Outcome**: Successfully compressed a full-color image to a 16-color palette. This project is a powerful visual demonstration of how clustering can be used in a non-obvious, creative domain.

### 13. Hierarchical Clustering
* **Objective**: To apply an alternative clustering method to the mall customer dataset and visualize the nested relationships between data points.
* **Techniques**: Agglomerative Hierarchical Clustering, **Dendrogram** visualization.
* **Outcome**: The dendrogram clearly suggested an optimal 5-cluster solution, confirming the findings from the K-Means project and demonstrating proficiency in multiple clustering approaches.

### 14. Dimensionality Reduction (PCA)
* **Objective**: To visualize a high-dimensional dataset (64 dimensions) in a 2D space.
* **Techniques**: Principal Component Analysis (PCA), `StandardScaler`.
* **Outcome**: Successfully projected the 64-dimensional (8x8 pixels) handwritten digits dataset onto just two principal components. The resulting 2D scatter plot clearly showed that digits of the same class (e.g., all "0"s, all "1"s) naturally clustered together, proving the effectiveness of PCA in capturing the most important variance.

### 15. Advanced Feature Engineering
* **Objective**: To improve the predictive accuracy of a linear regression model by engineering more powerful features.
* **Techniques**: Applied to the California Housing dataset, this project involved creating **Polynomial Features**, **Interaction Terms**  and **Binning** .
* **Outcome**: The model built with engineered features showed a significant **improvement in its $R^2$ score** compared to a baseline model, proving that thoughtful feature creation is critical for model performance.

### 16. Anomaly Detection (Isolation Forest)
* **Objective**: To identify rare outliers in a dataset.
* **Techniques**: **Isolation Forest** algorithm on a synthetic dataset designed with clear anomalies.
* **Outcome**: The model successfully identified and flagged the anomalous data points, demonstrating a key skill used in applications like credit card fraud detection and system health monitoring.

### 17. Market Basket Analysis (Apriori)
* **Objective**: To find frequently co-purchased items in a grocery transaction dataset.
* **Techniques**: **Apriori Algorithm** (using `mlxtend`) to generate **association rules** 
* **Outcome**: Discovered actionable "if-then" purchasing rules, measuring them by **Support**, **Confidence**, and **Lift**. These rules can be used by a retailer to optimize store layout, create bundles, and plan "buy-one-get-one" promotions.

---

## Technology Stack

* **Python**: Core programming language.
* **Pandas**: For data loading, manipulation, and preprocessing.
* **NumPy**: For numerical operations and data reshaping.
* **Scikit-learn (sklearn)**: For KMeans, PCA, AgglomerativeClustering, IsolationForest, StandardScaler, and train_test_split.
* **Matplotlib & Seaborn**: For all data visualizations (scatter plots, dendrograms, etc.).
* **`mlxtend`**: For implementation of the Apriori algorithm.
* **Jupyter Notebooks**: For interactive development and analysis.

## Getting Started

To run these projects locally:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  (Optional but recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install the required packages:
    *(**Note**: You should create a `requirements.txt` file for this folder. You can do this by running `pip freeze > requirements.txt` in your terminal after installing the libraries below).*
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn mlxtend jupyter
    ```
4.  Launch Jupyter Lab and explore the notebooks:
    ```bash
    jupyter lab
    ```
