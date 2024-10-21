# Feature Extraction

## Principal Component Analysis (PCA)

### Introduction
**Principal Component Analysis (PCA)** is a dimensionality reduction technique widely used in data analysis and machine learning. Its goal is to reduce the number of features in a dataset while retaining as much variance (information) as possible. PCA is particularly useful when dealing with high-dimensional data, as it helps simplify the data without losing essential patterns.

### Dimensionality Reduction
PCA transforms a dataset with many variables (features) into a smaller set of variables called **principal components**. These components capture most of the variability in the data, helping to simplify analysis and computation.

### Variance and Principal Components
- The **first principal component** captures the largest amount of variance in the data.
- Each subsequent principal component captures the maximum remaining variance under the constraint that it is orthogonal (uncorrelated) to the previous components.

### Orthogonality
Each principal component is **orthogonal** to the others, meaning they are uncorrelated. This ensures that each component represents distinct patterns in the data.

<h4>Orthogonal Matrix</h4>
<p>If we consider the matrix <strong>W</strong> composed of principal components, the orthogonality condition can be expressed as:</p>
<pre><strong>W</strong><sup>T</sup> <strong>W</strong> = <strong>I</strong></pre>
<p>where:</p>
<ul>
    <li><strong>W</strong><sup>T</sup> is the transpose of matrix <strong>W</strong>,</li>
    <li><strong>I</strong> is the identity matrix.</li>
</ul>
<p>This equation states that the product of the transpose of the matrix of principal components and the matrix itself results in the identity matrix, confirming that the columns of <strong>W</strong> (the principal components) are orthogonal to each other.</p>


### Eigenvectors and Eigenvalues
PCA is based on the **eigenvectors** and **eigenvalues** of the data’s covariance matrix:
- **Eigenvectors** represent the direction of the principal components.
- **Eigenvalues** indicate the amount of variance explained by each principal component.

<h4>Covariance Matrix</h4>
<p>Given a dataset represented as a matrix X of size n x p (where n is the number of observations and p is the number of features), the covariance matrix C is calculated as:</p>
<pre>C = (1 / (n - 1)) * (X<sup>T</sup> * X)</pre>

<h4>Eigenvalue Problem</h4>
<p>To find the principal components, we solve the eigenvalue problem represented by:</p>
<pre>C * v = λ * v</pre>
<p>This can also be expressed as:</p>
<pre>A * x = λ * x</pre>
<p>Where:</p>
<ul>
    <li>A is the covariance matrix C,</li>
    <li>x is the eigenvector (the direction of the principal component),</li>
    <li>λ is the corresponding eigenvalue (the variance explained by that component).</li>
</ul>

### Steps of PCA

1. **Standardization**: Standardize the data if features have different units.
2. **Covariance Matrix**: Calculate the covariance matrix of the data.
3. **Eigen Decomposition**: Compute the eigenvectors and eigenvalues of the covariance matrix.
4. **Sort Components**: Sort the principal components by the amount of variance they explain (largest to smallest).
5. **Projection**: Project the original data onto the principal components to obtain the reduced dataset.

<p>The projection of the original dataset X onto the principal components can be represented as:</p>
<pre>Z = X * W</pre>
<p>Where:</p>
<ul>
    <li>Z is the transformed dataset (reduced dimension),</li>
    <li>W is the matrix of selected eigenvectors (principal components).</li>
</ul>

### Applications of PCA

- **Data Visualization**: PCA can reduce high-dimensional data to 2 or 3 dimensions, making it easier to visualize.
- **Noise Reduction**: PCA can help remove less important components (those with low variance) that may be noise.
- **Feature Selection**: PCA can reduce the number of features in a dataset before applying machine learning algorithms.
- **Preprocessing**: PCA is often used as a preprocessing step for machine learning tasks, especially when working with high-dimensional datasets.

### PCA and Machine Learning
PCA is not a machine learning algorithm itself, but it is often used as a **preprocessing** step in machine learning workflows:
- It helps in **dimensionality reduction**, which can lead to faster model training and less overfitting.
- It can transform features into uncorrelated principal components, which may improve model performance.

### Conclusion
PCA is a powerful tool for simplifying data, reducing noise, and preparing datasets for further analysis or machine learning tasks. By reducing the number of dimensions while retaining most of the important information, PCA helps make data more manageable without compromising too much on accuracy.


### References
  - https://www.kaggle.com/code/avikumart/pca-principal-component-analysis-from-scratch
  - https://www.kaggle.com/code/ryanholbrook/principal-component-analysis
  - https://www.kaggle.com/code/tarkkaanko/pca-principal-component-analysis-cancer-dataset
  - https://www.kaggle.com/code/nirajvermafcb/principal-component-analysis-explained
  - https://www.youtube.com/watch?v=FgakZw6K1QQ
  - https://www.youtube.com/watch?v=iRbsBi5W0-c
  - https://www.youtube.com/watch?v=tXXnxjj2wM4
  - https://www.youtube.com/watch?v=tofVCUDrg4M
  - https://www.youtube.com/watch?v=WDjzgnqyz4s
  - https://towardsdatascience.com/principal-component-analysis-pca-79d228eb9d24

---

## Independent Component Analysis (ICA)

### Introduction
**Independent Component Analysis (ICA)** is a technique used to decompose multivariate data into components that are statistically independent. It is primarily used for **blind source separation** (BSS), where the goal is to extract individual signals from a mixture of signals. Unlike PCA, which focuses on maximizing variance, ICA aims to achieve statistical **independence** between components.

### Independence and Signal Separation
ICA tries to find components that are **statistically independent**, meaning that knowing the value of one component does not provide any information about the others. This makes ICA particularly useful in situations where signals are mixed together, such as in signal processing and neuroscience.

### Contrast with PCA
- **PCA**: Finds uncorrelated components that maximize variance.
- **ICA**: Finds independent components that may not necessarily capture the most variance but are statistically independent.

### Blind Source Separation
ICA is often applied in **blind source separation**, where mixed signals (e.g., overlapping voices or EEG signals) are separated into their original independent sources. A common example is the **cocktail party problem**, where ICA can separate different speakers’ voices from a recording of a crowded room.

### Steps of ICA

1. **Centering and Whitening**: Standardize the data to remove mean and scale variance.
2. **Maximizing Independence**: Identify the transformation that maximizes the independence of the components.
3. **Signal Extraction**: Recover the independent components that explain the mixed signals.

### Applications of ICA

- **Signal Processing**: ICA can separate overlapping signals, such as in audio recordings or EEG data.
- **Neuroscience**: ICA is used to separate independent sources of brain activity from EEG or fMRI signals.
- **Image Processing**: Used to extract features from images by separating out different components (e.g., background vs. foreground).

### ICA and Machine Learning
ICA is not primarily used as a machine learning algorithm, but it can be a valuable tool in **feature extraction** or **preprocessing**:
- It can be used to remove noise from datasets by separating independent signal sources.
- ICA-derived features can be fed into machine learning models for improved classification or regression accuracy.

### Conclusion
ICA is an advanced technique for uncovering independent signals or components from mixed data. It is particularly valuable in fields like signal processing and neuroscience, where separating independent factors is crucial. Unlike PCA, ICA prioritizes independence over variance, offering a different perspective on feature extraction.

### References
  - https://www.kaggle.com/code/chittalpatel/ica-the-musical-way
  - https://www.kaggle.com/code/chrisfilo/independent-component-decomposition-ica
  - https://www.kaggle.com/code/tarunchilkur/ica-for-eeg
  - https://www.youtube.com/watch?v=GgLaP4Des1Q
  - https://towardsdatascience.com/independent-component-analysis-ica-a3eba0ccec35

---
---

# Feature Selection

## Correlation-Based Techniques

### Introduction
**Correlation-based techniques** measure the strength and direction of relationships between variables in a dataset. These techniques are often used in feature selection, data analysis, and multicollinearity detection. Correlation quantifies how closely two variables are related in a linear sense.

### Types of Correlation

- **Positive Correlation**: As one variable increases, the other also increases.
- **Negative Correlation**: As one variable increases, the other decreases.
- **No Correlation**: No significant relationship between variables.

### Correlation Coefficient
The **correlation coefficient** is a numerical representation of the strength of the relationship between two variables. The most commonly used measure is **Pearson’s correlation coefficient**, which ranges from `-1` (perfect negative correlation) to `1` (perfect positive correlation), with `0` indicating no linear relationship.

- **Pearson Correlation**: Measures the linear relationship between two continuous variables.
- **Spearman’s Rank Correlation**: Measures the strength of the relationship based on the ranks of the variables, useful when the relationship is not linear.
- **Kendall’s Tau**: A non-parametric measure of correlation for ordinal data.

### Applications of Correlation-Based Techniques

- **Feature Selection**: Correlation-based techniques are used to identify features that have a strong relationship with the target variable or to eliminate features that are highly correlated with each other (to avoid multicollinearity).
- **Data Analysis**: Correlation is widely used to explore relationships between variables in fields like finance, healthcare, and social sciences.
- **Multicollinearity Detection**: In regression analysis, highly correlated features can cause multicollinearity, which makes it difficult for models to determine the individual effects of each feature.

### Steps in Correlation Analysis

1. **Calculate Correlation Coefficients**: Compute the correlation matrix for the dataset, showing relationships between pairs of features.
2. **Evaluate Correlation Strength**: Identify features with high correlation (positive or negative) with the target variable.
3. **Multicollinearity Check**: Remove or combine highly correlated features to improve model robustness.

### Correlation-Based Techniques and Machine Learning
Correlation-based techniques are commonly used in the **preprocessing** stage of machine learning:
- **Feature Selection**: Strongly correlated features can be dropped or transformed to improve model accuracy.
- **Multicollinearity Reduction**: Removing highly correlated features helps prevent overfitting and improves interpretability of models like linear regression.

### Conclusion
Correlation-based techniques provide an easy-to-interpret method for exploring relationships between features in a dataset. By understanding the correlation between variables, data scientists can reduce redundancy, avoid multicollinearity, and improve the performance of machine learning models. Though correlation-based methods focus on linear relationships, they are essential for feature selection and data analysis across many fields.

### References
  - https://www.kaggle.com/code/bbloggsbott/feature-selection-correlation-and-p-value
  - https://medium.com/@sariq16/correlation-based-feature-selection-in-a-data-science-project-3ca08d2af5c6
  - https://www.researchgate.net/publication/2805648_Correlation-Based_Feature_Selection_for_Machine_Learning
  - https://www.youtube.com/watch?v=FndwYNcVe0U

## Comparison of PCA, ICA and Correlation-Based Technique

| **Feature**                     | **PCA (Principal Component Analysis)**                    | **ICA (Independent Component Analysis)**                     | **Correlation-Based Techniques**                          |
|----------------------------------|-----------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------|
| **Objective**                    | Reduce dimensionality while preserving maximum variance   | Find statistically independent components from mixed signals  | Measure strength and direction of relationships between variables |
| **Feature Type**                 | **Feature Extraction**                                    | **Feature Extraction**                                       | **Feature Selection**                                      |
| **Components**                   | Principal components that are uncorrelated                | Independent components that are statistically independent     | Correlated or uncorrelated features based on correlation coefficients |
| **Method**                       | Eigenvectors and eigenvalues of covariance matrix         | Maximizes independence between components                     | Pearson, Spearman, Kendall correlation coefficients        |
| **Type of Relationship**         | Uncorrelated components                                   | Statistically independent components                          | Measures linear or monotonic relationships                 |
| **Orthogonality**                | Principal components are orthogonal (uncorrelated)        | Independent components are not necessarily orthogonal         | Not focused on orthogonality but on strength of association |
| **Variance**                     | Captures maximum variance in the first few components     | Focuses on independence, may not capture maximum variance     | No direct focus on variance, measures association strength |
| **Main Application**             | Dimensionality reduction and feature extraction           | Blind source separation and signal processing                 | Feature selection, multicollinearity reduction, data analysis |
| **Data Transformation**          | Projects data onto new axes (principal components)        | Separates mixed signals into independent components           | Measures relationships between original features           |
| **Steps**                        | Standardization, covariance matrix, eigen decomposition   | Centering, whitening, maximization of independence            | Compute correlation matrix, evaluate relationships         |
| **Output**                       | Reduced number of principal components                    | Independent components                                         | Correlation coefficients between pairs of variables        |
| **Handling of Noise**            | Can reduce noise by focusing on major components          | Separates noise if it forms an independent component          | Not designed to handle noise directly, but can identify noisy relationships |
| **Use in Machine Learning**      | Preprocessing for dimensionality reduction and feature extraction | Feature extraction and signal denoising                      | Preprocessing for feature selection and multicollinearity reduction |
| **Common Algorithms or Techniques** | Eigen decomposition, singular value decomposition (SVD)   | FastICA algorithm                                             | Pearson correlation, Spearman's rank, Kendall's Tau       |

---

## Recursive Feature Elimination (RFE)

### Introduction
**Recursive Feature Elimination (RFE)** is a feature selection technique that recursively removes the least important features from the dataset, aiming to find the subset of features that contribute the most to model performance. RFE works by ranking features based on their importance in a machine learning model and eliminates the least important feature(s) at each iteration.

### How RFE Works
1. **Train a model**: RFE starts by training a model (e.g., decision trees, SVM) on the entire dataset.
2. **Rank features**: The model ranks features based on their importance (e.g., using coefficients or feature importance scores).
3. **Eliminate features**: The least important feature(s) are eliminated.
4. **Recursion**: The process is repeated recursively, eliminating features until the desired number of features is reached.

### Advantages of RFE
- **Improves model performance**: By selecting only the most important features, RFE can improve model efficiency and accuracy.
- **Reduces overfitting**: Eliminating irrelevant features helps reduce overfitting and makes the model more generalizable.
- **Feature importance**: RFE gives insight into which features are most relevant for a given predictive model.

### Disadvantages of RFE
- **Computational cost**: RFE can be computationally expensive, especially for large datasets with many features, as it retrains the model multiple times.
- **Dependent on base model**: The results depend heavily on the performance and nature of the base model used for feature ranking.

### Applications of RFE
- **Feature selection**: RFE is widely used to select relevant features for various machine learning algorithms like decision trees, SVMs, or logistic regression.
- **Improving interpretability**: RFE can reduce the number of features, making the model more interpretable.

### Steps in RFE
1. **Fit the model**: Fit the base model on the entire dataset.
2. **Rank the features**: Rank the features based on importance.
3. **Eliminate least important features**: Recursively eliminate features with the lowest importance score.
4. **Finalize selected features**: Continue until the desired number of features is selected.

---

## Comparison of PCA, ICA, Correlation-Based Techniques, and RFE

| **Feature**                     | **PCA (Principal Component Analysis)**                    | **ICA (Independent Component Analysis)**                     | **Correlation-Based Techniques**                          | **RFE (Recursive Feature Elimination)**                   |
|----------------------------------|-----------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|
| **Objective**                    | Reduce dimensionality while preserving maximum variance   | Find statistically independent components from mixed signals  | Measure strength and direction of relationships between variables | Select the most important features based on model performance |
| **Feature Type**                 | **Feature Extraction**                                    | **Feature Extraction**                                       | **Feature Selection**                                      | **Feature Selection**                                       |
| **Components**                   | Principal components that are uncorrelated                | Independent components that are statistically independent     | Correlated or uncorrelated features based on correlation coefficients | Ranked features based on importance to model performance    |
| **Method**                       | Eigenvectors and eigenvalues of covariance matrix         | Maximizes independence between components                     | Pearson, Spearman, Kendall correlation coefficients        | Recursive elimination of least important features           |
| **Type of Relationship**         | Uncorrelated components                                   | Statistically independent components                          | Measures linear or monotonic relationships                 | Model-based relationship (importance determined by model)   |
| **Orthogonality**                | Principal components are orthogonal (uncorrelated)        | Independent components are not necessarily orthogonal         | Not focused on orthogonality but on strength of association | Not focused on orthogonality, focuses on feature importance |
| **Variance**                     | Captures maximum variance in the first few components     | Focuses on independence, may not capture maximum variance     | No direct focus on variance, measures association strength | Selects features that contribute most to model variance     |
| **Main Application**             | Dimensionality reduction and feature extraction           | Blind source separation and signal processing                 | Feature selection, multicollinearity reduction, data analysis | Feature selection for predictive modeling                   |
| **Data Transformation**          | Projects data onto new axes (principal components)        | Separates mixed signals into independent components           | Measures relationships between original features           | Removes less relevant features from dataset                 |
| **Steps**                        | Standardization, covariance matrix, eigen decomposition   | Centering, whitening, maximization of independence            | Compute correlation matrix, evaluate relationships         | Train model, rank features, recursively eliminate features  |
| **Output**                       | Reduced number of principal components                    | Independent components                                        | Correlation coefficients between pairs of variables        | A reduced subset of the most important features             |
| **Handling of Noise**            | Can reduce noise by focusing on major components          | Separates noise if it forms an independent component          | Not designed to handle noise directly, but can identify noisy relationships | May eliminate noisy or irrelevant features                  |
| **Use in Machine Learning**      | Preprocessing for dimensionality reduction and feature extraction | Feature extraction and signal denoising                      | Preprocessing for feature selection and multicollinearity reduction | Feature selection for improving model performance           |
| **Common Algorithms or Techniques** | Eigen decomposition, singular value decomposition (SVD)   | FastICA algorithm                                             | Pearson correlation, Spearman's rank, Kendall's Tau        | Decision trees, SVM, logistic regression                    |

---

### Conclusion
Each technique—PCA, ICA, correlation-based methods, and RFE—has its strengths depending on the context:
- **PCA** is useful for dimensionality reduction.
- **ICA** excels in blind 
