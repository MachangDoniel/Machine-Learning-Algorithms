# QNA
## Difference between Machine Learning (ML) and Deep Learning (DL)

### 1. Basic Definition:
- **Machine Learning (ML)**:
  - A subset of artificial intelligence (AI) that focuses on building algorithms that allow computers to learn from data and make decisions or predictions without being explicitly programmed for specific tasks.
  - Involves using statistical models and techniques like regression, decision trees, SVM, etc.

- **Deep Learning (DL)**:
  - A subset of machine learning that uses artificial neural networks (ANNs) with multiple layers (deep networks) to model complex patterns in data.
  - It excels at tasks like image recognition, natural language processing, and other high-dimensional data problems.

### 2. Data Dependency:
- **ML**:
  - Works well with smaller datasets, provided the data is well-structured and features are properly engineered.
  - Traditional machine learning models typically require manual feature extraction or engineering to capture patterns in data.
  
- **DL**:
  - Requires large datasets to perform well. It automates feature extraction by learning hierarchical representations from the raw data.
  - The more data available, the better deep learning models tend to perform, especially for tasks like image classification and NLP.

### 3. Feature Engineering:
- **ML**:
  - Significant emphasis is placed on feature engineering, where domain experts manually select and create the most relevant features for the model to learn from.
  - Example: In a machine learning model predicting house prices, features like "number of bedrooms" or "location" would be manually chosen.

- **DL**:
  - Deep learning models automatically learn features from raw data. They can extract low-level features (like edges in images) and higher-level features (like objects) without manual intervention.
  - Example: In image recognition, the deep learning model automatically learns the relevant features from the image pixels.

### 4. Algorithms:
- **ML**:
  - Algorithms include:
    - Linear Regression, Logistic Regression
    - Decision Trees, Random Forest
    - Support Vector Machines (SVM)
    - K-Nearest Neighbors (KNN)
    - Gradient Boosting Machines (GBM), XGBoost
  
- **DL**:
  - Deep learning primarily involves neural networks, such as:
    - Convolutional Neural Networks (CNNs) for image tasks.
    - Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) for sequence and time-series tasks.
    - Generative Adversarial Networks (GANs), Transformer models for NLP tasks like BERT, GPT.

### 5. Model Interpretation:
- **ML**:
  - Many traditional ML models (like decision trees, linear regression) are easier to interpret and explain. You can trace why a certain prediction was made based on the input features.
  - Example: In a decision tree, you can explain the decision path by following the tree’s branches.
  
- **DL**:
  - Deep learning models are often called "black boxes" because it's difficult to understand or explain how they arrived at specific predictions. They involve many layers and neurons that are challenging to interpret.
  - However, techniques like SHAP values, LIME, and saliency maps help improve model interpretability.

### 6. Computational Power:
- **ML**:
  - Traditional machine learning algorithms require less computational power and can run on standard CPUs.
  - Example: Training a logistic regression model or decision tree can be done on a standard computer.

- **DL**:
  - Deep learning models, especially large neural networks, require significant computational power, typically leveraging GPUs or TPUs for efficient training.
  - Example: Training a large CNN for image classification may take hours or days on a high-end GPU.

### 7. Problem Solving:
- **ML**:
  - Works well for structured, tabular data (like databases) and problems where domain-specific knowledge can guide feature selection.
  - Example: Predicting customer churn, loan default risk, or sales forecasting.

- **DL**:
  - Particularly useful for unstructured data such as images, audio, and text, and excels in areas like computer vision, speech recognition, and NLP.
  - Example: Object detection in images, machine translation, or generating music with neural networks.

### 8. Training Time:
- **ML**:
  - Typically has shorter training times compared to deep learning, especially for simpler models like linear regression or decision trees.
  
- **DL**:
  - Training can take much longer, especially for large datasets or complex architectures. However, with advancements in hardware (GPUs/TPUs), training times are becoming more manageable.

### 9. Applications:
- **ML**:
  - Commonly used in applications like:
    - Fraud detection
    - Recommendation systems
    - Predictive maintenance
    - Risk modeling
    - Simple classification tasks

- **DL**:
  - Found in more complex, data-rich domains like:
    - Image recognition (e.g., facial recognition)
    - Natural Language Processing (e.g., chatbots, language translation)
    - Autonomous driving
    - Voice assistants (e.g., Siri, Alexa)

### Summary:
- **ML**: Involves traditional algorithms that work well on structured data, require less data, and often involve manual feature engineering.
- **DL**: Involves neural networks with multiple layers, thrives on large datasets, automates feature extraction, and is particularly suited for unstructured data (like images, audio, text).


## Difference between ROC anf AUC graph

### Receiver Operating Characteristic (ROC) Curve

**Definition:**  
The ROC curve is a graphical representation used to assess the performance of a binary classification model at various threshold levels. It illustrates the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate) across different decision thresholds.

---

**Key Points:**

- **Axes:**
  - **X-Axis:** False Positive Rate (FPR) - The proportion of actual negatives that are incorrectly classified as positives. It is calculated as:
    \[
    \text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
    \]
  - **Y-Axis:** True Positive Rate (TPR) - The proportion of actual positives correctly classified as such (also known as sensitivity or recall). It is calculated as:
    \[
    \text{TPR} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
    \]

- **Curve Interpretation:**
  - Each point on the ROC curve represents a different threshold for classifying the positive class.
  - A curve closer to the top-left corner indicates a better-performing model, as it achieves a high true positive rate while maintaining a low false positive rate.

- **Area Under the Curve (AUC):**
  - The area under the ROC curve (AUC) quantifies the overall performance of the model. AUC values range from 0 to 1:
    - **AUC = 0.5:** No discrimination; the model performs no better than random guessing.
    - **AUC = 1:** Perfect discrimination; the model correctly classifies all instances.

- **Use Cases:**
  - ROC curves are widely used in various domains, including medical diagnostics, credit scoring, and machine learning model evaluation, especially when dealing with imbalanced datasets.

### Example:
1. **Generating ROC Curve:**
   - Suppose you have a binary classifier that predicts whether a patient has a disease. You can calculate TPR and FPR at different thresholds and plot these points to create the ROC curve.
   
2. **Evaluating Performance:**
   - If the ROC curve for your model shows a high AUC (e.g., 0.85), it indicates that your model is effective at distinguishing between the positive and negative classes across various thresholds.

### Area Under the Curve (AUC)

**Definition:**  
AUC (Area Under the Curve) refers to the area under the ROC (Receiver Operating Characteristic) curve, which quantifies the performance of a binary classification model. It measures the model's ability to distinguish between positive and negative classes.

---

**Key Points:**

- **ROC Curve Context:**  
  AUC is derived from the ROC curve, which plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

- **AUC Value Interpretation:**
  - **Range:** AUC values range from 0 to 1.
    - **AUC = 0.5:** Indicates no discrimination ability; the model performs no better than random guessing.
    - **AUC < 0.5:** Indicates worse performance than random guessing.
    - **AUC = 1:** Indicates perfect discrimination; the model correctly classifies all positive and negative instances.
  - A higher AUC value indicates better model performance, suggesting a higher true positive rate with a lower false positive rate.

- **Advantages of AUC:**
  - Provides a single metric to summarize the model's performance across all possible thresholds.
  - Particularly useful in evaluating models on imbalanced datasets, where one class is more frequent than the other.

- **Use Cases:**  
  AUC is commonly used in various fields, including:
  - Medical diagnostics (e.g., disease detection)
  - Credit scoring
  - Machine learning model evaluation
  - Any scenario where understanding the ranking of predictions is essential.

### Example:
In a binary classification task, if a model has an AUC of 0.85, it means there is an 85% chance that the model will rank a randomly chosen positive instance higher than a randomly chosen negative instance.

### Ref: 
  - https://www.youtube.com/watch?v=4jRBRDbJemM
