## Linear Discriminant Analysis (LDA):

<img src="images/LDA.png" alt="LDA" width="600" height="400">

- **Type:** Probabilistic, linear classifier.
- **Approach:** LDA assumes that the data from each class is normally distributed, and it works by modeling the distribution of the data in each class and finding a linear combination of features that separates the classes well. It tries to maximize the distance between means of classes while minimizing the variation within each class.
- **Assumptions:**
    - Assumes that the data for each class comes from a Gaussian (normal) distribution.
    - Assumes that all classes share the same covariance matrix.
- **Output:** Gives probabilistic class membership predictions.
- **Efficiency:** Works well for problems where class distributions are approximately Gaussian and can handle small datasets well.
- **Use Cases:** Suitable when the data is linearly separable and normally distributed with shared covariances. It is efficient when there's a small sample size compared to the number of features.
- **Ref:**
    - https://en.wikipedia.org/wiki/Linear_discriminant_analysis

---
## Quadratic Discriminant Analysis (QDA):

<img src="images/QDA.png" alt="LDA" width="600" height="400">

- **Type:** Probabilistic, non-linear classifier.
- **Approach:** QDA models the data distribution of each class separately and finds a quadratic decision boundary. Unlike LDA, QDA allows for different covariance matrices for each class, which makes it more flexible when dealing with data that is not linearly separable.
- **Assumptions:**
  - Assumes that the data for each class comes from a Gaussian (normal) distribution.
  - Allows different covariance matrices for each class, thus capturing more complex relationships in the data.
- **Output:** Provides probabilistic class membership predictions.
- **Efficiency:** More computationally expensive than LDA due to the separate covariance matrices for each class, but better suited for cases where the decision boundary between classes is non-linear.
- **Use Cases:** Suitable when data is not linearly separable and has varying covariance structures between different classes. Often used in cases where a quadratic decision boundary is required.
- **Ref:**  
    - https://www.kaggle.com/discussions/general/448328

---
## Naive Bayes:
- **Type:** Probabilistic classifier.
- **Approach:** Naive Bayes is based on Bayes' Theorem and assumes that the features are conditionally independent of each other given the class (this is the "naive" assumption). It calculates the probability of each class based on the input features and selects the class with the highest posterior probability.
  
- **Assumptions:**
  - Assumes that all features are independent of each other (which is rarely true in real-world data).
  - Assumes that the features contribute equally and independently to the outcome.
  
- **Variants:**
  - **Gaussian Naive Bayes**: Assumes that the continuous features follow a Gaussian distribution.
  - **Multinomial Naive Bayes**: Suitable for discrete feature counts, commonly used in text classification.
  - **Bernoulli Naive Bayes**: Works for binary/boolean features, also used in text classification (presence or absence of words).

- **Output:** Returns the probability of each class given the input features and predicts the class with the highest probability.

- **Efficiency:** Fast and computationally efficient, making it suitable for real-time applications. It works well with large datasets and is particularly effective in text classification and spam filtering.

- **Use Cases:** 
  - Document classification (spam detection, sentiment analysis)
  - Medical diagnosis
  - Recommender systems
  - Multi-class classification problems where feature independence can be assumed.

- **Advantages:**
  - Works well even with small datasets.
  - Simple to implement and highly scalable for large datasets.
  
- **Limitations:**
  - The assumption of independence between features is rarely true, which can lead to suboptimal performance if the assumption is strongly violated.
  - Not ideal for highly correlated data.

- **Ref:** 
    - https://www.geeksforgeeks.org/naive-bayes-classifiers/
    - https://scikit-learn.org/1.5/modules/naive_bayes.html

---
## Gaussian Process Classification (GPC):
- **Type:** Probabilistic, non-parametric classifier.
- **Approach:** Gaussian Process Classification uses Gaussian processes to model the distribution of functions that map input features to output classes. It provides a flexible approach to classification by treating the output as a sample from a Gaussian process, allowing for uncertainty estimation in predictions. It uses the properties of Gaussian distributions to make predictions about the likelihood of each class given the input data.

- **Assumptions:**
  - Assumes that the data can be modeled as a Gaussian process.
  - The observations are noisy, and the model accounts for uncertainty in predictions.

- **Output:** Provides class probabilities for each input instance, allowing for uncertainty quantification in predictions. The model predicts the mean and variance of the output, helping to understand the confidence of the classification.

- **Efficiency:** More computationally intensive than many traditional classifiers, particularly for large datasets, as it involves matrix inversions and can scale cubically with the number of training samples. However, it is highly expressive and flexible.

- **Use Cases:** 
  - Suitable for applications requiring uncertainty estimates alongside predictions, such as:
    - Medical diagnosis
    - Robotics and control
    - Environmental modeling
    - Any domain where understanding the confidence of predictions is critical.

- **Advantages:**
  - Can model complex decision boundaries.
  - Provides a measure of uncertainty in predictions, useful for risk-sensitive applications.
  
- **Limitations:**
  - Computationally expensive, especially for large datasets.
  - Requires careful selection of the kernel function, which can significantly impact performance.

- **Ref:**
    - https://scikit-learn.org/1.5/modules/gaussian_process.html
    - https://www.youtube.com/watch?v=5Cqi-RAwAu8

---
## Support Vector Machine (SVM):
- **Type:** Supervised learning classifier.
- **Approach:** SVM aims to find the optimal hyperplane that separates classes in a high-dimensional space. It maximizes the margin between the nearest points of the classes (support vectors) while allowing for some misclassification through a soft margin.
  
- **Assumptions:**
  - Assumes that data can be linearly separable or can be transformed into a higher-dimensional space to achieve separation.

- **Output:** Predicts class labels based on the position of the input data relative to the hyperplane.

- **Efficiency:** Effective in high-dimensional spaces and works well with a clear margin of separation. However, it can be less efficient with large datasets.

- **Use Cases:** Suitable for classification tasks, including image recognition, text categorization, and bioinformatics.

- **Ref:**
    - https://www.geeksforgeeks.org/support-vector-machine-algorithm/

---

## Artificial Neural Network (ANN):
- **Type:** Supervised learning model inspired by biological neural networks.
- **Approach:** ANNs consist of interconnected layers of neurons, including an input layer, one or more hidden layers, and an output layer. They learn to map inputs to outputs by adjusting weights based on error feedback through backpropagation.
  
- **Assumptions:**
  - Assumes that the relationship between inputs and outputs can be approximated by a complex, nonlinear function.

- **Output:** Produces predictions for classification or regression tasks.

- **Efficiency:** Computationally intensive, especially for deep networks, but capable of learning complex patterns from large datasets.

- **Use Cases:** Widely used in image recognition, natural language processing, and any domain requiring nonlinear function approximation.

- **Ref:**
    - https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/

---

## AdaBoost (Adaptive Boosting):
- **Type:** Ensemble learning method.
- **Approach:** AdaBoost combines multiple weak classifiers to create a strong classifier. It assigns weights to each training instance and focuses on the misclassified instances, adjusting weights iteratively to improve overall classification accuracy.
  
- **Assumptions:**
  - Assumes that weak classifiers can be combined to form a strong classifier.

- **Output:** Provides a final classification based on the weighted majority vote of the weak classifiers.

- **Efficiency:** Generally fast and effective, but can be sensitive to noisy data and outliers.

- **Use Cases:** Commonly used in image classification, face detection, and any application where boosting can improve weak learner performance.

- **Ref:** 
    - https://www.almabetter.com/bytes/tutorials/data-science/adaboost-algorithm
    - https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
---

## XGBoost (Extreme Gradient Boosting):
- **Type:** Ensemble learning method based on gradient boosting.

- **Approach:** XGBoost improves upon traditional gradient boosting by optimizing computational efficiency and model performance. It builds an ensemble of decision trees in a sequential manner, where each new tree corrects errors made by the previous ones. It employs regularization techniques to prevent overfitting.

- **Assumptions:**
  - Assumes that boosting weak learners (typically decision trees) will lead to a strong predictive model.
  - It can handle both regression and classification problems.

- **Output:** Provides predictions based on the weighted sum of the outputs of all trees in the ensemble.

- **Efficiency:** Highly efficient and scalable, often faster than other boosting algorithms due to optimizations like parallel processing and tree pruning.

- **Use Cases:** Widely used in competitive machine learning, Kaggle competitions, and applications such as fraud detection, risk assessment, and customer churn prediction.

- **Ref:** 
    - https://xgboost.readthedocs.io/en/latest/
    - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html


---

## Logistic Regression:
- **Type:** Statistical method for binary classification.
- **Approach:** Logistic regression models the probability of a binary outcome based on one or more predictor variables using the logistic function. It predicts the likelihood that an instance belongs to a particular class by estimating coefficients for the input features.
  
- **Assumptions:**
  - Assumes a linear relationship between the log-odds of the outcome and the predictor variables.

- **Output:** Provides probabilities that can be thresholded to classify instances into binary classes.

- **Efficiency:** Computationally efficient and interpretable, making it suitable for smaller datasets or when interpretability is important.

- **Use Cases:** Used in medical diagnosis, marketing response prediction, and any scenario where binary outcomes are analyzed.

- **Ref:** 
    - https://www.geeksforgeeks.org/understanding-logistic-regression/

---

## Decision Tree:
- **Type:** Supervised learning model.
- **Approach:** Decision trees split the data into subsets based on feature values, forming a tree structure where each internal node represents a decision based on a feature, and each leaf node represents a class label. It recursively partitions the data to minimize impurity (e.g., Gini impurity or entropy).
  
- **Assumptions:**
  - Assumes that the data can be split effectively using decision rules based on feature values.

- **Output:** Predicts class labels based on the path traversed in the tree from the root to the leaf node.

- **Efficiency:** Fast and easy to interpret, but can be prone to overfitting if not properly regularized.

- **Use Cases:** Suitable for both classification and regression tasks, including customer segmentation, loan approval, and more.

- **Ref:** 
    - https://www.geeksforgeeks.org/decision-tree/

---

## Random Forest:
- **Type:** Ensemble learning method based on decision trees.
- **Approach:** Random Forest constructs multiple decision trees during training and outputs the mode of the classes (for classification) or mean prediction (for regression) of the individual trees. It introduces randomness by sampling data points and features to create diverse trees.
  
- **Assumptions:**
  - Assumes that combining multiple trees can lead to better predictive performance than any single tree.

- **Output:** Provides class predictions based on the aggregate results of the individual trees.

- **Efficiency:** Generally more robust and accurate than single decision trees, reduces overfitting, and handles large datasets well.

- **Use Cases:** Widely used in classification and regression tasks across various domains, including finance, healthcare, and marketing.

- **Ref:** 
    - https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/

---

## ResNet Model

ResNet, short for **Residual Network**, is a deep learning model architecture introduced by Kaiming He et al. in their 2015 paper titled “Deep Residual Learning for Image Recognition.” ResNet was designed to address the problem of vanishing gradients in very deep neural networks, which made training such models challenging.

ResNet architecture won ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2015 competition.

### Key Features of ResNet

1. **Residual Learning**:
   - The fundamental idea behind ResNet is the concept of residual learning, where the model learns to predict the residuals (differences) between the desired output and the input. This allows the network to focus on learning the changes rather than the entire mapping.
   - In practice, this is implemented using **skip connections** (or shortcut connections) that bypass one or more layers. The output of these layers is added to the output of the layer further along the network, helping to alleviate the vanishing gradient problem.

2. **Skip Connections**:
   - Skip connections allow the gradient to flow through the network more easily during backpropagation. They enable the model to learn an identity mapping, making it easier to train deep networks by allowing the model to "skip" layers if they are not needed.
   - A typical residual block in ResNet consists of two or more convolutional layers, with the input being added to the output of the block.

3. **Architecture**:
   - ResNet architectures come in various depths, such as ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152, where the numbers represent the total number of layers in the model.
   - Deeper models, like ResNet-50 and above, use **bottleneck layers**. These layers contain a 1x1 convolution that reduces dimensionality before passing data through 3x3 convolutions, followed by another 1x1 convolution to restore dimensionality.

4. **Batch Normalization**:
   - ResNet incorporates batch normalization layers to stabilize and accelerate the training process. This helps in reducing the sensitivity of the network to the initial weights and leads to faster convergence.

5. **Performance**:
   - ResNet achieved significant improvements in performance on various computer vision tasks, including image classification, object detection, and segmentation. It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2015, achieving a top-5 test accuracy of 96.43%.

### Advantages of ResNet

- **Easier Training of Deep Networks**: The architecture allows for training of much deeper networks (over 100 layers) without suffering from performance degradation.
- **Improved Generalization**: By mitigating the vanishing gradient problem, ResNet can generalize better on unseen data.
- **Flexibility**: The residual blocks can be easily integrated into other architectures, making them versatile for various tasks.

### Conclusion

ResNet has become one of the foundational models in deep learning, especially in the field of computer vision. Its innovative use of residual connections has influenced the design of many subsequent architectures, leading to the development of even deeper and more efficient models.

### References
- https://arxiv.org/abs/1512.03385
- https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/
- https://www.youtube.com/watch?v=woEs7UCaITo
- https://github.com/yacineMahdid/deep-learning-model-explained/blob/main/resnet/README.md
- https://www.youtube.com/watch?v=fvrIqFCUWV4 
