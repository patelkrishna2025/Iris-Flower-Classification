# Iris-Flower-Classification
Iris Flower Classification is a beginner-friendly ML project that predicts flower species (Setosa, Versicolor, Virginica) based on sepal and petal measurements. Built using Scikit-learn, it includes data exploration, visualization, model training (Logistic Regression), and evaluation with high accuracy.

# Dataset Description
## Name: Iris Flower Dataset
Source: Built-in in libraries like Scikit-learn
Samples: 150 total
* 50 Iris setosa
* 50 Iris versicolor
* 50 Iris virginica

Features (Measurements):
* Sepal length (cm)
* Sepal width (cm)
* Petal length (cm)
* Petal width (cm)

Target: Species label (setosa, versicolor, virginica)
Type: Multiclass classification dataset

# Key Insights from Analysis

* Iris setosa is easily separable from other species by petal dimensions.
* Versicolor and Virginica have some overlapping feature ranges.
* Petal length and petal width are the most useful features for separating classes.
* Sepal features contribute less to class separation compared to petal features.

# Model Evaluation & Performance

Typical outcomes (models can vary):

## Model	Test Accuracy
Logistic Regression	~ 96%
K-Nearest Neighbors	~ 97%
Support Vector Machine	~ 98%
Decision Tree	~ 95%

High overall accuracy indicates strong separability in feature space.
Confusion matrix shows few misclassifications, usually between versicolor & virginica.

# Conclusion

* The Iris dataset is a classic classification problem with clean, well-separated classes.
* Petal measurements are key to distinguishing between species.
* Simple models like Logistic Regression and KNN perform very well.
* This project demonstrates fundamental ML concepts: dataset handling, feature importance, model training, evaluation, and prediction.
* It provides a solid introduction to supervised learning and classification.
