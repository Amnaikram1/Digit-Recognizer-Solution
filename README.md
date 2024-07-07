# MNIST Dataset Classification

## Overview
This repository explores the classification of handwritten digits (0-9) from the MNIST dataset using various machine learning algorithms and convolutional neural networks (CNNs).

### Dataset Overview
The MNIST dataset consists of 70,000 grayscale images: 60,000 for training and 10,000 for testing, each with dimensions of 28x28 pixels.

## Exploratory Data Analysis
The dataset is evenly distributed with approximately 5800 images per digit. Images are stored in .idx3-ubyte files, containing pixel intensities.

## Data Preprocessing
### Standard Scaling
All image pixels are scaled to have a mean of zero and a standard deviation of 1. This normalization facilitates faster convergence during training.

## Models Explored

### Machine Learning Classifiers
- **Random Forest Classifier**: Achieved an accuracy of 97.01%.
- **K-Nearest Neighbors Classifier**: Achieved an accuracy of 96.88%.
- **Support Vector Machine Classifier**: Achieved an accuracy of 96.6%.
- **Multi-Layer Perceptron (MLP) Classifier**: Achieved an accuracy of 97.36%.
- **Logistic Regression**: Achieved an accuracy of 92.55%.
- **XGBoost Classifier**: Achieved an accuracy of 97.95%.

### Convolutional Neural Network (CNN)

#### Architecture
- Designed with four convolutional layers, followed by max-pooling layers and fully connected layers.

#### Accuracy
- Achieved **99.23%** accuracy on the test set.

#### CNN Optimization Techniques
- **Dropout**: Used to prevent overfitting by randomly dropping neurons during training.
- **Batch Normalization**: Improved training speed and stability by normalizing activations.
- **Data Augmentation**: Increased dataset diversity and generalization by applying transformations like rotation, scaling, and flipping. Boosted accuracy to **99.9%** when applied to the CNN.

| Model                       | Accuracy (%) |
|-----------------------------|--------------|
| RandomForestClassifier      | 97.01        |
| KNeighborsClassifier        | 96.88        |
| SupportVectorMachine        | 96.6         |
| MultiLayerPerceptron        | 97.36        |
| LogisticRegression          | 92.55        |
| XgboostClassifier           | 97.95        |
| CNN                         | 99.23        |
| CNN with Data Augmentation  | 99.9         |

## Conclusion
This project demonstrates the effectiveness of CNNs and traditional machine learning models in digit recognition tasks using the MNIST dataset. Through careful preprocessing, model selection, and optimization techniques, significant accuracies were achieved across different methodologies.
