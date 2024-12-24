# RPG Character Classification Analysis
This project implements various machine learning models to analyze and predict character attributes in an RPG (Role-Playing Game) dataset. The analysis focuses on two main prediction tasks:
1. Character class classification
2. Final boss (FBoss) prediction

# Dataset 
The project uses the [RPG dataset](RPG.csv)
Features include:
* Armor
* Weapon
* Physical
* Magic
* Class (target variable)
* FBoss (target variable)

# Models Implemented
The following machine learning models were implemented and compared:
* Logistic Regression (for FBoss prediction only)
* Linear Discriminant Analysis (LDA)
* Quadratic Discriminant Analysis (QDA)
* Decision Trees (using both rpart and C5.0)
* Support Vector Machines (SVM)
* Neural Networks

# Dependencies
```
library(keras)
library(readr)
library(dplyr)
library(MASS)
library(rpart)
library(C50)
library(e1071)
```

# Neural Network Architecture
The project implements two neural network models:
### Class Classification Model
* Input layer: 4 features
* Hidden layers:
  * First layer: 8 units with ReLU activation
  * Second layer: 6 units with ReLU activation
* Output layer: 3 units with softmax activation

### FBoss Prediction Model
* Input layer: 4 features
* Hidden layers:
  * First layer: 8 units with ReLU activation
  * Second layer: 6 units with ReLU activation
* Output layer: 1 unit with sigmoid activation

### Data Preprocessing
* Missing values are removed using na.omit()
* Features are normalized using mean and standard deviation
* Class labels are encoded numerically
* Data is split into training (70%) and testing (30%) sets
* Neural network training uses a validation split of 20%

### Neural Network Training Parameters 
* Epochs: 25
* Batch size: 5
* Optimizer: Adam (learning rate: 1e-2)
* Loss functions:
  * Class prediction: categorical crossentropy
  * FBoss prediction: binary crossentropy

# Usage 
1. Ensure all required R packages are installed
2. Load the dataset from the provided URL
3. Run the analysis scripts to train and evaluate all models
4. Compare model performances using the generated accuracy table

# Notes
* The neural network implementation may be computationally intensive for larger datasets
* Model performance can vary based on the random seed used for data splitting
* Consider the trade-offs between model complexity and performance when choosing the final model

# Future Improvements
* Implement cross-validation for more robust model evaluation
* Add feature importance analysis
* Optimize neural network hyperparameters
* Include model prediction visualization
* Add confusion matrices for classification results
