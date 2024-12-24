# Load the dataset from URL
df <- read.csv("https://raw.githubusercontent.com/YanaYerokhina/RPGCharacterClassificationAnalysis/main/RPG.csv", header = TRUE, sep = ",")

library(keras)
library(readr)
library(dplyr)
library(MASS)
library(rpart)
library(C50)
library(e1071)

df$Class <- as.factor(df$Class)
df$FBoss <- as.factor(df$FBoss)
# Logistic Regression for FBoss ----
# Define predictors and response variable
X_logit <- as.matrix(df[, c("Weapon", "Magic")])
Y_logit <- as.factor(df$FBoss)

# Fit logistic regression model
df$FBoss <- ifelse(df$FBoss == "True", 1, 0)
logit_model <- glm(FBoss ~ Weapon + Magic, data = df, family = binomial)
summary(logit_model)

# Calculate the accuracy of the logistic regression model
predicted_fboss <- predict(logit_model, type = "response")
predicted_fboss <- ifelse(predicted_fboss > 0.5, 1, 0)
accuracy_logit <- mean(predicted_fboss == df$FBoss)
accuracy_logit

# LDA ----
# LDA for Class prediction
lda_class <- lda(Class ~ Armor + Weapon + Physical + Magic, data = df)

# Calculate the accuracy of the LDA model for Class prediction
predicted_classes_lda <- predict(lda_class, newdata = df)$class
accuracy_lda_class <- mean(predicted_classes_lda == df$Class)

# LDA for FBoss prediction
lda_fboss <- lda(FBoss ~ Armor + Weapon + Physical + Magic, data = df)

# Calculate the accuracy of the LDA model for FBoss prediction
predicted_fboss_lda <- predict(lda_fboss, newdata = df)$class
accuracy_lda_fboss <- mean(predicted_fboss_lda == df$FBoss)

# QDA ----
# QDA for Class prediction
qda_class <- qda(Class ~ Armor + Weapon + Physical + Magic, data = df)

# Calculate the accuracy of the QDA model for Class prediction
predicted_classes_qda <- predict(qda_class, newdata = df)$class
accuracy_qda_class <- mean(predicted_classes_qda == df$Class)

# QDA for FBoss prediction
qda_fboss <- qda(FBoss ~ Armor + Weapon + Physical + Magic, data = df)

# Calculate the accuracy of the QDA model for FBoss prediction
predicted_fboss_qda <- predict(qda_fboss, newdata = df)$class
accuracy_qda_fboss <- mean(predicted_fboss_qda == df$FBoss)

# Decision Tree (rpart) ----
# Decision Tree (rpart) for Class prediction
tree_class <- rpart(Class ~ Armor + Weapon + Physical + Magic, data = df)

# Predict the class of characters
predicted_class_tree <- predict(tree_class, newdata = df, type = "class")
# Calculate the accuracy of the Decision Tree model for Class prediction
accuracy_tree_class <- mean(predicted_class_tree == df$Class)

# Decision Tree (rpart) for FBoss prediction
tree_fboss <- rpart(FBoss ~ Armor + Weapon + Physical + Magic, data = df)

# Predict the FBoss status
predicted_fboss_tree <- predict(tree_fboss, newdata = df, type = "class")
# Calculate the accuracy of the Decision Tree model for FBoss prediction
accuracy_tree_fboss <- mean(predicted_fboss_tree == df$FBoss)

# Decision Tree (C50) ----
# Decision Tree (C50) for Class prediction
tree_class_c50 <- C5.0(Class ~ Armor + Weapon + Physical + Magic, data = df)

# Predict the class of characters
predicted_class_tree_c50 <- predict(tree_class_c50, newdata = df, type = "class")
# Calculate the accuracy of the Decision Tree model for Class prediction
accuracy_tree_class_c50 <- mean(predicted_class_tree_c50 == df$Class)

# Decision Tree (C50) for FBoss prediction
tree_fboss_c50 <- C5.0(FBoss ~ Armor + Weapon + Physical + Magic, data = df)

# Predict the FBoss status
predicted_fboss_tree_c50 <- predict(tree_fboss_c50, newdata = df, type = "class")
# Calculate the accuracy of the Decision Tree model for FBoss prediction
accuracy_tree_fboss_c50 <- mean(predicted_fboss_tree_c50 == df$FBoss)

# SVM ----
# SVM for Class prediction
svm_class <- svm(Class ~ Armor + Weapon + Physical + Magic, data = df)

# Predict the class of characters
predicted_class_svm <- predict(svm_class, newdata = df)
# Calculate the accuracy of the SVM model for Class prediction
accuracy_svm_class <- mean(predicted_class_svm == df$Class)

# SVM for FBoss prediction
svm_fboss <- svm(FBoss ~ Armor + Weapon + Physical + Magic, data = df)

# Predict the FBoss status
predicted_fboss_svm <- predict(svm_fboss, newdata = df)
# Calculate the accuracy of the SVM model for FBoss prediction
accuracy_svm_fboss <- mean(predicted_fboss_svm == df$FBoss)

# NN ----
# Step 3: Preprocess the Data
# Remove any missing values
df <- na.omit(df)

# Encoding Class as a numeric factor
df$Class <- as.numeric(factor(df$Class)) - 1

# Splitting dataset into features and labels
features <- as.matrix(df[, c("Armor", "Weapon", "Physical", "Magic")])
labels_class <- to_categorical(df$Class)
labels_fboss <- as.numeric(df$FBoss)

# Normalizing features
mean <- apply(features, 2, mean)
std <- apply(features, 2, sd)
features <- scale(features, center = mean, scale = std)

# Splitting into training and testing sets
set.seed(123)
indices <- sample(1:nrow(features), size = 0.7 * nrow(features))
x_train <- features[indices,]
y_train_class <- labels_class[indices,]
y_train_fboss <- labels_fboss[indices,]
x_test <- features[-indices,]
y_test_class <- labels_class[-indices,]
y_test_fboss <- labels_fboss[-indices,]

# Step 4: Construct and Train Neural Network Models
# Model for Class classification
model_class <- keras_model_sequential(input_shape = 4) %>%
  layer_dense(units = 8, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dense(units = 6, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

model_class %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(learning_rate = 1e-2),
    metrics = 'accuracy'
  )

# Model for FBoss prediction
model_fboss <- keras_model_sequential(input_shape = 4) %>%
  layer_dense(units = 8, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dense(units = 6, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model_fboss %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adam(learning_rate = 1e-2),
    metrics = 'accuracy'
  )

# Train the models
history_class <- model_class %>%
  fit(
    x_train, y_train_class,
    epochs = 25,
    batch_size = 5,
    validation_split = 0.2
  )

history_fboss <- model_fboss %>%
  fit(
    x_train, y_train_fboss,
    epochs = 25,
    batch_size = 5,
    validation_split = 0.2
  )

# Step 5: Evaluate the Models
# Evaluate model for Class classification
scores_class <- model_class %>%
  evaluate(x_test, y_test_class)
accuracy_nn_class <- scores_class$acc

# Evaluate model for FBoss prediction
scores_fboss <- model_fboss %>%
  evaluate(x_test, y_test_fboss)
accuracy_nn_fboss <- scores_fboss$acc

# Table -----
# Create a table
accuracy_table <- data.frame(
  Model = c("Logistic Regression", "LDA", "QDA", "Decision Tree (rpart)", 
            "Decision Tree (C50)", "SVM", "Neural Network"),
  Accuracy_Class = c(0, accuracy_lda_class, accuracy_qda_class, 
                     accuracy_tree_class, accuracy_tree_class_c50, 
                     accuracy_svm_class, accuracy_nn_class),
  
  Accuracy_FBoss = c(accuracy_logit, accuracy_lda_fboss, accuracy_qda_fboss, 
                     accuracy_tree_fboss, accuracy_tree_fboss_c50, 
                     accuracy_svm_fboss, accuracy_nn_fboss)
)

# Print the table
print(accuracy_table)

# Opinion ----
# Provide opinion about using NN on this dataset
# Using neural networks on this dataset could offer several advantages. 
# Neural networks are capable of capturing complex patterns in the data and can 
# potentially outperform traditional linear models or tree-based models in terms
# of accuracy. However, they also come with some challenges such as requiring 
# careful tuning of hyperparameters, being computationally expensive to train, 
# and being more prone to overfitting, especially on smaller datasets. 
# Therefore, while neural networks may provide high accuracy, it's essential to 
# consider the trade-offs and the specific characteristics of the dataset before 
# choosing them as the modeling approach.












