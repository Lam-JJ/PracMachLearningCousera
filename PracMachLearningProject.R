## Practical Machine Learning Project
# Goal : Predict the manner they did the exercise("classe" variable)
# Predictor = every other variable
# Build model -> Cross Validation -> expected out-of-sample error 
# Will need to use prediction model to predict 20 different test cases

setwd("C:/Users/User/Desktop/Studies/Data Science/part 10/PracMachLearningCousera")

# Load library
library(caret)
library(dplyr)
library(corrplot)
library(rpart)
library(rattle)
library(glmnet)
library(Hmisc)

# Load data
trainCSV <- read.csv("pml-training.csv",header=T)
testCSV <- read.csv("pml-testing.csv",header=T)

# Filter acceleration datas and classe
trainSet <- trainCSV %>% select(contains("accel"),classe) %>% select(-contains("var"))
testing <- testCSV %>% select(contains("accel")) %>% select(-contains("var")) 

trainSet$classe <- as.factor(trainSet$classe)

# Create seperation
set.seed(123)
inTrain = createDataPartition(trainSet$classe, p = 3/4)[[1]]
training <- trainSet[inTrain,]
validation <- trainSet[-inTrain,]

# Exploratory Analysis
featurePlot(x=training[,c("total_accel_belt","total_accel_arm",
                          "total_accel_dumbbell","total_accel_forearm")],
            y=training$classe,
            plot="density")

res <- cor(subset(training,select=-c(classe)))
corrplot(res, type = "upper")

# Imputing doesnt work for variable "var" as 97% of the time is NA

# Cross Validation [ k fold with k = 10 ]
control <- trainControl(method = "cv",number = 5)

## Training the model
# rPart
modelPart <- train(classe~., data = training, method = "rpart", trControl = control,
                   tuneLength = 10)
predPart <- predict(modelPart,validation)
accPart <- sum(predPart==validation$classe)/length(predPart)

classFit <- rpart(classe~.,data = training,method="class")
predClass <- predict(classFit,validation,type="class")
accClass <- sum(predClass==validation$classe)/length(predClass)
fancyRpartPlot(classFit)

# Boosting
modelGBM <- train(classe~., data = training, method = "gbm", 
                  trControl = control, verbose = FALSE, preProcess=c("center","scale"))
predGBM <- predict(modelGBM,validation)
accGBM <- sum(predGBM==validation$classe)/length(predGBM)

# K nearest neightbour
modelKNN <- train(classe~., data = training, method = "knn", trControl = control, 
                    preProcess = c("center","scale"),tuneLength=20)
predKNN <- predict(modelKNN,validation)
accKNN <- sum(predKNN==validation$classe)/length(predKNN)

# Random Forest 
modelRF <- train(classe~., data = training, method = "rf", trControl = control)
predRF <- predict(modelRF,validation)
accRF <- sum(predRF==validation$classe)/length(predRF)

# Combined model
predDF <- data.frame(predGBM,predKNN,predRF,classe=validation$classe)
modelComb <- train(classe~.,method="rf",data=predDF)
predComb <- predict(modelComb,predDF)
accComb <- sum(predComb==validation$classe)/length(predComb)

# Test
testGBM <- predict(modelGBM,testing)
testKNN <- predict(modelKNN,testing)
testRF <- predict(modelRF,testing)
testDF <- data.frame(predGBM=testGBM,predKNN=testKNN,predRF=testRF)
testComb <- predict(modelComb,testDF)





# SVM 
modelSVM <- train(classe~.-accel_belt_y-accel_belt_z-accel_dumbbell_x-accel_dumbbell_y-accel_dumbbell_z,
                  data = training, method = "svmLinear",
                  preProcess = c("center","scale"))
predSVM <- predict(modelSVM,validation)
accSVM <- sum(predSVM==validation$classe)/length(predSVM)

# Navies-Bayes : Useful for binary/catogorical variables
modelNB <- train(classe~., data = training, method = "nb", trControl = control)
predNB <- predict(modelNB,validation)
accNB <- sum(predNB==validation$classe)/length(predNB)



# Linear Discriminant Analysis [ Same as Navier Bayes ]
modelLDA <- train(classe~., data = training, method = "lda", trControl = control)
predLDA <- predict(modelLDA,validation)
accLDA <- sum(predLDA==validation$classe)/length(predLDA)

# General Additive Model [ slow ]
modelGAM <- train(classe~., data =  training, method = "gam", trControl = control)

# Ridge regression
modelRR <- train(classe~., data = training, method = "glmnet", trControl = control)
predRR <- predict(modelRR,validation)
accRR <- sum(predRR==validation$classe)/length(predRR)
