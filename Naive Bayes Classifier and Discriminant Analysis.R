

# Naive Bayes classifier and Discriminant Analysis


# Task 1: Application of a classifier
library(naivebayes)
library(tibble)
library(tidyverse)
library(dplyr)
library(purrr)
library(caret)
library(klaR)
library(MASS)
library(pander)
library(gmodels)
library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(lattice)
library(WVPlots)
library(dslabs)
library(klaR)
library(gridExtra)
library(dplyr)
library(ggridges)
library(questionr)
library(MLmetrics)
library(lift)
library(corrplot)
library(resample)


# Question 1  Import the Mushroom Data into RStudio.
mushrooms <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
                        header = FALSE, sep = ",", dec = ".", na.strings = c("?")) # read the data in the csv file
# and load into R environment as a dataframe mushrooms

# Arguments of the function read.csv() are:
# a) file: name of the file with its remote access(path);
# b) header = FALSE: a logical (FALSE or TRUE) indicating if the file contains the names of the variables on its first line;
# The csv file does  not contains names of variables.
# c) sep = ",": the field separator used in the file is comma, (",");
# d) dec = ".": the character used for the decimal point;
# e) na.strings = c("?"): the symbol "?" given to missing data and converted as NA by default;

summary(mushrooms) # result summary of the dataframe
dim(mushrooms) # get the dimensions of the dataframe
sapply(mushrooms,class) # user-friendly version and wrapper of lapply function but returns a vector of column classes

# Rename the column names assigned by default as V1, V2...V23 into names shown below:
colnames(mushrooms) <- c("edibility", "cap_shape", "cap_surface",
                         "cap_color", "bruises", "odor",
                         "gill_attachement", "gill_spacing", "gill_size",
                         "gill_color", "stalk_shape", "stalk_root",
                         "stalk_surface_above_ring", "stalk_surface_below_ring", "stalk_color_above_ring",
                         "stalk_color_below_ring", "veil_type", "veil_color",
                         "ring_number", "ring_type", "spore_print_color",
                         "population", "habitat")

# Recode each variable as a factor and display all factor levels for all variables:
mushrooms %>% mutate_all(as.factor) %>% map(levels)

# Define and rename the levels of all categorical variables:
levels(mushrooms$edibility) <- c("edible", "poisonous")
levels(mushrooms$cap_shape) <- c("bell", "conical", "flat", "knobbed", "sunken", "convex")
levels(mushrooms$cap_surface) <- c("fibrous", "grooves", "smooth", "scaly")
levels(mushrooms$cap_color) <- c("buff", "cinnamon", "red", "gray", "brown", "pink",
                                 "green", "purple", "white", "yellow")
levels(mushrooms$bruises) <- c("no", "yes")
levels(mushrooms$odor) <- c("almond", "creosote", "foul", "anise", "musty", "none", "pungent", "spicy", "fishy")
levels(mushrooms$gill_attachement) <- c("attached", "free")
levels(mushrooms$gill_spacing) <- c("close", "crowded")
levels(mushrooms$gill_size) <- c("broad", "narrow")
levels(mushrooms$gill_color) <- c("buff", "red", "gray", "chocolate", "black", "brown", "orange",
                                  "pink", "green", "purple", "white", "yellow")
levels(mushrooms$stalk_shape) <- c("enlarging", "tapering")
levels(mushrooms$stalk_root) <- c("bulbous", "club", "equal", "rooted")
levels(mushrooms$stalk_surface_above_ring) <- c("fibrous", "silky", "smooth", "scaly")
levels(mushrooms$stalk_surface_below_ring) <- c("fibrous", "silky", "smooth", "scaly")
levels(mushrooms$stalk_color_above_ring) <- c("buff", "cinnamon", "red", "gray", "brown", "orange", "pink",
                                              "white", "yellow")
levels(mushrooms$stalk_color_below_ring) <- c("buff", "cinnamon", "red", "gray", "brown", "orange", "pink",
                                              "white", "yellow")
levels(mushrooms$veil_type) <- ("partial")
levels(mushrooms$veil_color) <- c("brown", "orange", "white", "yellow")
levels(mushrooms$ring_number) <- c("none", "one", "two")
levels(mushrooms$ring_type) <- c("evanescent", "flaring", "large", "none", "pendant")
levels(mushrooms$spore_print_color) <- c("buff", "chocolate", "black", "brown", "orange",
                                         "green", "purple", "white", "yellow")
levels(mushrooms$population) <- c("abundant", "clustered", "numerous", "scattered", "several", "solitary")
levels(mushrooms$habitat) <- c("woods", "grasses", "leaves", "meadows", "paths", "urban", "waste")

mushrooms %>% map(levels) %>% map(length) %>% as.data.frame() # determine the number (length) of factor levels for each variable

# We can see that variable veil_type has only one factor level: "partial". However,it will bring issues during modelling stage.
# R will throw an error for any categorical variable that has only one level and low counts. However, this variable will be kept
#because it has 8124 counts and as per assessment requirements to use all 23 variables.

map_dbl(mushrooms, function(.x) {sum(is.na(.x))}) # determine the number of missing values in each variable. Stalk_root variable has 2480 missing values

# Question 2 Randomly split the dataset into a training subset and a test subset containing 80% and 20% of the data.
set.seed(0) # random seed
no_observations <- dim(mushrooms)[1] # number of observations
test_index <- sample(no_observations, size = as.integer(no_observations*0.2), replace = FALSE)  # 20% data for testing
train_index <- -test_index # remaining 80% data observations for training
train_set = mushrooms[train_index,] # assigning 80% train data into a variable train_set
test_set = mushrooms[-train_index,] # assigning 20% train data into a variable test_set
View(test_set)
# Comparison of class label distribution among mushroom datasets (Original dataset (mushrooms), Training dataset and Testing dataset):
mush_original <- round(table(mushrooms$edibility) %>% prop.table() * 100, 1) # class label distribution as frequency and proportions of original dataset (mushrooms) dataset
mush_train <- round(table(train_set$edibility) %>% prop.table() * 100, 1) # class label distribution as frequency and proportions of training dataset
mush_test <- round(table(test_set$edibility) %>% prop.table() * 100, 1) # class label distribution as frequency and proportions of testing dataset
mush_df <- as.matrix(cbind(mush_original, mush_train, mush_test)) # create the dataframe
colnames(mush_df) <- c("Original", "Training set", "Test set") # assign name to columns
pander(mush_df, style = "rmarkdown", caption = paste0("Comparison of Class Label Distribution among Mushroom Datasets")) # print an R object in Pandoc's markdown

T# Question 4. Implement the proposed classifier from step 3usingthe training data subset from step 1
set.seed(0) # random seed
NaiveBayesModel <- naive_bayes(edibility ~. , data = train_set, laplace = 1,
                               na.action = na.pass, prior = NULL,
                               usekernel = FALSE, usepoisson = FALSE) # implementing Naive_Bayes classifier using training  data subset



# via general formula interface of naive_bayes function in which predictors are assumed to be independent.
NaiveBayesModel
# Arguments of the function are:
# a) x can be matrix or dataframe with categorical (character/factor/logical) or metric (numeric) predictors;
# b) y is a class vector (character/factor/logical);
# c) formula an object of class "formula"of the form: class ~ predictors (class has to be a factor/character/logical);
# d) data matrix or dataframe with categorical (character/factor/logical) or metric (numeric) predictors;
# e) prior vector with prior probabilities of the classes;
# f) laplace value used for Laplace smoothing. Defaults is 0;
# g) usekernel logical; if TRUE, density is used to estimate the class conditional densities of metric predictors;
# h) na.action is a function which indicates what should happen when the data contain NAs. By default
# (na.pass), missing values are not removed from the data and are then omited while constructing
# tables.



# Question 5  Display the summary of the fitted model implemented in Question 4.

summary(NaiveBayesModel)
# Calculate model performance metrics on test subset of data
nb_metrics = function(model, data){
   # calculate accuracy
   predictClass <- predict(model, newdata = data[, -1],  type = "class")
   (cm_nb = table(predictClass, test_set$edibility)) # create the confusion matrix
   accuracy = round((cm_nb[2,2] + cm_nb[1,1]) / sum(cm_nb[2,2],cm_nb[2,1],cm_nb[1,1], cm_nb[1,2])*100, 2)
   class_error_rate = round(sum(cm_nb[2,1] + cm_nb[1,2]) / sum(cm_nb[2,2] + cm_nb[2,1] + cm_nb[1,1] + cm_nb[1,2])*100,2) # calculate class_error_rate
   precision = round(cm_nb[2,2] / sum(cm_nb[2,2] + cm_nb[2,1]),2) # calculate precision
   sensitivity = round(cm_nb[2,2] / sum(cm_nb[2,2] + cm_nb[1,2]),2) # calculate sensitivity
   specificity = round(cm_nb[1,1] / sum(cm_nb[1,1] + cm_nb[2,1]),2)# calculate specificity
   recall = round((cm_nb[2,2])/sum(cm_nb[2,2] + cm_nb[1,2]),2) # calculate recall
   f1_score = round((2 * precision * sensitivity) / (precision + sensitivity),2) # calculate F1 score
   metrics = c(accuracy, class_error_rate, precision, sensitivity, specificity, recall, f1_score)
   names(metrics) = c("Accuracy", "Class_Error_Rate", "Precision", "Sensitivity", "Specificity", "Recall", "F1 score")
   return(metrics)
}

# Calculate model performance metrics on test subset set
nb_metrics(NaiveBayesModel, test_set)
# Calculate classification rate on training and testing subsets of data:
mush_nb_trn_pred <- predict(NaiveBayesModel, train_set,  type = "class") # predict probability for training set
mush_nb_tst_pred = predict(NaiveBayesModel, test_set[, -1], type = "class") # predict probability for testing set

calc_class_err = function(actual, predicted) {
   mean(actual != predicted)
}
nb_train_err <- calc_class_err(predicted = mush_nb_trn_pred, actual = train_set$edibility) # apply function on training set
nb_test_err <- calc_class_err(predicted = mush_nb_tst_pred, actual = test_set$edibility) # apply function on testing set

training_set <- round(c(nb_train_err)*100, 2) # making percentages for train error
testing_set <- round(c(nb_test_err)*100, 2) # making percentages for test error
method <- c("NaiveBayes") # assigning a vector method
mush_df <- as.matrix(cbind(method, training_set,testing_set)) # create the dataframe
colnames(mush_df) <- c("Method", "Training set", "Test set") # assign name to columns
pander(mush_df, style = "rmarkdown", caption = paste0("Comparison of Classification Error Rates among Mushroom Datasets")) # print an R
# object in Pandoc's markdown

# Q7 Load the HBblood dataset into R environment
HBblood <- read.csv("path", header = TRUE, sep = ",", dec = ".", na.strings = c("?"))# read the data
# and load into R environment as a dataframe HBblood

# Summarize dataset:
str(HBblood) # compact structure display of dataset
sapply(HBblood,class) # list types of each variable
summary(HBblood) # summarize the variable distribution
levels(HBblood$Ethno) # factor levels of the variable Ethno
HBblood$Ethno = factor(HBblood$Ethno, ordered = FALSE, levels = c("A", "B","C")) # coercing variable Ethno to a factor
map_dbl(HBblood, function(.x) {sum(is.na(.x))}) # determine the number of missing values in each variable.
#There are no missing values in the dataset

# Randomly split the dataset into a training subset and a test subset containing 80% and 20% of the data:
HBblood_idx <- createDataPartition(HBblood$Ethno, p = 0.80, list = FALSE) # create a list of 80% of the
# rows in the original dataset we can use for training
HBblood_test <- HBblood[-HBblood_idx, ] # select 20% of the data for testing
HBblood_train <- HBblood[HBblood_idx, ] # use the remaining 80% of data to train the models

Ethno <- HBblood_train$Ethno # assigning variable to a vector
HbA1c <- HBblood_train$HbA1c # assigning variable to a vector
SBP <- HBblood_train$SBP # assigning variable to a vector

# Density plots for the predictors in the training subset:
 HBblood_train %>% mutate(y = factor(Ethno)) %>%
   ggplot(aes(HbA1c, SBP, fill = Ethno, color = Ethno)) +
   geom_point(show.legend = FALSE) +
   stat_ellipse(type = "norm", lwd = 1.5)

 # Exploratory data analysis: multivariate plots
 # Density plot for each variable by Ethno variable class value
 transparentTheme(trans = 0.9)
 caret::featurePlot(x = HBblood_train[, c("HbA1c", "SBP")],
                    y = HBblood_train$Ethno,
                    plot = "density",
                    scales = list(x = list(relation = "free"),
                                  y = list(relation = "free")),
                    adjust = 1.5,
                    pch = "|",
                    layout = c(2, 1),
                    auto.key = list(columns = 3))
 # Pairplot of the predictors in the training subset
 transparentTheme(trans = .9)
 caret::featurePlot(x = HBblood_train[, c("HbA1c", "SBP")],
                    y = HBblood_train$Ethno,
                    plot = "pairs",
                    auto.key = list(columns = 3))


 # Box plot for each variable
 caret::featurePlot(x = HBblood_train[, c("HbA1c", "SBP")],
                    y = HBblood_train$Ethno,
                    plot = "box",
                    scales = list(y = list(relation = "free"),
                                  x = list(rot = 90)),
                    layout = c(2, 1))

 ggplot(data = HBblood_train) + aes(x = Ethno, y = HbA1c) +
    aes(color = Ethno) + geom_boxplot(outlier.size = 2,
                                      outlier.color = "red", outlier.shape = 3) +
    geom_jitter(width = 0.1, alpha = 0.05, color = "blue")
 # boxplot for Ethno and SBP
 ggplot(data = HBblood_train) + aes(x = Ethno, y = SBP) +
    aes(color = Ethno) + geom_boxplot(outlier.size = 2,
                                      outlier.color = "red", outlier.shape = 3) +
    geom_jitter(width = 0.1, alpha = 0.05, color = "blue")
 # Pairplot of the classess of the response variable Ethno
 library(WVPlots)
 PairPlot(HBblood_train,
          colnames(HBblood_train)[2:3],
          "HBblood Data - 3 Ethnical Groups",
          group_var = "Ethno",
          alpha = 1)

 # Implementing the models for Quadratic Discriminant Analysis (QDA), Linear Discrmininat Analysis (LDA) and naive Bayes model:
 # set up tuning grid
 set.seed(0) # random seed

 control <- trainControl(method = "cv", number = 10)
 metric <- "Accuracy"

 fit.lda <- train(Ethno~., data = HBblood_train, method = "lda", metric = metric, trControl = control)
 fit.qda <- train(Ethno~., data = HBblood_train, method = "qda", metric = metric, trControl = control)
 nb_model = naive_bayes(Ethno ~., data = HBblood_train)

 # summarize accuracy of models
 results <- resamples(list(lda = fit.lda, qda = fit.qda, nb = fit.nb))
 summary(results)

 bwplot(results) # box_plot comparison of results

 # Measures of predicted classes for naive Bayes model:
 predictClass_a <- predict(nb_model, newdata = HBblood_test[, -1], type = "class") # make predictions on the test subset of data
 predictClass_a

 cm.nb <- confusionMatrix(HBblood_test$Ethno, predict(nb_model, HBblood_test[, -1])) # create the confusion matrix
 cm.nb

 # Measures of predicted classes for LDA model:
 predictClass_b <- predict(fit.lda, newdata = factor(HBblood_test[, -1]))$class # make predictions on the test subset of data
 predictClass_b
 cm.lda <- confusionMatrix(HBblood_test$Ethno, predict(fit.lda, HBblood_test[, -1])) # create the confusion matrix
 cm.lda

 # Measures of predicted classes for QDA model
 predictClass_c <- predict(fit.qda, HBblood_test[, -1]) # make predictions on the test subset of data
 predictClass_c
 cm.qda <- confusionMatrix(HBblood_test$Ethno, predict(fit.qda, HBblood_test[, -1])) # create the confusion matrix
 cm.qda

 # Summarize the results of model performance on training subset of data of the three models:
 model_list <- resamples(list(naive_bayes = nb_model, lda = fit.lda, qda = fit.qda)) # create a list of model names
  # collect, analyze and visualize a set of results of model performance
 summary(model_list) # summarise the results
 bwplot(model_list) # box_plot comparison of results

 # Integrated display of summary statistics for both models:
 cm_list <- list(naiveBayesModel = cm.nb, LDA = cm.lda, QDA = cm.qda)
 results <- map_df(cm_list, function(x)x$byClass) %>% as_tibble() %>%
    mutate(stat = names(cm.nb$"byClass"))

 # Estimates on conditaional probablities:
 # Decsion boundries plots with library(klaR)
 HBblood_train[HBblood_train$SBP > 750] = NA # outliers were removed in this image
 HBblood_train_omit = na.omit(HBblood_train)

 par(mfrow = c (1,3)) # combining plots
 partimat(Ethno ~ HbA1c + SBP, data = HBblood_train_omit, method = "lda") # partition plot for LDA
 partimat(Ethno ~ HbA1c + SBP, data = HBblood_train_omit, method = "qda") # partition plot for QDA
 partimat(Ethno ~ HbA1c + SBP, data = HBblood_train_omit, method = "naiveBayes") # partition plot for QDA

 # Assessing ROC curve, calibration curve precision recall gain curve for the threemodels with library(MLeval)
 ctrl <- trainControl(method = "cv", summaryFunction = multiClassSummary, classProbs = T,
                      savePredictions = T) # defining parameters of train control function with multiClassSummary for naive Bayes model
 fit1 <- train(Ethno ~ .,data = HBblood_train,method = "nb",trControl = ctrl) # fitting the model
 ctrl <- trainControl(method = "cv", summaryFunction = multiClassSummary, classProbs = T,
                      savePredictions = T)
 fit2 <- train(Ethno ~ .,data = HBblood_train, method = "lda",trControl = ctrl)
 ctrl <- trainControl(method = "cv", summaryFunction = multiClassSummary, classProbs = T,
                      savePredictions = T)
 fit3 <- train(Ethno ~ .,data = HBblood_train,method = "qda",trControl = ctrl)

 # Collect the train function results of the three models with evalm function to evaluate the predictions:
 res <- evalm(list(fit1, fit2, fit3), gnames = c("nb","lda", "qda"))
 # Draw the ROC curve
 res$roc

 # Draw the Calibration Curve
 res$cc

 # Draw Precision Recall Gain Curve
 res$prg
 
# Q8 Analysis ofHeart dataset
 heartData = read.table('heart.txt', col.names = c('Country', 'Sex', "SBP", "SBP_LCI", "SBP_UCI")) # read the data
 # and load into R environment as a dataframe heartData

 summary(heartData) # summarize the variable distribution There are no missing values
 str(heartData) # data structure compact display
 heartData$Sex = factor(heartData$Sex, ordered = FALSE, levels = c("Men", "Women")) # coerce variable Sex to a factor
 heartData[,c(1,4,5)] <- NULL # remove the variables(Country, SBP_LCI and SBP_UCI)

 # Randomly split the dataset into a training subset and a test subset containing 80% and 20% of the data.
 set.seed(0) # random seed
 heartsample <- caret::createDataPartition(y = heartData$Sex, times = 1, p = 0.8, list = FALSE) # create a list of 80% of the
 # rows in the original dataset we can use for training
 train_heart <- heartData[heartsample, ] # use the remaining 80% of data to train the models
 test_heart <- heartData[-heartsample, ] # select 20% of the data for testing
 # Exploartory analysis of the train data:
 # Density plots for the predictors with Ridgeline graph, also called a joyplot. It displays the distribution
 # of a quantitative variable for Sex categorical variable. This allows us to map the probabilities in color:
 Sex <- heartData$Sex # assigning variable as a vector
 SBP <- heartData$SBP # assigning variable as a vector
 ggplot(train_heart,
        aes(x = SBP,
            y = Sex,
            fill = 0.5 - abs(0.5 - stat(ecdf)))) +
    stat_density_ridges(geom = "density_ridges_gradient", calc_ecdf = TRUE) +
    scale_fill_viridis_c(name = "Tail probability", direction = -1)

 # Comparison of class label distribution among datasets (original dataset (heartData), training dataset and testing dataset):
 freq(train_heart$Sex, cum = FALSE, sort = "dec", total = TRUE) # class label distribution as frequency and proportions of training dataset
 freq(heartData$Sex, cum = FALSE, sort = "dec", total = TRUE) # class label distribution as frequency and proportions of original dataset (heart)
 freq(test_heart$Sex, cum = FALSE, sort = "dec", total = TRUE) # class label distribution as frequency and proportions of testing dataset

 # Implementing the models: LDA naive Bayes:
 # set up tuning grid
 set.seed(0) # random seed
 control = trainControl(method = "cv", number = 5, repeats = 3, classProbs = TRUE) # control the computational nuances of the train function
 searchGrid <- expand.grid(fL = c(0:5), usekernel = c(TRUE, FALSE), adjust = seq(0, 5, by = 1)) # define the grid parameters for the train function
 naiveBayesModel <- train(Sex ~., data = train_heart, method = "nb", trControl = control,
                          na.action = "na.pass", metric = "Accuracy", tuneGrid = expand.grid(fL = 1, usekernel = FALSE, adjust = 1)) # fit NaiveBayes model with caret package

 fit.lda <- train(Sex ~., data = train_heart, method = "lda", metric = "Accuracy", trControl = control) # implement the LDA model

 # Make prediction for the new data on test subsets of data:
 predictClass_1 <- predict(naiveBayesModel, test_heart) # test subset of naïve Bayes model
 predictClass_2 <- predict(fit.lda, test_heart) # make predictions on the test subset of LDA model

 # Displaying model metrics of performance with the help of confusion matrix
 cm_nb <- confusionMatrix(reference = test_heart$Sex, data = predictClass_1,
                          mode = "everything", positive = "Men")
 cm_nb

 cm_lda <- confusionMatrix(reference = test_heart$Sex, data = predictClass_2,
                           mode = "everything", positive = "Men")
 cm_lda # display the performace metics for LDA model

 # Present the summary of Accuracy and Kappa for both models:
 results <- resamples(list(lda = fit.lda, nb = naiveBayesModel)) # connect the results
 summary(results) # summarise the results

 # Measures of predicted classes for both models:
 models <- list(nb = naiveBayesModel, lda = fit.lda) # list of models
 testPred <- predict(models, newdata = test_heart) # predictions on test data subset of two models
 lapply(testPred, function(x)x) # returns a list of the same length as X, each element of which is the result of applying FUN to the corresponding element of X
 # Integrated display of summary statistics for both models:
 cm_list <- list(naiveBayesModel = cm_nb, LDA = cm_lda)
 results <- map_df(cm_list, function(x)x$byClass) %>% as_tibble() %>%
    mutate(stat = names(cm_nb$"byClass"))

 #Q9. Calculate classification error rate of the models:
 heart_nb_trn_pred <- predict(naiveBayesModel, train_heart) # predict probability
 # for training set naiveBayes
 heart_nb_tst_pred = predict(naiveBayesModel, test_heart) # predict
 # probability for testing set naiveBayes

 heart_lda_trn_pred <- predict(fit.lda, train_heart) # predict probability
 # for training set LDA
 heart_lda_tst_pred <- predict(fit.lda, test_heart) # predict probability
 calc_class_err = function(actual, predicted) {
    mean(actual != predicted)
 }
 nb_train_err <- calc_class_err(predicted = heart_nb_trn_pred, actual = train_heart$Sex)
 nb_test_err <- calc_class_err(predicted = heart_nb_tst_pred, actual = test_heart$Sex)
 # calculate error for training and test subsets for naïve Bayes model

 lda_train_err <- calc_class_err(predicted = heart_lda_trn_pred, actual = train_heart$Sex)
 lda_test_err <- calc_class_err(predicted = heart_lda_tst_pred, actual = test_heart$Sex)
 # calculate error for training and test subsets for LDA model

 training_set <- c(nb_train_err, lda_train_err)
 testing_set <- c(nb_test_err, lda_test_err)
 method <- c("NaiveBayes", "LDA")
 heart_df <- as.data.frame(cbind(method, training_set,testing_set)) # create the dataframe
 heart_df

 colnames(heart_df) <- c("Method", "Training set", "Test set") # assign name to columns
 pander(heart_df, style = "rmarkdown", caption = paste0("Comparison of Classification Error Rates among heartData Datasets")) # print an R object in Pandoc's markdown

 # Q10 Discuss the findings from Q8 and Q9:
 # Create Lift Chart: train both models once again because we need values for parameter “ROC”:
 set.seed(0) # random seed
 ctrl <- trainControl(method = "cv", classProbs = TRUE,
                      summaryFunction = twoClassSummary)

 set.seed(0)
 lda_lift <- train(Sex ~ ., data = train_heart,
                   method = "lda", metric = "ROC",
                   tuneLength = 20,
                   trControl = ctrl)

 set.seed(0)
 nb_lift <- train(Sex ~ ., data = train_heart,
                  method = "nb", metric = "ROC",
                  trControl = ctrl)

 ## Generate the test set results for both models
 lift_results <- data.frame(Sex = test_heart$Sex)
 lift_results$LDA <- predict(lda_lift, test_heart, type = "prob")[,"Men"]
 lift_results$naiveBayesModel <- predict(nb_lift, test_heart, type = "prob")[,"Men"]
 head(lift_results)

 dim(lift_results) # dimensions of the lift data

 trellis.par.set(caretTheme()) # function for creating lift object from package caret
 lift_obj <- lift(Sex ~ LDA + naiveBayesModel, data = lift_results)
 plot(lift_obj, values = 80, auto.key = list(columns = 2,
                                             lines = TRUE,
                                             points = FALSE))
 ggplot(lift_obj, values = 80) # the same function with ggplot2 library

 # Train the models once again because we need parameters from twoClassSummary function and
 #library Mleval to plot ROC curve and precision recall gain curves:

 ctrl <- trainControl(method = "cv", summaryFunction = twoClassSummary, classProbs = T,
                      savePredictions = T)
 fit1 <- train(Sex ~ .,data = train_heart, method = "nb",metric = "ROC", trControl = ctrl)
 ctrl <- trainControl(method ="cv", summaryFunction = twoClassSummary, classProbs=T,
                      savePredictions = T)
 fit2 <- train(Sex ~ .,data = train_heart, method ="lda", metric = "ROC", trControl = ctrl)

 res <- evalm(list(fit1,fit2),gnames=c("nb","lda"))

 ## Generate ROC curve:

 res$roc

 ## Generate calibration curve:

 res$cc

 ## Generate precision recall gain curve:

 res$prg

 # Pairplot of the classess of the response variable Sex
 library(WVPlots)
 PairPlot(train_heart,
          colnames(train_heart)[2:2],
          "Heart Data - Variation of Sistolic Pressure among Man and Women",
          group_var = "Sex",
          alpha = 1)
