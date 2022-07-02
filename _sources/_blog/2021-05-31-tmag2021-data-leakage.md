<!-- ---
key: post20210531
title: "Data Leakage"
date: 2021-05-31
categories: ["Math and Stats"]
tags: ["Machine learning", "Prediction"]
published: true
--- -->

# Data Leakage

  *(This post is from a talk on data leakage at [the 2021 Technology and Measurement Around the Globe (TMAG) virtual symposium](https://www.purdue.edu/hhs/psy/tmag/). Code and data can be found [here](https://github.com/tangchen/data-leakage-tmag2021))*


## What is Data leakage

Data leakage happens when information about the outcome variable “leaks”
into the model training process, and such information is not available
when the trained model is later used to make predictions on new data
(Ambroise & McLachlan, 2002).

For example, data analysts may want use images (shown below) to predict
whether a mole/melanoma is malignant or benign. Notice that one issue in
these images is that a ruler is shown in the malignant image. If
majority of the malignant images has a ruler, then the presence of a
ruler will leak information about the outcome (malignant or benign) into
the model training process: a ruler was shown because it was already
diagnosed as malignant, and doctors probably wanted to track the size of
the melanoma.

![](https://raw.githubusercontent.com/tangchen/data-leakage-tmag2021/main/melanoma.jpg)

When a model trained on such image data is used in real applications (,
where there will be no rulers, because diagnosis has not been done), the
model will become inaccurate. Therefore, data leakage may lead to bad
model performance on new data and hinder model generalizability. In this
example, one solution is to crop the images and only include information
about the melanoma. Similarly, there are possibilities for data leakage
in organizational research. Let’s look at some examples and discuss how
to avoid this issue.

## Example Dataset

We will use an example dataset to demonstrate some common data leakage
issues. Note that the dataset is simulated, it’s not real data.

Assuming that we are interested in predicting employee turnover, and we
have decided to train a model based on this dataset. Turnover is a
binary variable (Y = yes, the employee has left; N = no, the employee
hasn’t left). The predictors include the year the employee was hired
(“year\_hired”), age, gender, education, tenure in the organization
(“tenure”), job satisfaction (“jobsat”), performance (“perf”), and
salary. There are 200 observations in this dataset, the first ten rows
are shown below:

```r
library(caret)

SEED <- 8424 # this random seed was for replicating the results

dt <- read.csv("toydata.csv")
head(dt, 10)
```
    ##    year_hired age gender edu tenure jobsat perf salary turnover
    ## 1        2012  36      2   3      4   3.86 1.93     NA        Y
    ## 2        2017  NA      2   3      4   5.00   NA   5000        N
    ## 3        2016  40      2   4      5   2.07   NA   6800        N
    ## 4        2015  NA      1   3      6   3.34 2.84  10000        N
    ## 5        2017  22      1   3      4     NA 5.00   5000        N
    ## 6        2017  24      2   3      4   3.81 4.87   5000        N
    ## 7        2015  NA      2   3      6   1.90 4.39  10000        N
    ## 8        2002  57      2   3      3     NA 2.21     NA        Y
    ## 9        2012  33      1   3      4   1.00   NA     NA        Y
    ## 10       2019  25      2   2      2   3.17 3.77   5500        N

### Data Preprocessing

There are missing data in this dataset. For simplicity, let’s use mean
imputation.

```r
dt$age[is.na(dt$age)] <- mean(dt$age, na.rm = TRUE)
dt$jobsat[is.na(dt$jobsat)] <- mean(dt$jobsat, na.rm = TRUE)
dt$perf[is.na(dt$perf)] <- mean(dt$perf, na.rm = TRUE)
dt$salary[is.na(dt$salary)] <- mean(dt$salary, na.rm = TRUE)

head(dt, 10)
```
    ##    year_hired      age gender edu tenure   jobsat     perf salary turnover
    ## 1        2012 36.00000      2   3      4 3.860000 1.930000   6850        Y
    ## 2        2017 35.96815      2   3      4 5.000000 3.389545   5000        N
    ## 3        2016 40.00000      2   4      5 2.070000 3.389545   6800        N
    ## 4        2015 35.96815      1   3      6 3.340000 2.840000  10000        N
    ## 5        2017 22.00000      1   3      4 2.955568 5.000000   5000        N
    ## 6        2017 24.00000      2   3      4 3.810000 4.870000   5000        N
    ## 7        2015 35.96815      2   3      6 1.900000 4.390000  10000        N
    ## 8        2002 57.00000      2   3      3 2.955568 2.210000   6850        Y
    ## 9        2012 33.00000      1   3      4 1.000000 3.389545   6850        Y
    ## 10       2019 25.00000      2   2      2 3.170000 3.770000   5500        N

### Train-Test Split and Train the Model

Now let’s set aside a test set for evaluating model performance on new
data, and train the model on the remaining training data. We are using
the elastic net algorithm and random parameter search via 10-fold
cross-validation.

```r
set.seed(SEED)

# train-test split, use a random 70% of the data for training
idx <- sample(1:nrow(dt), floor(nrow(dt) * 0.70))
dt_train <- dt[idx, ]
dt_test <- dt[-idx, ]

# specify the predictors
predictors <- c("year_hired", "age", "gender", "edu",
                "tenure", "jobsat", "perf", "salary")

# train the model
control <- trainControl(method = "cv",
                        number = 10,
                        search = "random",
                        verboseIter = FALSE)

fit1 <- train(x = dt_train[, predictors],
                y = dt_train[, "turnover"],
                metric = "Accuracy",
                method = "glmnet",
                trControl = control,
                tuneLength = 30)
```

Let’s check the prediction accuracy in the test set:

```r
yhat <- predict(fit1, dt_test[, predictors])

sum(yhat == dt_test[, "turnover"]) / nrow(dt_test)
```
    ## [1] 1

It looks like we achieved 100% accuracy. Such a result is almost
impossible, and it probably suggests some data leakage issues.

## Issue 1: Variable Selection

Let’s check the data again.

```r
head(dt, 10)
```
    ##    year_hired      age gender edu tenure   jobsat     perf salary turnover
    ## 1        2012 36.00000      2   3      4 3.860000 1.930000   6850        Y
    ## 2        2017 35.96815      2   3      4 5.000000 3.389545   5000        N
    ## 3        2016 40.00000      2   4      5 2.070000 3.389545   6800        N
    ## 4        2015 35.96815      1   3      6 3.340000 2.840000  10000        N
    ## 5        2017 22.00000      1   3      4 2.955568 5.000000   5000        N
    ## 6        2017 24.00000      2   3      4 3.810000 4.870000   5000        N
    ## 7        2015 35.96815      2   3      6 1.900000 4.390000  10000        N
    ## 8        2002 57.00000      2   3      3 2.955568 2.210000   6850        Y
    ## 9        2012 33.00000      1   3      4 1.000000 3.389545   6850        Y
    ## 10       2019 25.00000      2   2      2 3.170000 3.770000   5500        N

Notice that in the data, the variable “salary” has an interesting
pattern: salary is missing (imputed by the mean salary, $6,850) for
employees who have left the organization. This makes sense because if an
employee has left, we do not need to pay a salary anymore. But this
directly implies that the employee has left, and the algorithm will be
able to capture this relationship (salary = 6,850 -&gt; turnover = “Y”,
otherwise -&gt; turnover = “N”). However, the goal of the model is to
predict turnover, so we will not be able to observe such a pattern in
“salary” when the model is used on future data. Therefore, we should not
include this variable. If we want to use salary as a predictor, we
should obtain salary information that does not indicate turnover, such
as starting salary, annual salary, last paid salary, etc.

Let’s remove the salary column and train the model again.

```r
# remove the salary variable
predictors <- c("year_hired", "age", "gender", "edu",
                "tenure", "jobsat", "perf")

# fit the model again
set.seed(SEED)
fit2 <- train(x = dt_train[, predictors],
                y = dt_train[, "turnover"],
                metric = "Accuracy",
                method = "glmnet",
                trControl = control,
                tuneLength = 30)

# check prediction results
yhat <- predict(fit2, dt_test[, predictors])
sum(yhat == dt_test[, "turnover"]) / nrow(dt_test)
```
    ## [1] 1

We have removed the salary variable, but our model still achieved
perfect accuracy. Let’s take a closer look.

## Issue 2: Domain Knowledge

The second issue requires the data analyst to have relevant knowledge
about the data and task. In this example, “year\_hired” and “tenure”
together leaked information about turnover into model training.

Notice that:

-   For employees who stayed in the organization, year\_hired + tenure =
    current year
-   For employees who have left, year\_hired + tenure &lt; current year

In reality, this may be due to the data management workflow. For
example, one database stores data about recruitment and hiring, and
another database stores data regarding performance management. When the
analyst merges data, he/she may include “year\_hired” from the
recruitment and hiring database, and “tenure” from the performance
management database. To identify and avoid such kind of data leakage
issues, the analyst should have specific knowledge about the problem.

To deal with this issue, let’s remove “year\_hired”. After removing
“year\_hired”, the prediction accuracy is 80%.

```r
set.seed(SEED)

# remove year_hired
predictors <- c("age", "gender", "edu",
                "tenure", "jobsat", "perf")

# fit the model again
fit3 <- train(x = dt_train[, predictors],
              y = dt_train[, "turnover"],
              metric = "Accuracy",
              method = "glmnet",
              trControl = control,
              tuneLength = 30)

# check prediction results
yhat <- predict(fit3, dt_test[, predictors])
sum(yhat == dt_test[, "turnover"]) / nrow(dt_test)
```
    ## [1] 0.8


## Issue 3: Data Preprocessing

So far we have looked at two examples of column-wise leakage, or leaking
features. There is also another form of leakage called row-wise leakage,
or leakage in training data (Kaufman et al., 2012).

Recall that we imputed missing data using the means of the whole sample
(which contained both thr training and test set). But this is not
possible because we in reality we only have the training set. To prevent
data leakage, we have to pretend that we don’t have access to the test
set. Imputing missing data using the whole sample will expose the model
to some information about the test data.

Therefore, missing data imputation should only happen within the
training set. Operationally, we should do train-test split first, then
impute missing data (also see a note at the end).

Let’s fix this issue:

```r
# reload data
dt <- read.csv("toydata.csv")

# train-test split first
dt_train <- dt[idx, ]
dt_test <- dt[-idx, ]

# impute missing data in the training set
dt_train$age[is.na(dt_train$age)] <- mean(dt_train$age, na.rm = TRUE)
dt_train$jobsat[is.na(dt_train$jobsat)] <- mean(dt_train$jobsat, na.rm = TRUE)
dt_train$perf[is.na(dt_train$perf)] <- mean(dt_train$perf, na.rm = TRUE)

# do the same to test set
dt_test$age[is.na(dt_test$age)] <- mean(dt_train$age, na.rm = TRUE)
dt_test$jobsat[is.na(dt_test$jobsat)] <- mean(dt_train$jobsat, na.rm = TRUE)
dt_test$perf[is.na(dt_test$perf)] <- mean(dt_train$perf, na.rm = TRUE)
# notice that we used means of the training set to impute missing data in the test set

# fit the model again
set.seed(SEED)

fit4 <- train(x = dt_train[, predictors],
              y = dt_train[, "turnover"],
              metric = "Accuracy",
              method = "glmnet",
              trControl = control,
              tuneLength = 30)

# check prediction results
yhat <- predict(fit4, dt_test[, predictors])
sum(yhat == dt_test[, "turnover"]) / nrow(dt_test)
```
    ## [1] 0.7

## Avoiding Data Leakage

From the above example, we can see how data leakage can cause the model
to perform unrealistically well in the available data. Such good model
performance is almost certainly not going to happen in real
applications. Thus, detecting and avoiding data leakage is crucial.

First of all, while a prediction accuracy of 100% often means leakage
(especially column-wise leakage, as in two of our examples), we cannot
rely on a 100% prediction accuracy to detect data leakage. In our
example, “year\_hired” and “tenure” together determined turnover, but in
real-world datasets, there might be missing data issues or errors when
inputting data, making prediction accuracy not perfectly 100%, even when
data leakage is present.

There are two steps we can take to avoid column-wise leakage:

-   Carefully select predictors, avoid predictors that imply the
    outcome. Exploratory data analysis often helps. For example, if a
    variable is highly correlated (e.g., correlation is close to 1) with
    the outcome, then a more careful inspection of the nature of the
    variable is probably needed.
-   Acquire domain-specific knowledge about the problem. This is highly
    context-dependent and there is not a standard solution. In our
    example, the analyst should be sensitive of potential relationships
    in the data stored in different databases, and work closely with
    coworkers who are in charge of collecting and managing employee
    data. Also, the analyst should have relevant knowledge about
    organizational tenure to avoid issues such as “year\_hired” +
    “tenure” =&gt; “turnover”.

To avoid row-wise leakage, a healthy model training pipeline is very
beneficial. In our example, we should do train-test split first, and set
the test data aside. After this, we preprocess training data, use
training data to process the test data, and never use any information
from the test data. A pipeline can get very complicated. Popular machine
learning packages (e.g., “tidymodels” and “caret” in R, “scikit-learn”
in Python) have tools to build up pipelines.

*Note. The data preprocessing procedure (i.e., train-test split first,
then impute missing data in the training set only) was in fact not ideal
and for demonstration purposes only. One should incorporate data
preprocessing into the cross-validation process, and it can be treated
as a part of the model parameter as well. For example, a model training
pipeline can be established so that it searches the best missing data
imputation strategy based on cross-validation.*
