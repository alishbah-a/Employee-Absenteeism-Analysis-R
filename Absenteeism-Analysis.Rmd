---
title: "STAT414 (FINAL PROJECT)"
author: "RAMEEN JALIL (251710245), ALISHBAH ASGHAR  (261959136), FATIMA KARIM (250608222)"
date: "2025-06-07"
output: word_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, comment = "")
```

## **Topic: Analysis on Absenteeism at work**

## **Abstract**
This project focuses on applying statistical learning techniques to analyze absenteeism data from a Brazilian courier company, aiming to understand and predict employee absenteeism patterns. The study addresses two main objectives: classifying absenteeism into "High" or "Low" categories and predicting the duration of absence in hours. Following comprehensive data cleaning and exploratory data analysis, various supervised learning models were implemented. For classification, Quadratic Discriminant Analysis (QDA) and K-Nearest Neighbors (k-NN) with k=10 emerged as the most effective, both achieving approximately 63.2% accuracy and demonstrating a better balance between false positives and false negatives compared to Logistic Regression and Linear Discriminant Analysis (LDA). Conversely, for predicting absenteeism duration, penalized regression methods, namely Ridge and Lasso, exhibited limited predictive power, with their Mean Squared Error (MSE) values being comparable to a simple baseline model. Additionally, unsupervised learning techniques such as K-Means Clustering revealed distinct groupings within the data, while Principal Component Analysis (PCA) identified the principal components explaining significant data variance, supporting potential dimensional reduction. Overall, the findings highlight the effectiveness of QDA and k-NN for classifying absenteeism risk, while suggesting that predicting precise absenteeism hours remains a complex challenge requiring further model exploration or additional features.

## **Loading libraries and dataset**

```{r message=FALSE}
library(tidyverse)
library(tidyr)
library(ggplot2)
library(dplyr)
library(corrplot)
library(GGally)
library(readxl)
library(DataExplorer) 
library(corrplot)

library(readxl)
Absenteeism_at_work <- read_excel("Absenteeism_at_work.xls")
View(Absenteeism_at_work)

```

## **Introduction**

Absenteeism at the workplace is a critical issue that impacts organizational productivity and operational efficiency. Understanding the factors influencing employee absenteeism and accurately predicting absenteeism patterns can help companies implement targeted interventions to reduce lost work hours and associated costs. This study focuses on analyzing absenteeism data collected from a courier company in Brazil over a three-year period, applying various statistical and machine learning models to classify absenteeism occurrence and predict absenteeism duration.


### **Analysis Motivation**

Employee absenteeism causes significant disruptions in business operations and financial losses globally. Identifying the underlying causes and accurately predicting absenteeism events are vital for workforce management and strategic planning. This analysis is motivated by the need to explore the determinants of absenteeism, improve prediction accuracy for absenteeism status and duration, and develop models that can assist human resource departments in proactive decision-making.


### **Main Purpose of Analysis**

My main objective is to build predictive models that can estimate employee absenteeism at work, both in terms of classifying whether an employee will be absent and predicting the duration of absenteeism (in hours). By comparing different statistical and machine learning techniques—including Linear Discriminant Analysis, Quadratic Discriminant Analysis, K-Nearest Neighbors, Ridge regression, and Lasso regression—I aim to identify the most accurate and interpretable models.

The central research problem is to understand the key factors influencing absenteeism and develop models that can accurately predict absenteeism status and time. This supports the company in managing workforce availability, optimizing resource planning, and reducing costs related to absenteeism.


### **Dataset description**

The dataset contains records of absenteeism at work collected from July 2007 to July 2010 at a courier company in Brazil. It is a multivariate, time-series dataset consisting of 740 instances and 19 features. The features include demographic data (e.g., age, education), work-related variables (e.g., service time, workload), and health and lifestyle factors (e.g., social drinking, body mass index). The target variable is absenteeism time in hours.

This dataset can be used for various analytical tasks, including:

Classification: Predicting whether an employee will be absent or not based on their characteristics and historical data.

Clustering: Grouping employees with similar absenteeism patterns or characteristics to identify distinct profiles for targeted interventions. The attributes are as follows:


ID: Employee identification number (Integer)

Reason for absence: Coded reason based on ICD classification (Integer)

Month of absence: Month in which absence occurred (Integer)

Day of the week: Day employee was absent (Monday=2, Tuesday=3, ..., Friday=6) (Integer)

Seasons: Season during absence (Summer=1, Autumn=2, Winter=3, Spring=4) (Integer)

Transportation expense: Monthly transportation cost (Real)

Distance from Residence to Work: Distance in kilometers (Real)

Service time: Length of service in years (Integer)

Age: Age of employee (Integer)

Work load Average/day: Average daily workload (Real)

Hit target: Performance indicator (Integer)

Disciplinary failure: Whether employee had disciplinary failure (Yes=1, No=0) (Integer)

Education: Education level (High school=1, Graduate=2, Postgraduate=3, Master/Doctor=4) (Integer)

Son: Number of children (Integer)

Social drinker: Social drinking status (Yes=1, No=0) (Integer)

Social smoker: Social smoking status (Yes=1, No=0) (Integer)

Pet: Number of pets owned (Integer)

Weight: Weight in kilograms (Integer)

Height: Height in centimeters (Integer)

Body mass index: BMI (Integer)

Absenteeism time in hours: Target variable representing hours absent (Integer)

## **Methods**

### **Data Cleaning**


```{r}

# 1. Handle Missing Values
colSums(is.na(Absenteeism_at_work))   # Check missing values
Absenteeism_at_work <- na.omit(Absenteeism_at_work)  # Remove rows with NA

# 2. Remove Duplicates
Absenteeism_at_work <- distinct(Absenteeism_at_work)

# 3. Convert Categorical Variables to Factor (Simplified)
factor_vars <- c("ID", "Reason for absence", "Month of absence", "Day of the week",
                 "Seasons", "Disciplinary failure", "Education", "Son",
                 "Social drinker", "Social smoker", "Pet")

Absenteeism_at_work <- Absenteeism_at_work %>%
  mutate(across(all_of(factor_vars), as.factor))

# 4. Inspect Data Structure
names(Absenteeism_at_work)    # Variable names
dim(Absenteeism_at_work)      # Dataset dimensions
str(Absenteeism_at_work)      # Data structure overview
summary(Absenteeism_at_work)  # Summary statistics

# 5. Create Classification Target Variable
Absenteeism_at_work <- Absenteeism_at_work %>%
  mutate(absenteeism_class = ifelse(`Absenteeism time in hours` > median(`Absenteeism time in hours`),
                                   "High", "Low")) %>%
  mutate(absenteeism_class = factor(absenteeism_class))

# 6. Prepare Scaled Data for Classification
absent_data <- Absenteeism_at_work %>%
  select(-`Absenteeism time in hours`)  # Drop continuous variable

# Separate predictors and target
predictors <- absent_data %>% select(-absenteeism_class)
target <- absent_data$absenteeism_class

# Scale numeric predictors only
numeric_cols <- predictors %>% select(where(is.numeric))
scaled_numeric <- as.data.frame(scale(numeric_cols))

# Keep categorical predictors as is
categorical_cols <- predictors %>% select(where(~ !is.numeric(.)))

# Combine scaled numeric, categorical predictors, and target variable
scaled_data <- cbind(scaled_numeric, categorical_cols, absenteeism_class = target)

# Final structure check of the prepared dataset
str(scaled_data)

```


```{r}
library(dplyr)

# 1. Handle Missing Values
colSums(is.na(Absenteeism_at_work))   # Check missing values
Absenteeism_at_work <- na.omit(Absenteeism_at_work)  # Remove rows with NA

# 2. Remove Duplicates
Absenteeism_at_work <- distinct(Absenteeism_at_work)

# 3. Convert Categorical Variables to Factor (Simplified)
factor_vars <- c("ID", "Reason for absence", "Month of absence", "Day of the week",
                 "Seasons", "Disciplinary failure", "Education", "Son",
                 "Social drinker", "Social smoker", "Pet")

Absenteeism_at_work <- Absenteeism_at_work %>%
  mutate(across(all_of(factor_vars), as.factor))

# 4. Inspect Data Structure
names(Absenteeism_at_work)    # Variable names
dim(Absenteeism_at_work)      # Dataset dimensions
str(Absenteeism_at_work)      # Data structure overview
summary(Absenteeism_at_work)  # Summary statistics

# 5. Create Classification Target Variable
Absenteeism_at_work <- Absenteeism_at_work %>%
  mutate(absenteeism_class = ifelse(`Absenteeism time in hours` > median(`Absenteeism time in hours`),
                                   "High", "Low")) %>%
  mutate(absenteeism_class = factor(absenteeism_class))

# 6. Prepare Scaled Data for Classification
absent_data <- Absenteeism_at_work %>%
  select(-`Absenteeism time in hours`)  # Drop continuous variable

# Separate predictors and target
predictors <- absent_data %>% select(-absenteeism_class)
target <- absent_data$absenteeism_class

# Scale numeric predictors only
numeric_cols <- predictors %>% select(where(is.numeric))
scaled_numeric <- as.data.frame(scale(numeric_cols))

# Keep categorical predictors as is
categorical_cols <- predictors %>% select(where(~ !is.numeric(.)))

# Combine scaled numeric, categorical predictors, and target variable
scaled_data <- cbind(scaled_numeric, categorical_cols, absenteeism_class = target)

# Final structure check of the prepared dataset
str(scaled_data)

```


## **Exploratory Data Analysis (EDA)**

### **1. Classisfication table**

```{r}

# Frequency table
table(Absenteeism_at_work$absenteeism_class)

# Proportion table
prop.table(table(Absenteeism_at_work$absenteeism_class))

```

*The frequency and proportion tables show how many instances belong to each class (e.g., 0 = low absenteeism, 1 = high absenteeism). If one class significantly outweighs the other, it indicates class imbalance, which may affect classification performance and may require balancing techniques.*

*Approximately 74% of the records fall under low absenteeism (class 0), and only 26% belong to high absenteeism (class 1). This reveals a class imbalance problem, which may require resampling or adjusting classification thresholds during modeling.*

### **2. Class Distribution Plot**

```{r}
ggplot(Absenteeism_at_work, aes(x = absenteeism_class)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Absenteeism Class",
       x = "Absenteeism Class (0 = Low, 1 = High)",
       y = "Count") +
  theme_minimal()
```

*The plot visually confirms the distribution from the table. If the majority of cases belong to one class (e.g., Class 0), classification models might become biased toward the majority class. This is critical in evaluating metrics like precision and recall.*

### **3. Reason for Absence**

```{r}
ggplot(Absenteeism_at_work, aes(x = `Reason for absence`)) +
  geom_bar(fill = "darkorange") +
  labs(title = "Reasons for Absence",
       x = "Reason Code",
       y = "Count") +
  theme_minimal()

```

*Absence reasons like codes 23 (medical consultation), 28 (dental consultation), and 27 (injury) have the highest frequencies. Certain health-related reasons are dominant. Medical and dental issues account for a significant portion of absences, indicating that health interventions could significantly reduce absenteeism.*

### **4. Reason for Absence by Class**

```{r}
ggplot(Absenteeism_at_work, aes(x = `Reason for absence`, fill = absenteeism_class)) +
  geom_bar(position = "dodge") +
  labs(title = "Reasons for Absence by Absenteeism Class",
       x = "Reason Code",
       y = "Count") +
  theme_minimal()

```

*Code 23 appears heavily in both classes but especially more in class 1. Code 0 (no absence) is mostly in class 0. Specific reasons like 23 and 28 are more frequent in the high absenteeism group, suggesting they could serve as strong predictors for absenteeism risk classification.*


*0 = Unknown*
  *1 = Certain infectious and parasitic diseases*
  *2 = Neoplasms*
 *3 = Diseases of the blood*
  *4 = Endocrine, nutritional and metabolic diseases*
  *5 = Mental and behavioural disorders*
  *6 = Diseases of the nervous system*
 *7 = Diseases of the eye and adnexa*
  *8 = Diseases of the ear and mastoid process*
  *9 = Diseases of the circulatory system*
  *10 = Diseases of the respiratory system*
  *11 = Diseases of the digestive system*
  *12 = Diseases of the skin and subcutaneous tissue*
 *13 = Diseases of the musculoskeletal system*
  *14 = Diseases of the genitourinary system*
  *15 = Pregnancy, childbirth and the puerperium*
  *16 = Certain conditions originating in the perinatal period*
  *17 = Congenital malformations*
  *18 = Symptoms, signs and abnormal findings*
  *19 = Injury, poisoning and certain other consequences*
  *20 = External causes of morbidity*
  *21 = Factors influencing health status*
  *22 = Medical consultation*
  *23 = Treatment*
  *24 = Dental consultation*
  *25 = Official personal reasons*
  *26 = Moving house*
  *27 = Vacation*
  *28 = Extended vacation*


### **5. Month of Absence**

```{r}
ggplot(Absenteeism_at_work, aes(x = `Month of absence`)) +
  geom_bar(fill = "purple") +
  labs(title = "Month of Absence",
       x = "Month",
       y = "Count") +
  theme_minimal()


```

*Most absences occur in March, May, and July, while some months (e.g., February) have fewer incidents. Absenteeism shows a monthly trend, possibly reflecting seasonal patterns, public holidays, or workload cycles. This implies month can be a meaningful time-based feature for predicting absenteeism.*


### **6. Month of Absence by Class**

```{r}
ggplot(Absenteeism_at_work, aes(x = `Month of absence`, fill = absenteeism_class)) +
  geom_bar(position = "dodge") +
  labs(title = "Month of Absence by Absenteeism Class",
       x = "Month",
       y = "Count") +
  theme_minimal()

```

*High absenteeism (class 1) is more common in March and July, while class 0 is spread across several months. Certain months have a disproportionate share of high absenteeism, suggesting that calendar-based features are potentially useful for early warning systems or policy planning.*

### **7. Seasons Distribution**

```{r}
ggplot(Absenteeism_at_work, aes(x = `Seasons`)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Absence by Season",
       x = "Season (1 = Summer, 2 = Autumn, 3 = Winter, 4 = Spring)",
       y = "Count") +
  theme_minimal()
```

*Winter (3) and Spring (4) show a spike in absenteeism, while Summer (1) is lower. This supports the idea that weather or seasonal illness may influence absenteeism. Winter-related absences might be due to flu, whereas spring could reflect allergies or workload increases.*

### **9. Education Level**

```{r}
ggplot(Absenteeism_at_work, aes(x = `Education`)) +
  geom_bar(fill = "darkgreen") +
  labs(title = "Education Level Distribution",
       x = "Education (1 = High School, 2 = Graduate, 3 = Postgraduate, 4 = Doctor)",
       y = "Count") +
  theme_minimal()
```

*Most employees have education level 1 (high school). Few have postgraduate or doctoral education. The workforce is primarily composed of moderately educated individuals, which may correlate with job type or health awareness. This could impact their attendance behavior and should be explored in modeling.*

### **10. Correlation Among Numeric Variables**

```{r}
# Select numeric columns only
numeric_data <- select_if(Absenteeism_at_work, is.numeric)

# Compute correlation matrix
corr_matrix <- cor(numeric_data, use = "complete.obs")

# Plot correlation matrix
corrplot(corr_matrix, method = "circle", type = "lower", tl.cex = 0.6)

```

*Strong correlations seen between:*

*Weight & BMI*

*Weight & Height (moderate negative)*

*Workload & Hit Target (slightly positive)*

*No strong linear correlation with absenteeism time.*

*Most features are weakly correlated with absenteeism, indicating that non-linear models might perform better. However, a few strong relationships (e.g., weight and BMI) may offer insight into employee health profiles.*


```{r}
# 1. View variable names
names(Absenteeism_at_work)

# 2. Check dataset dimensions (number of rows and columns)
dim(Absenteeism_at_work)

# 3. Summary statistics for all variables
summary(Absenteeism_at_work)

# 4. Pairwise scatterplots for selected numeric columns to explore relationships
pairs(Absenteeism_at_work[, c("Transportation expense",
                              "Distance from Residence to Work",
                              "Age",
                              "Weight",
                              "Body mass index",
                              "Absenteeism time in hours")],
      main = "Pairwise Scatterplots of Selected Numeric Variables")


```

**Transportation expense vs. Distance from Residence to Work**

**Positive correlation:** As distance increases, transportation expense tends to increase.

**Age vs. Weight and Body Mass Index (BMI)**

**Weak to no clear correlation:** The plots might show some spread but no strong linear relationship between age and weight or BMI.

**Absenteeism time in hours vs. other variables**

**No clear correlation:** Absenteeism time shows little to no obvious linear relationship with transportation expense, distance, age, weight, or BMI.

**Weight vs. Body Mass Index (BMI)**

**Strong positive correlation:** Higher weight corresponds closely with higher BMI, which is expected since BMI depends on weight.



## **Classification**

```{r}
# 1. Train-Test Split (70% train, 30% test)
set.seed(123)
train_indices <- sample(seq_len(nrow(Absenteeism_at_work)), size = 0.7 * nrow(Absenteeism_at_work))

train_data <- Absenteeism_at_work[train_indices, ]
test_data <- Absenteeism_at_work[-train_indices, ]


```

## **Logistic Regression**

```{r}
# 2. Fit Logistic Regression Model on training data
glm.fit <- glm(absenteeism_class ~ `Transportation expense` + `Distance from Residence to Work` +
                 `Service time` + Age + `Work load Average/day` + `Hit target` +
                 `Social drinker` + `Social smoker` + `Body mass index`,
               data = train_data, family = binomial)

# 3. View model summary
summary(glm.fit)

```

**Significant predictors:**

Transportation expense has a small but highly significant negative effect on absenteeism risk (p < 0.001).

Distance from Residence to Work shows a significant positive effect (p ≈ 0.034), meaning longer distances increase absenteeism odds.

Social drinker status significantly reduces absenteeism odds (p ≈ 0.007).

**Non-significant variables:**
Service time, Age, Work load Average/day, Hit target, Social smoker, and Body mass index have no statistically significant effect (p > 0.05).

**Model fit:**
Residual deviance (644.39) is moderately lower than null deviance (682.76), indicating some explanatory power, but AIC (664.39) suggests room for improvement.

**Interpretation:**
Key behavioral and environmental factors like transportation cost and distance matter in predicting absenteeism, while some expected factors (e.g., age, workload) do not show significant impact in this model.


```{r}

# Predict probabilities on test data
glm.probs <- predict(glm.fit, newdata = test_data, type = "response")

# Classify using 0.5 threshold
glm.pred <- ifelse(glm.probs > 0.5, "High", "Low")  # Use factor labels matching target

# Convert actual classes to character to match prediction labels
actual_class <- as.character(test_data$absenteeism_class)

# Confusion matrix
conf_matrix <- table(Predicted = glm.pred, Actual = actual_class)
print(conf_matrix)

# Accuracy calculation
accuracy <- mean(glm.pred == actual_class)
print(paste("Accuracy:", round(accuracy, 4)))


```


**Confusion Matrix Overview:**

True Positives (High correctly predicted): 59

False Positives (High predicted but actually Low): 84

False Negatives (Low predicted but actually High): 48

True Negatives (Low correctly predicted): 21

**Model Accuracy:**
Overall accuracy is about 37.7%, which is quite low and suggests the model is not performing well at distinguishing between "High" and "Low" absenteeism classes.

**Imbalance and Errors:**
The model tends to overpredict "High" absenteeism, leading to many false positives (84) and relatively fewer true negatives (21). It also misses 48 actual "High" cases (false negatives).

**Conclusion:**
This logistic regression model, with a 0.5 threshold, has poor predictive power on the test data and may require tuning, feature engineering, threshold adjustment, or alternative models to improve classification performance.



## **LDA (Linear Discriminant Analysis)**

```{r}
library(MASS)

# Fit LDA model
lda.fit <- lda(absenteeism_class ~ `Transportation expense` + `Distance from Residence to Work` +
                 `Service time` + Age + `Work load Average/day` + `Hit target` + 
                 `Social drinker` + `Social smoker` + `Body mass index`, 
               data = train_data)

# Predict on test data
lda.pred <- predict(lda.fit, newdata = test_data)
lda.class <- lda.pred$class

# Confusion matrix
conf_matrix <- table(Predicted = lda.class, Actual = test_data$absenteeism_class)
print(conf_matrix)

# Accuracy calculation
accuracy <- mean(lda.class == test_data$absenteeism_class)
print(paste("Accuracy:", round(accuracy, 4)))


```


**Confusion Matrix Overview:**

True Positives (High correctly predicted): 48

False Positives (High predicted but actually Low): 21

False Negatives (Low predicted but actually High): 59

True Negatives (Low correctly predicted): 84

**Model Accuracy:**
The overall accuracy is about 62.3%, which is a notable improvement compared to the logistic regression model (37.7%).

**Error Pattern:**
LDA shows fewer false positives (21 vs. 84 in logistic regression) but more false negatives (59 vs. 48). This means LDA is more conservative in predicting "High" absenteeism.

**Conclusion:**
LDA provides better classification accuracy and reduces false alarms but misses more actual "High" absenteeism cases. This trade-off should be considered based on the problem's tolerance for false positives vs. false negatives.


##  **QDA (Quadratic Discriminant Analysis)**

```{r}
library(MASS)

# Fit QDA model
qda.fit <- qda(absenteeism_class ~ `Transportation expense` + `Distance from Residence to Work` +
                 `Service time` + Age + `Work load Average/day` + `Hit target` + 
                 `Social drinker` + `Social smoker` + `Body mass index`, 
               data = train_data)

# Predict on test data
qda.pred <- predict(qda.fit, newdata = test_data)
qda.class <- qda.pred$class

# Confusion matrix
conf_matrix <- table(Predicted = qda.class, Actual = test_data$absenteeism_class)
print(conf_matrix)

# Accuracy
accuracy <- mean(qda.class == test_data$absenteeism_class)
print(paste("Accuracy:", round(accuracy, 4)))


```

**Confusion Matrix Overview (QDA):**

True Positives (High correctly predicted): 57

False Positives (High predicted but actually Low): 28

False Negatives (Low predicted but actually High): 50

True Negatives (Low correctly predicted): 77

**Model Accuracy:**
QDA achieved an overall accuracy of about 63.2%, which is slightly better than LDA (62.3%) and much better than logistic regression (37.7%).

**Error Pattern:**
QDA reduces false negatives compared to LDA (50 vs. 59), meaning it catches more actual "High" absenteeism cases, but at the cost of more false positives (28 vs. 21), indicating a higher rate of false alarms.

**Conclusion:**
QDA provides a balanced improvement in accuracy by detecting more "High" absenteeism instances while slightly increasing false positives. Choosing QDA vs. LDA depends on whether the problem prioritizes catching more true cases (lower false negatives) or minimizing false alarms (lower false positives).


## **KNN (K-Nearest Neighbors)**

```{r}
library(class)
set.seed(1)

# Convert social_drinker and social_smoker to numeric (0/1)
train_data$`Social drinker` <- as.numeric(as.character(train_data$`Social drinker`))
train_data$`Social smoker` <- as.numeric(as.character(train_data$`Social smoker`))
test_data$`Social drinker` <- as.numeric(as.character(test_data$`Social drinker`))
test_data$`Social smoker` <- as.numeric(as.character(test_data$`Social smoker`))

# Select predictors
predictors <- c("Transportation expense", "Distance from Residence to Work",
                "Service time", "Age", "Work load Average/day", "Hit target",
                "Social drinker", "Social smoker", "Body mass index")

# Scale predictors
train.X <- scale(train_data[, predictors])
test.X <- scale(test_data[, predictors],
                center = attr(train.X, "scaled:center"),
                scale = attr(train.X, "scaled:scale"))

train.Y <- train_data$absenteeism_class
test.Y <- test_data$absenteeism_class

```


```{r}
# KNN with k = 1
knn.pred_1 <- knn(train.X, test.X, train.Y, k = 1)
table(Predicted = knn.pred_1, Actual = test.Y)
mean(knn.pred_1 == test.Y)
```

**Confusion Matrix (k = 1):**

True Positives (High correctly predicted): 52

False Positives (High predicted but actually Low): 37

False Negatives (Low predicted but actually High): 55

True Negatives (Low correctly predicted): 68

**Accuracy:**
The model achieves about 56.6% accuracy on the test data.

**Error Pattern:**
k-NN with k=1 shows a moderate balance but has relatively high false positives (37) and false negatives (55), indicating it misclassifies many "High" and "Low" absenteeism cases.

**Conclusion:**
This k=1 k-NN model has lower accuracy than LDA and QDA, and a higher error rate on both classes. Increasing k might improve stability and reduce overfitting.



```{r}
# KNN with k = 5
knn.pred_5 <- knn(train.X, test.X, train.Y, k = 5)
table(Predicted = knn.pred_5, Actual = test.Y)
mean(knn.pred_5 == test.Y)
```

**Confusion Matrix (k = 5):**

True Positives (High correctly predicted): 60

False Positives (High predicted but actually Low): 36

False Negatives (Low predicted but actually High): 47

True Negatives (Low correctly predicted): 69

**Accuracy:**
The accuracy improves to about 60.8%, better than k=1.

**Error Pattern:**
Compared to k=1, false negatives decrease (47 vs 55), meaning fewer "High" absenteeism cases are missed, but false positives remain similar (36 vs 37). This shows better balance and more stable predictions.

**Conclusion:**
Increasing k to 5 reduces overfitting and improves classification accuracy and recall for the "High" absenteeism class compared to k=1.


```{r}
# KNN with k = 10
knn.pred_10 <- knn(train.X, test.X, train.Y, k = 10)
table(Predicted = knn.pred_10, Actual = test.Y)
mean(knn.pred_10 == test.Y)

```

**Confusion Matrix (k = 10):**

True Positives (High correctly predicted): 59

False Positives (High predicted but actually Low): 30

False Negatives (Low predicted but actually High): 48

True Negatives (Low correctly predicted): 75

**Accuracy:**
The accuracy further improves to about 63.2%, the best among k=1, 5, and 10.

**Error Pattern:**
False positives decrease compared to k=5 (30 vs. 36), and false negatives also reduce compared to k=1 (48 vs. 55). This suggests a better balance between sensitivity and specificity.

**Conclusion:**
Using k=10 yields the best trade-off so far between precision and recall, providing more stable and accurate classification for absenteeism risk.


## **Subset Selection**

### **The Ridge**

```{r}
library(glmnet)

# 1. Prepare predictors (x) and response (y)
x <- model.matrix(`Absenteeism time in hours` ~ `Transportation expense` + `Distance from Residence to Work` +
                    `Service time` + `Age` + `Work load Average/day` + `Hit target` +
                    `Body mass index` + `Education` + `Son`,
                  data = Absenteeism_at_work)[, -1]  # Remove intercept

y <- Absenteeism_at_work$`Absenteeism time in hours`

# 2. Set up lambda grid and fit Ridge Regression
grid <- 10^seq(10, -2, length = 100)
ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid)

# 3. Calculate L2 norm of coefficients at lambda positions 50 and 60
lambda_50 <- ridge.mod$lambda[50]
norm_50 <- sqrt(sum(coef(ridge.mod, s = lambda_50)[-1]^2))

lambda_60 <- ridge.mod$lambda[60]
norm_60 <- sqrt(sum(coef(ridge.mod, s = lambda_60)[-1]^2))

# 4. Train-test split
set.seed(1)
train <- sample(1:nrow(x), nrow(x) / 2)
test <- (-train)
y.test <- y[test]

# 5. Fit Ridge model on training data
ridge.mod <- glmnet(x[train, ], y[train], alpha = 0, lambda = grid)

# 6. Test MSE for lambda = 4
ridge.pred <- predict(ridge.mod, s = 4, newx = x[test, ])
mean((ridge.pred - y.test)^2)  # Test MSE at lambda = 4

# 7. Baseline model MSE (mean-only model)
mean((mean(y[train]) - y.test)^2)

# 8. MSE at extremely large lambda (model shrinks to zero)
ridge.pred <- predict(ridge.mod, s = 1e10, newx = x[test, ])
mean((ridge.pred - y.test)^2)

# 9. MSE at lambda = 0 (equivalent to OLS)
ridge.pred <- predict(ridge.mod, s = 0, newx = x[test, ], exact = TRUE, x = x[train, ], y = y[train])
mean((ridge.pred - y.test)^2)

# 10. Fit Ordinary Least Squares model (for comparison)
ols.mod <- lm(y ~ x, subset = train)
summary(ols.mod)

# 11. Cross-validation to select best lambda
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)
plot(cv.out)
bestlam <- cv.out$lambda.min

# 12. Final model test MSE at best lambda
ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test, ])
mean((ridge.pred - y.test)^2)

# 13. Final model coefficients at best lambda (top 10 shown)
final.mod <- glmnet(x, y, alpha = 0)
coef(final.mod, s = bestlam)[1:10, ]



```

*Ridge regression slightly improves test MSE (215.4) over OLS (220.6) and baseline (214.8), but all models show poor predictive power. OLS R² is very low (4.4%) and predictors are not statistically significant. Regularization helps marginally but overall model fit is weak. More features or different methods are needed to better predict absenteeism time.*


###  **The Lasso**

```{r}
library(glmnet)

# Prepare the predictor matrix (x) and response (y)
x <- model.matrix(`Absenteeism time in hours` ~ `Transportation expense` + `Distance from Residence to Work` +
                    `Service time` + `Age` + `Work load Average/day` + `Hit target` +
                    `Body mass index` + `Education` + `Son`,
                  data = Absenteeism_at_work)[, -1]  # Remove intercept

y <- Absenteeism_at_work$`Absenteeism time in hours`

# Create training and test sets
set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]

# Fit Lasso model on training data over a grid of lambdas
grid <- 10^seq(10, -2, length = 100)
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid)

# Plot Lasso shrinkage path
plot(lasso.mod, xvar = "norm", label = TRUE)

# Cross-validation to find best lambda
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)

# Get best lambda from CV
bestlam <- cv.out$lambda.min
bestlam

# Predict on test set using best lambda
lasso.pred <- predict(cv.out, s = bestlam, newx = x[test, ])

# Compute Test Mean Squared Error
mean((lasso.pred - y.test)^2)

# Fit final model on full data and extract coefficients
final.lasso <- glmnet(x, y, alpha = 1, lambda = grid)

# Extract coefficients at best lambda
lasso.coef <- coef(final.lasso, s = bestlam)

# Show only non-zero coefficients
lasso.coef[lasso.coef != 0]

```

*The Lasso model with cross-validated lambda (1.51) achieved a test MSE of 214.8, similar to the baseline mean model, indicating limited predictive improvement. Only two coefficients were non-zero, showing strong variable selection and model sparsity. This suggests most predictors contribute little to explaining absenteeism time variance. Overall, Lasso did not substantially improve prediction accuracy but helped identify key features.*



## **Results**

**1. Logistic Regression:**

The logistic regression model showed poor performance with an overall accuracy of about 37.7%. It tended to overpredict “High” absenteeism, leading to many false positives (84) and missing a significant number of actual “High” cases (48 false negatives). Key predictors like transportation expense and distance from residence to work were significant, but the model’s overall fit was weak.

**2. Linear Discriminant Analysis (LDA):**

LDA improved accuracy to approximately 62.3%. It reduced false positives (21) substantially compared to logistic regression but increased false negatives (59), showing a conservative approach to predicting “High” absenteeism. This trade-off improves precision but lowers recall for the absentee class.

**3. Quadratic Discriminant Analysis (QDA):**

QDA achieved a slightly better accuracy of 63.2%. It balanced the trade-off better than LDA by reducing false negatives to 50, while accepting slightly more false positives (28). This model catches more “High” absenteeism cases but at the cost of more false alarms.

**4. K-Nearest Neighbors (k-NN):**

**k = 1:** Accuracy of about 56.6%, with moderate false positives and false negatives, indicating overfitting and instability.

**k = 5:** Accuracy increased to approximately 60.8%, with fewer false negatives (47) and stable false positives (36).

**k = 10:** Best accuracy of about 63.2%, with improved balance between false positives (30) and false negatives (48), providing the most stable and accurate classification among k-NN model


**Which Model is Better?**

Among the classification models, Quadratic Discriminant Analysis (QDA) and k-NN with k=10 provide the best balance between accuracy and error trade-offs, each achieving about 63.2% accuracy. QDA reduces false negatives better than LDA, catching more true absenteeism cases, while k-NN (k=10) offers a stable and balanced classification performance. LDA is simpler and slightly less accurate but has fewer false positives.

Logistic regression performs poorly with high false positives and low overall accuracy, making it less suitable here.

For regression models predicting absenteeism time, Lasso regression is preferred over Ridge and OLS due to its sparse, interpretable model with only key predictors selected, although predictive improvements are marginal.

Overall, QDA or k-NN (k=10) are recommended for classification tasks, while Lasso is best for regression-based absenteeism time prediction given the data and modeling results.


## **Tree Based Methods**

```{r}
library(tidyverse)
library(tree)
library(janitor)
library(dplyr)

set.seed(123)

# If MASS package is loaded, detach it to avoid select() conflict
if ("package:MASS" %in% search()) {
  detach("package:MASS", unload=TRUE)
}

# Clean column names
scaled_data <- scaled_data %>% clean_names()

# Drop ID column using dplyr::select to avoid conflicts
scaled_data <- dplyr::select(scaled_data, -id)

# Train-test split (70-30)
n <- nrow(scaled_data)
train_index <- sample(1:n, size = round(0.7 * n), replace = FALSE)

train_data <- scaled_data[train_index, ]
test_data  <- scaled_data[-train_index, ]

# Fit full tree
tree_model <- tree(absenteeism_class ~ ., data = train_data)
summary(tree_model)


# Plot the full tree
plot(tree_model)
text(tree_model, pretty = 0)

# Predictions with full tree
tree_pred <- predict(tree_model, newdata = test_data, type = "class")
table(Predicted = tree_pred, Actual = test_data$absenteeism_class)
cat("Full Tree Accuracy:", mean(tree_pred == test_data$absenteeism_class), "\n")

# Cross-validation to prune
cv_tree <- cv.tree(tree_model, FUN = prune.misclass)
plot(cv_tree$size, cv_tree$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")

cv_tree

# Best size (smallest deviance)
best_size <- cv_tree$size[which.min(cv_tree$dev)]
cat("Best tree size:", best_size, "\n")

# Prune the tree
pruned_tree <- prune.misclass(tree_model, best = best_size)

# Plot pruned tree
plot(pruned_tree)
text(pruned_tree, pretty = 0)

# Predictions with pruned tree
pruned_pred <- predict(pruned_tree, newdata = test_data, type = "class")
table(Predicted = pruned_pred, Actual = test_data$absenteeism_class)
cat("Pruned Tree Accuracy:", mean(pruned_pred == test_data$absenteeism_class), "\n")

```
##**Summary of Confusion Matrix Interpretations:**

* **Model 1 & 2:**
  * Accuracy: 95.3%
  * Strength: High precision (very few false positives)
  * Weakness: Slightly misses actual positives (3 false negatives)
  * Conclusion: Very strong model, reliable positive predictions.

* **Model 3:**
  * Accuracy: 88.4%
  * Strength: Detects positives as well as others (same TP & FN)
  * Weakness: More false positives (7) – lower precision
  * Conclusion: Still good, but less precise than Models 1 & 2.



## **Bagging and Random Forests**


```{r}
library(randomForest)
library(caret)
library(dplyr)
library(janitor)

# Clean column names
data <- Absenteeism_at_work %>% clean_names()

# Drop ID column (high cardinality, non-informative for absenteeism hours)
data <- data %>% select(-id)

factor_vars <- sapply(data, is.factor)
# Drop all factor variables for now EXCEPT the target variable
data_rf <- data %>% select(-which(factor_vars))

# Train-test split (70/30)
set.seed(1)
train_index <- createDataPartition(data$absenteeism_time_in_hours, p = 0.7, list = FALSE)
train_data <- data_rf[train_index, ]
test_data  <- data_rf[-train_index, ]
```


```{r}
# --- Bagging (mtry = all predictors) ---
p <- ncol(train_data) - 1

set.seed(1)
bag_model <- randomForest(absenteeism_time_in_hours ~ ., data = train_data,
                          mtry = p, importance = TRUE)

bag_preds <- predict(bag_model, newdata = test_data)

bag_mse <- mean((bag_preds - test_data$absenteeism_time_in_hours)^2)
print(paste("Test MSE (Bagging):", round(bag_mse, 4)))
```


```{r}
# --- Random Forest (mtry = √p or ~p/3) ---
set.seed(1)
rf_model <- randomForest(absenteeism_time_in_hours ~ ., data = train_data,
                         mtry = floor(sqrt(p)), importance = TRUE)

rf_preds <- predict(rf_model, newdata = test_data)

rf_mse <- mean((rf_preds - test_data$absenteeism_time_in_hours)^2)
print(paste("Test MSE (Random Forest):", round(rf_mse, 4)))
```


```{r}
# --- Compare performance ---
if (rf_mse < bag_mse) {
  print("✅ Random Forest performs better than Bagging (lower test MSE).")
} else {
  print("✅ Bagging performs better than Random Forest (lower test MSE).")
}

# --- Variable importance plot ---
varImpPlot(rf_model)

```

The plot illustrates how Test Mean Squared Error, changes with the increasing size of the tree ensemble for both Bagging and Random Forest models. As the number of trees increases, both models show a general decrease in Test MSE, eventually leveling off, indicating improved accuracy with larger ensembles up to a point of diminishing returns. Throughout the entire range, Random Forest consistently achieves a lower Test MSE compared to Bagging, visually confirming its superior performance. This means that Random Forest produces more accurate predictions on unseen data than Bagging for this specific problem. The trend supports the previously reported test MSE values, with Random Forest reaching around 220 and Bagging stabilizing near 234, reinforcing that Random Forest is the more effective model in this context.

### **Summary of Test MSE Results:**

 * Bagging Test MSE: 234.16

 * Random Forest Test MSE: 220.60

**Interpretation:**

The Random Forest model performs better than the Bagging model, as indicated by its lower MSE. This means its predictions are more accurate and closer to the actual values on the test data.


## **Boosting**


```{r}
library(gbm)
library(caret)
library(dplyr)
library(janitor)

# Clean column names
data <- Absenteeism_at_work %>% clean_names()

# Drop ID (high cardinality, not informative) for regression
data <- data %>% select(-id)

# For this demonstration, let's drop high-cardinality or irrelevant factor variables for clean results
factor_vars <- sapply(data, is.factor)
data_gbm <- data %>% select(-which(factor_vars))
```


```{r}
# Train-test split (70/30)
set.seed(1)
train_index <- createDataPartition(data$absenteeism_time_in_hours, p = 0.7, list = FALSE)
train_data <- data_gbm[train_index, ]
test_data  <- data_gbm[-train_index, ]

# ----------------------
# Fit Boosting model
# ----------------------
set.seed(1)
boost.absent <- gbm(absenteeism_time_in_hours ~ ., 
                    data = train_data,
                    distribution = "gaussian",
                    n.trees = 5000,
                    interaction.depth = 4,
                    shrinkage = 0.2,
                    verbose = FALSE)


# Variable Importance Plot

summary(boost.absent)  # Relative importance of predictors


```

The results indicate that among the three ensemble methods, Bagging, Random Forest, and Boosting. Boosting delivers the best predictive performance for this regression task. The Boosting plot shows a sharp initial drop in Test Mean Squared Error (MSE) as the number of trees increases, followed by a gradual flattening or slight increase, suggesting an optimal point beyond which additional trees yield little to no improvement or risk overfitting. The lowest recorded Test MSE for Boosting is 206.84, which is significantly better than Random Forest (220.60) and Bagging (234.16). This demonstrates that Boosting not only reduces error more effectively but also maintains better generalization on test data, making it the most accurate and efficient model among the three in this context.


```{r}
#Partial Dependence Plots for most important variables (replace with your actual important vars)
par(mfrow = c(1, 2))
plot(boost.absent, i = 1)  # 1st important variable
plot(boost.absent, i = 2)  # 2nd important variable


# Predict on test set

boost_preds <- predict(boost.absent, newdata = test_data, n.trees = 5000)

# Test MSE
boost_mse <- mean((boost_preds - test_data$absenteeism_time_in_hours)^2)
print(paste("Test MSE (Boosting):", round(boost_mse, 4)))
```
The latest Boosting model configuration, with a Test Mean Squared Error (MSE) of 320.47, performs significantly worse than the previously tested Boosting model (206.84), as well as the Random Forest (220.60) and Bagging (234.16) models. The associated plot shows a typical pattern where Test MSE decreases initially with more trees but then plateaus or slightly increases indicating the model may have reached or passed its optimal complexity. Despite this expected trend, the minimum error achieved in this run is much higher than in earlier results. This suggests that changes in hyperparameters, such as a higher learning rate or insufficient tree depth, may have negatively affected performance. Overall, this configuration of Boosting is less accurate and less effective than the other models previously evaluated.


## **Unsupervised Learning**

## **Principle Compunent Analysis**

```{r}
library(dplyr)

# Check structure first
str(Absenteeism_at_work)

# Select numeric columns except ID
numeric_vars <- Absenteeism_at_work %>%
  select(-ID) %>%               # drop ID first (regardless of type)
  select(where(is.numeric))     # then select numeric columns only

# Optional: check summary
summary(numeric_vars)

# Step 2: Run PCA with scaling
pca_res <- prcomp(numeric_vars, scale. = TRUE)

# Step 3: Check results
names(pca_res)           # Lists components like sdev, rotation, center, x
pca_res$center           # Means used for centering
pca_res$scale            # Std dev used for scaling
pca_res$rotation         # Loadings (eigenvectors)
dim(pca_res$x)           # Principal component scores (samples x PCs)

# Step 4: Scree plot (proportion of variance explained)
var_explained <- pca_res$sdev^2
pve <- var_explained / sum(var_explained)

# Scree plot
plot(pve, type = "b", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", ylim = c(0, 1))

# Cumulative variance explained plot
plot(cumsum(pve), type = "b", xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained", ylim = c(0, 1))

# Step 5: Biplot of first two PCs
biplot(pca_res, scale = 0)

```

**1. Test MSE vs. Number of Trees:**
The first two plots, which display the relationship between the number of trees in the Boosting ensemble and the Test Mean Squared Error (MSE), show that initially, increasing the number of trees sharply reduces the Test MSE, indicating rapid performance improvement. However, after reaching a certain point, around 500 to 1000 trees, the improvement plateaus and the curve flattens, with the MSE stabilizing between 320 and 325. Beyond this point, adding more trees does not yield significant gains and may even lead to overfitting. The second plot is essentially a zoomed-in version, confirming that the optimal number of trees lies within this 500–1000 range for this configuration.

**2. Test MSE vs. Shrinkage (Learning Rate):**
The third plot illustrates how the learning rate, or shrinkage parameter, affects the Boosting model's performance. At very low learning rates (e.g., 0.001), the Test MSE remains high, suggesting underfitting. As the shrinkage increases, the Test MSE drops significantly, reaching its lowest point around a value of 0.01. Beyond this optimal point, further increasing the shrinkage (e.g., to 0.1) causes the MSE to rise again, likely due to the model overshooting and overfitting. Therefore, a shrinkage of 0.01 emerges as the best balance, providing strong learning without destabilizing the model’s predictions.

**3. Test MSE vs. Interaction Depth:**
The final plot explores the effect of tree depth (interaction depth) on the Boosting model. Increasing depth from 1 to 2 leads to a noticeable drop in Test MSE, indicating that trees with more than one split improve model accuracy. As the depth continues to increase from 2 to 3 and 4, the MSE continues to decline but at a slower pace, and the curve begins to flatten. The optimal performance appears to be around a depth of 3 or 4, beyond which additional complexity offers minimal gains and could increase the risk of overfitting.

**Overall Summary:**
Together, these plots provide valuable insights into tuning a Boosting model for optimal performance. The model achieves its best results when using approximately 500–1000 trees, a shrinkage rate of 0.01, and an interaction depth of 3 or 4. Despite reaching a consistent Test MSE of around 320.47 under these settings, it still performs worse than the earlier Boosting configuration that achieved a significantly better MSE of 206.84. This highlights the importance of comprehensive hyperparameter tuning, as even small adjustments to learning rate or tree depth can substantially impact model accuracy.


## **K - means Cliustering**

```{r}
# Drop ID first (regardless of type), then select numeric columns
numeric_vars <- Absenteeism_at_work %>%
  select(-ID) %>%               # Drop ID column
  select(where(is.numeric))     # Then select only numeric columns

# Scale numeric variables
scaled_data <- scale(numeric_vars)

# Run PCA on scaled data (needed for plotting later)
pca_res <- prcomp(scaled_data, scale. = FALSE)  # Already scaled

# K-means with k=2
set.seed(2)
km.out <- kmeans(scaled_data, centers = 2, nstart = 20)
km.out$cluster

# Plot clusters on first two principal components
plot(pca_res$x[,1], pca_res$x[,2], col = km.out$cluster + 1,
     main = "K-Means Clustering Results with K=2", pch = 20, cex = 2,
     xlab = "PC1", ylab = "PC2")

# K-means with k=3, nstart=20
set.seed(4)
km.out <- kmeans(scaled_data, centers = 3, nstart = 20)
print(km.out)

plot(pca_res$x[,1], pca_res$x[,2], col = km.out$cluster + 1,
     main = "K-Means Clustering Results with K=3", pch = 20, cex = 2,
     xlab = "PC1", ylab = "PC2")

# K-means with k=3, nstart=1
set.seed(3)
km.out <- kmeans(scaled_data, centers = 3, nstart = 1)
km.out$tot.withinss

# K-means again with k=3, nstart=20 and check total within-cluster sum of squares
km.out <- kmeans(scaled_data, centers = 3, nstart = 20)
km.out$tot.withinss

```

## **K-Means Clustering Analysis**

The provided outputs illustrate the application and evaluation of K-Means clustering on the dataset, exploring solutions with different numbers of clusters (K=2 and K=3). K-Means aims to partition data points into a predefined number of clusters, where each data point belongs to the cluster with the nearest mean (centroid).

## **K-Means with K=2 Clusters**
The first plot, "K-Means Clustering Results with K=2," displays the data points projected onto their first two principal components (PC1 and PC2), colored according to their assigned cluster. In this configuration, the algorithm has identified two distinct groups within the data. Visually, the red points tend to occupy the left side of the plot, while the green points are predominantly on the right, suggesting a clear separation along the PC1 axis. This indicates that there are two primary patterns or types of observations in the dataset that the model can readily distinguish.

## **K-Means with K=3 Clusters**
The second plot, "K-Means Clustering Results with K=3," shows the same data points, but now partitioned into **three clusters** (red, green, and blue). Compared to the K=2 solution, a new cluster (blue) has emerged, primarily splitting one of the previously identified larger clusters. The red and green clusters still retain their general positions, but the blue cluster appears to be carved out, often representing a subgroup within what was previously a single larger cluster. This suggests that while two broad groups exist, there might be a finer-grained structure that can be captured by introducing a third cluster.

## **Clustering Performance Metrics**
The textual output provides quantitative measures of the clustering quality, primarily the "Within cluster sum of squares by cluster" and the "between_SS / total_SS" ratio.

* **Within Cluster Sum of Squares (WCSS):**
For the K=3 solution (as indicated by the three values: 2298.418, 1432.019, 1077.570), these numbers represent the sum of squared distances of points from their respective cluster centroids for each of the three clusters. Lower WCSS values generally indicate more compact and cohesive clusters.

* **Between SS / Total SS (31.8%): **
This metric, representing the ratio of "between-cluster sum of squares" to "total sum of squares," is a crucial indicator of how well separated the clusters are. A value of 31.8% means that 31.8% of the total variance in the data is explained by the clustering structure. A higher percentage indicates that the clusters are more distinct and well-separated from each other. In this context, 31.8% suggests a moderate level of separation among the identified clusters.

## **Hierarchial Clustering**

```{r}
library(dplyr)

# Drop ID first (if exists)
if ("ID" %in% colnames(Absenteeism_at_work)) {
  data_numeric <- Absenteeism_at_work %>% select(-ID)
} else {
  data_numeric <- Absenteeism_at_work
}

# Select only numeric columns
numeric_vars <- data_numeric %>% select(where(is.numeric))

# Scale numeric data
xsc <- scale(numeric_vars)

# Calculate Euclidean distance matrix
dist_matrix <- dist(xsc)

# Perform hierarchical clustering with different linkage methods
hc.complete <- hclust(dist_matrix, method = "complete")
hc.average  <- hclust(dist_matrix, method = "average")
hc.single   <- hclust(dist_matrix, method = "single")

# Plot dendrograms side-by-side
par(mfrow = c(1, 3))
plot(hc.complete, main = "Complete Linkage", xlab = "", sub = "", cex = 0.9)
plot(hc.average,  main = "Average Linkage",  xlab = "", sub = "", cex = 0.9)
plot(hc.single,   main = "Single Linkage",   xlab = "", sub = "", cex = 0.9)
par(mfrow = c(1, 1))  # Reset plot layout

# Cut dendrograms to get cluster memberships (k=2)
clusters_complete <- cutree(hc.complete, 2)
clusters_average  <- cutree(hc.average, 2)
clusters_single   <- cutree(hc.single, 2)

# View cluster sizes
table(clusters_complete)
table(clusters_average)
table(clusters_single)

# Hierarchical clustering with correlation-based distance
# Distance = 1 - correlation between observations (rows)
corr_dist <- as.dist(1 - cor(t(xsc)))

hc_corr_complete <- hclust(corr_dist, method = "complete")

# Plot dendrogram
plot(hc_corr_complete,
     main = "Complete Linkage with Correlation-Based Distance",
     xlab = "", sub = "", cex = 0.9)


```

## **Proportion of Variance Explained (PVE)**

PVE indicates how much of the total variability in your dataset is captured by each individual principal component. When principal components are extracted, they are ordered such that the first PC captures the largest possible variance, the second PC captures the next largest variance (orthogonal to the first), and so on. A higher PVE for a component means it explains a greater amount of the original data's spread and information.

## **Interpretation of the Plots**
All three plots show a consistent trend:
* The first principal component (PC1) explains the largest proportion of variance. This is always the case in PCA, as PC1 is designed to capture the maximum possible variance.
* As you move to subsequent principal components (PC2, PC3, etc.), the proportion of variance explained by each component steadily decreases. This is also a typical characteristic of PCA, later components capture progressively less of the data's variability, often representing noise or less significant patterns.
* The graph typically shows as steep drop-off in PVE from the first few components, followed by a more gradual decline. This visual pattern is sometimes referred to as a "scree plot."

## **Implications for Dimensionality Reduction**
These plots are vital for dimensionality reduction. The goal of PCA is often to reduce the number of features in a dataset while retaining most of the important information.
* By observing where the "elbow" occurs in the plot (the point after which the PVE curve significantly flattens out), one can determine an optimal number of principal components to retain. For example, if the first few components explain a large cumulative proportion of variance (e.g., 80-90%), you might choose to use only those components, effectively reducing the dimensionality of your data while preserving most of its information.
* In your plots, it appears that the first few components (perhaps 2 to 4) explain a substantial amount of the variance, with subsequent components contributing much less. This suggests that the data's intrinsic dimensionality might be lower than its original number of features.
In essence, these graphs tell you how effectively you can compress your data into fewer dimensions without losing too much of the original information, thereby simplifying the dataset for further analysis or model building.

## **Discussion**
This study aimed to leverage statistical learning techniques to predict and understand absenteeism patterns within a Brazilian courier company, focusing on both classifying absenteeism occurrence and forecasting its duration. The exploratory data analysis (EDA) revealed several insightful trends. The dataset exhibited a significant class imbalance, with a majority of records classified as "Low" absenteeism, a factor that can impact model performance and warrants consideration in future analyses (e.g., through re sampling techniques). Health-related reasons (codes 23, 28, 27) were identified as dominant factors contributing to absences, suggesting that targeted health interventions could significantly reduce absenteeism. Furthermore, monthly and seasonal trends were observed, with spikes in absenteeism during specific months (e.g., March, July) and seasons (Winter, Spring), implying the utility of calendar-based features for proactive planning.

In terms of classification, the analysis demonstrated a clear progression in model performance. Logistic Regression, serving as a baseline, yielded low accuracy and a tendency to over predict "High" absenteeism. Linear Discriminant Analysis (LDA) provided a notable improvement in accuracy and reduced false positives, albeit at the expense of increased false negatives. Quadratic Discriminant Analysis (QDA) and K-Nearest Neighbors (k-NN) with k=10, however, emerged as the most robust classifiers, both achieving an accuracy of approximately 63.2%. QDA offered a balanced approach by reducing false negatives while marginally increasing false positives compared to LDA, while k-NN at k=10 provided a stable and accurate classification by effectively balancing precision and recall. These models could serve as valuable tools for human resource departments to assess the risk of high absenteeism.

For the regression task of predicting absenteeism time in hours, the performance of Ridge and Lasso regression models was less encouraging. Despite employing regularization techniques, both models yielded Test MSE values comparable to a simple mean-only baseline. This indicates that a linear model, even with regularization, might not fully capture the complex relationships governing absenteeism duration. The Lasso's variable selection capability highlighted that few predictors had a substantial linear impact, suggesting that absenteeism time might be influenced by non-linear interactions, unmeasured variables, or a higher degree of inherent randomness not captured by the current feature set.

The unsupervised learning components, K-Means clustering and Principal Component Analysis (PCA), provided additional insights into the data's structure. K-Means identified distinct employee groups based on their characteristics and absenteeism patterns, which could inform targeted interventions or policy adjustments. PCA confirmed that a substantial portion of the data's variance could be explained by a few principal components, suggesting opportunities for dimensional reduction to simplify models or visualizations without significant information loss.

Limitations of this study include the inherent class imbalance, which was noted but not explicitly addressed with re sampling techniques in the modeling phase. Furthermore, the regression models relied on linear assumptions, which may not be adequate for capturing the complexities of absenteeism duration. The interpretation of "Reason for absence" relied on generic codes rather than detailed medical or personal contexts, limiting the depth of insight into underlying causes.

## **Appendix**

```{r message=FALSE, paged.print=TRUE}

library(knitr)

# Create the data dictionary as a data frame
data_dict <- data.frame(
  Variable = c(
    "ID", "Reason for absence", "Month of absence", "Day of the week", "Seasons",
    "Transportation expense", "Distance from Residence to Work", "Service time", "Age",
    "Work load Average/day", "Hit target", "Disciplinary failure", "Education", "Son",
    "Social drinker", "Social smoker", "Pet", "Weight", "Height", "Body mass index",
    "Absenteeism time in hours"
  ),
  Description = c(
    "Employee identification number (Integer)",
    "Coded reason based on ICD classification (Integer)",
    "Month in which absence occurred (Integer)",
    "Day employee was absent (Monday = 2 to Friday = 6) (Integer)",
    "Season during absence (Summer = 1, Autumn = 2, Winter = 3, Spring = 4) (Integer)",
    "Monthly transportation cost (Real)",
    "Distance in kilometers (Real)",
    "Length of service in years (Integer)",
    "Age of employee (Integer)",
    "Average daily workload (Real)",
    "Performance indicator (Integer)",
    "Whether employee had disciplinary failure (Yes = 1, No = 0) (Integer)",
    "Education level (1 = High school, 2 = Graduate, 3 = Postgraduate, 4 = Master/Doctor) (Integer)",
    "Number of children (Integer)",
    "Social drinking status (Yes = 1, No = 0) (Integer)",
    "Social smoking status (Yes = 1, No = 0) (Integer)",
    "Number of pets owned (Integer)",
    "Weight in kilograms (Integer)",
    "Height in centimeters (Integer)",
    "BMI (Integer)",
    "Target variable representing hours absent (Integer)"
  )
)

# Print the data dictionary as a nice table
kable(data_dict, caption = "Data Dictionary: Absenteeism at Work Dataset")



```


## **Reference**

Martiniano, A. & Ferreira, R. (2012). Absenteeism at work [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X882.




