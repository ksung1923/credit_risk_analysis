# Credit Risk Analysis 

## Overview of Loan Prediction Risk Analysis 
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. With the growth of personal lending, FinTech firms are using the latest machine learning techniques to analyze large amounts of data and predict trends to optimize lending.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Finally, I evaluated the performance of these models and make a written recommendation on whether they should be used to predict credit risk. 


## Results

### Oversampling 

#### Naive Random Oversampling
For Naive Random Oversampling, the balanced accuracy test is 66%. The precision is 1% for "high_risk" and 100% for "low_risk", which indicates an overfitting for the "low_risk". The recall is 74% for "high_risk" and 58% for "low_risk", which are both not ideal. 

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.74      0.58      0.02      0.66      0.44       101
   low_risk       1.00      0.58      0.74      0.73      0.66      0.42     17104

avg / total       0.99      0.58      0.74      0.73      0.66      0.42     17205
```

#### SMOTE Oversampling
For SMOTE Oversampling, the balanced accuracy test is 64%. The precision is 1% for "high_risk" and 100% for "low_risk", which indicates an overfitting for the "low_risk". The recall is 60% for "high_risk" and 67% for "low_risk", which are both not ideal. 

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.60      0.67      0.02      0.64      0.40       101
   low_risk       1.00      0.67      0.60      0.80      0.64      0.41     17104

avg / total       0.99      0.67      0.60      0.80      0.64      0.41     17205
```


### Undersampling 
For Undersampling, the balanced accuracy test is 64%. The precision is 1% for "high_risk" and 100% for "low_risk", which indicates an overfitting for the "low_risk". The recall is 64% for "high_risk" and 53% for "low_risk", which are both not ideal. 

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.64      0.53      0.02      0.58      0.34       101
   low_risk       1.00      0.53      0.64      0.69      0.58      0.34     17104

avg / total       0.99      0.53      0.64      0.69      0.58      0.34     17205
```

### Combination (Over and Under) Sampling
For Combination (Over and Under) Sampling, the balanced accuracy test is 64%. The precision is 1% for "high_risk" and 100% for "low_risk", which indicates an overfitting for the "low_risk". The recall is 70% for "high_risk" and 58% for "low_risk", which are both not ideal. 

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.70      0.58      0.02      0.64      0.41       101
   low_risk       1.00      0.58      0.70      0.73      0.64      0.40     17104

avg / total       0.99      0.58      0.70      0.73      0.64      0.40     17205
```


### Balanced Random Forest Classifier
For Balanced Random Forest Classifier, the balanced accuracy test is 77%. The precision is 3% for "high_risk" and 100% for "low_risk", which indicates an overfitting for the "low_risk". The recall is 66% for "high_risk" and 88% for "low_risk".

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.03      0.66      0.88      0.06      0.77      0.57       101
   low_risk       1.00      0.88      0.66      0.94      0.77      0.60     17104

avg / total       0.99      0.88      0.66      0.93      0.77      0.60     17205
```

### Easy Ensemble AdaBoost Classifier
For Easy Ensemble AdaBoost Classifier, the balanced accuracy test is 93%. The precision is 9% for "high_risk" and 100% for "low_risk", which indicates an overfitting for the "low_risk". The recall is 92% for "high_risk" and 94% for "low_risk".

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
   low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104

avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205
```


## Summary
The four models listed below are not recommended because all the models have a balanced accuracy score of less than 70% and a precision score that appears to be overfit for the credit scorers. 

1. Naive Random Oversampling
2. SMOTE Oversampling
3. Undersampling
4. Combination (Over and Under) Sampling

For the Balanced Random Forest Classifier, we see an improved balanced accuracy score of 77%; however, the precision and sensitivity are still not ideal. My recommendation out of the six models would be the Easy Ensemble AdaBoost Classifier. The model sees a balanced accuracy score of 93% and has improved precision and sensitivity compared to the other models. 
 

