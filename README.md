README
-----------------------------------------

## MLSP 2014 Schizophrenia Classification Challenge: 2nd position (solution)

*Author: Alexander V. Lebedev*

*Date: 26/07/2014*


### 1. Summary
The goal of the competition (https://www.kaggle.com/c/mlsp-2014-mri) was to automatically detect subjects with schizophrenia based on multimodal features derived from the magnetic resonance imaging (MRI) data.
For this challenge, I implemented so-called "feature trimming", consisting of 1) introducing a random vector into the feature set, 2) calculating feature importance, 3) removing the features with importance below the "dummy feature".
At the first step, I ran Random Forest [1] model and performed trimming based on Gini-index [2]. Then, after estimation of the inverse width parameter ("sigma"), I tuned C-parameter for my final model - Support Vector Machine with Gaussian Kernel (RBF-SVM) [3].


### 2. Feature Selection
The key step was "feature trimming". Further steps were also quite simple and none of any sophisticated approaches (like ensembling, hierarchical models) were implemented. Generally, I tried to keep the design as simple as possible due to limited number of subjects available in the training set and therefore being concerned about overfitting.


### 3. Modeling Techniques and Training
Details of the model and training procedures for each technique used in the final model. If models were combined or ensembled, describe that procedure as well. If external data was used, explain how this data was obtained and used.


### 4. Code Description

#### 4.1 Preparatory step:

##### 4.1.1 Load the libraries:

```r
library(caret)
library(randomForest)
library(e1071)
library(kernlab)
library(doMC)
library(foreach)
library(RColorBrewer)
```


##### 4.1.2 Read the data:



```r
# Training set:
trFC <- read.csv('/YOUR-PATH/Kaggle/SCH/Train/train_FNC.csv')
trSBM <- read.csv('/YOUR-PATH/Kaggle/SCH/Train/train_SBM.csv')
tr <- merge(trFC, trSBM, by='Id')

# Test set:
tstFC <- read.csv('/YOUR-PATH/Kaggle/SCH/Test/test_FNC.csv')
tstSBM <- read.csv('/YOUR-PATH/Kaggle/SCH/Test/test_SBM.csv')
tst <- merge(tstFC, tstSBM, by='Id')

y <- read.csv('/YOUR-PATH/Kaggle/SCH/Train/train_labels.csv')
```

#### 4.2 Analysis

##### 4.2.1 "Feature Trimming"

Registering 6 cores to speed up my computations:

```r
registerDoMC(cores=6)
```

Converting a y-label vector into appropriate format:


```r
y <- as.factor(paste('X.', y[,2], sep = ''))
```

Introducing a random vector into my feature set:

```r
all <- cbind(tr, rnorm(1:dim(tr)[1]))
colnames(all)[412] <- 'rand'
```

Now I train Random Forest with this (full) feature set:

```r
rf.mod <- foreach(ntree=rep(2500, 6), .combine=combine, .multicombine=TRUE,
                  .packages='randomForest') %dopar% {
                    randomForest(all[,2:412], y, ntree=ntree)
                  }
```

Looking at the feature importances:

```r
color <- brewer.pal(n = 8, "Dark2")
imp <- as.data.frame(rf.mod$importance[order(rf.mod$importance),])
barplot(t(imp), col=color[1])
points(which(imp==imp['rand',]),0.6, col=color[2], type='h', lwd=2)
```

![plot of chunk simpleplot](https://cloud.githubusercontent.com/assets/4508892/3711386/e5b4496c-14d3-11e4-9c1d-5a94987dc4ac.png) 

Everything below importance of our "dummy" feature (random vector) can likely be ignored.
So, we "cut" everything that is on the left side of the orange line.


```r
imp <- subset(imp, imp>imp['rand',])
```


Saving the data in one rda-file for further analyses:

```r
save('all', 'y', 'tst', 'imp',  file = '/YOUR-PATH/Kaggle/SCH/Train/AllData.rda')
```

Now I reduce my feature set:

```r
dat <- all[,rownames(imp)]
```

##### 4.2.2 Final Model
I usually start from SVM and then proceed with ensemble methods. However, in this competition, the use of boosted trees did not result in superior performance and I stopped.

First, I estimate "sigma" (inverse width parameter for the RBF-SVM)
(of note, sometimes I use a subset of my data, but here I used the whole training set due to its very limited size)

```r
sigDist <- sigest(y ~ as.matrix(dat), data=dat, frac = 1)
```

Creating a tune grid for further C-parameter selection):

```r
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-20:100))
```

```
## Warning: row names were found from a short variable and have been
## discarded
```

And training the final RBF-SVM model:

```r
svmFit <- train(dat,y,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneGrid = svmTuneGrid,
                trControl = trainControl(method = "cv", number = 86, classProbs =  TRUE))
```


Making predictions:

```r
ttst <- tst[,rownames(imp)]
predTst <- predict(svmFit, ttst, type='prob')
predTst <- predTst[,2]
```

Formatting submission:

```r
pred <- cbind(as.integer(tst$Id), as.numeric(predTst))
colnames(pred) <- c('Id', 'Probability')
```

Writing:

```r
write.table(pred, file = '/YOUR-PATH/Kaggle/SCH/submissions/submission_rbfSVM_RFtrimmed.csv', sep=',', quote=F, row.names=F, fileEncoding = 'UTF-16LE')
```


### 5. Dependencies
To execute the code the following libraries must be installed: caret [3], randomForest [4], e1071 [5], kernlab [6], doMC [7], foreach [8], RColorBrewer [9]

### 6. Additional Comments and Observations
In general, it was somewhat difficult to evaluate performance of the models, since there was a substantial mismatch between cross-validated accuracies and the feedback that I was receiving during my submissions. It was one of the reasons why I decided not to go further with feature selection and more complex modeling approaches.

### 7. References

[1] V.N. Vapnik (1995) The Nature of Statistical Learning Theory. Springer-Verlag New York, Inc. New York, NY, USA;

[2] L. Breiman (2001) Random Forests. Machine Learning Volume 45, Number 1: 5-32;

[3] M. Kuhn. Contributions from Jed Wing SW, Andre Williams, Chris Keefer and Allan Engelhardt (2012) caret: Classification and Regression Training. R package version 5.15-023. http://cran.r-project.org/packages/caret/;

[4] L. Breiman, A. Cutler, R port by Andy Liaw and Matthew Wiener (2014). randomForest: Breiman and Cutler's random forests for classification and regression. http://cran.r-project.org/web/packages/randomForest/;

[5] D. Meyer, E. Dimitriadou, K. Hornik, A. Weingessel, F. Leisch, C-C. Chang. C-C. Lin. Misc Functions of the Department of Statistics (e1071), TU Wien. http://cran.r-project.org/web/packages/e1071/;

[6] A. Karatzoglou, A. Smola, K. Hornik (2013). kernlab: Kernel-based Machine Learning Lab. http://cran.r-project.org/web/packages/kernlab/;

[7] Revolution Analytics. doMC (2014): Foreach parallel adaptor for the multicore package. http://cran.r-project.org/web/packages/doMC/;

[8] Revolution Analytics, Steve Weston (2014). foreach: Foreach looping construct for R. http://cran.r-project.org/web/packages/foreach/;

[9] Erich Neuwirth (2011). RColorBrewer: ColorBrewer palettes. http://cran.r-project.org/web/packages/RColorBrewer/.
