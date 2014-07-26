# MLSP Kaggle competition
# 2nd place solution
# Author: Alexander V. Lebedev


# I. PREPARATORY STEP:

# Ia. Load the libraries:
library(caret)
library(e1071)
library(kernlab)
library(doMC)
library(foreach)

# Ib. Read the data:
trFC <- read.csv('/YOUR-PATH/Kaggle/SCH/Train/train_FNC.csv')
trSBM <- read.csv('/YOUR-PATH/Kaggle/SCH/Train/train_SBM.csv')
tr <- merge(trFC, trSBM, by='Id')

tstFC <- read.csv('/YOUR-PATH/Kaggle/SCH/Test/test_FNC.csv')
tstSBM <- read.csv('/YOUR-PATH/Kaggle/SCH/Test/test_SBM.csv')
tst <- merge(tstFC, tstSBM, by='Id')

y <- read.csv('/YOUR-PATH/Kaggle/SCH/Train/train_labels.csv')


# II. ANALYSIS

# IIa. "Feature trimming"

# Registering 6 cores to speed up my computations:
registerDoMC(cores=6)

# Just converting a y-label vector into appropriate format:
y <- as.factor(paste('X.', y, sep = ''))

# This step is important (I introduce a random vector into my feature set):
all <- cbind(tr, rnorm(1:dim(tr)[1]))
colnames(all)[412] <- 'rand'

# Now I train Random Forest with this (full) feature set:
rf.mod <- foreach(ntree=rep(2500, 6), .combine=combine, .multicombine=TRUE,
                  .packages='randomForest') %dopar% {
                    randomForest(all[,2:412], y, ntree=ntree)
                  }


# Now, I am looking at the feature importances:
imp <- as.data.frame(rf.mod$importance[order(rf.mod$importance),])

# Everything below importance of my "dummy" feature (random vector) can likely be ignored
imp <- subset(imp, imp>imp['rand',])

# Saving the data in one rda-file for further analyses: 
save('all', 'y', 'tst', 'imp',  file = '/YOUR-PATH/Kaggle/SCH/Train/AllData.rda')


# Now, I reduce my feature set:
dat <- all[,rownames(imp)]


IIb. Training the final model:

# I usually start from SVM and then proceed with ensemble methods
# (in this competition, the use of boosted trees did not result in superior performance and I stopped)
# I would have tried other algorithms and more sophisticated feature selection approaches
# (like SCAD-SVM and recursive feature elimination), but I was not impressed with my intermediate results and gave up =)

# So, here is the model that provided me 2nd position:

# First, I estimate "sigma" (inverse width parameter for the RBF-SVM)
# Of note, sometimes I use a subset of my data, but here I used the whole training set due to its very limited size:
sigDist <- sigest(y ~ as.matrix(dat), data=dat, frac = 1)

# Creating a tune grid for further C-parameter selection):
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-20:100))

# And... training the final RBF-SVM model with leave-one-out cross-validation:
# (Yes. It's as simple as that!)

svmFit <- train(dat,y,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneGrid = svmTuneGrid,
                trControl = trainControl(method = "cv", number = 86, classProbs =  TRUE))


# III. FINAL STEP

# Making predictions
ttst <- tst[,rownames(imp)]
predTst <- predict(svmFit, ttst, type='prob')
predTst <- predTst[,2]

# Formatting submission:
pred <- cbind(as.integer(tst$Id), as.numeric(predTst))
colnames(pred) <- c('Id', 'Probability')

# Writing:
write.table(pred, file = '/YOUR-PATH/Kaggle/SCH/submissions/submission_rbfSVM_RFtrimmed.csv', sep=',', quote=F, row.names=F, fileEncoding = 'UTF-16LE')
