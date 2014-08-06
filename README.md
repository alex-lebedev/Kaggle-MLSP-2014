README
-----------------------------------------

## MLSP 2014 Schizophrenia Classification Challenge: 3rd position (solution)

*Author: Karolis Koncevicius*

*Email: Karolis Koncevicius*

*Location: Vilnius University, Vilnius, Lithuania*

*Date: 05/08/2014*


### 1. Summary

The goal of the competition (https://www.kaggle.com/c/mlsp-2014-mri) was to automatically detect subjects with schizophrenia based on multimodal features derived from the magnetic resonance imaging (MRI) data.
The data used for this task had a peculiar property of having more features than available samples. Such situations are often called "High Dimensional Small Sample Size Data" (HDLSS) [1] in the literature and often
present a lot of challenges in both model selection and error estimation [2]. To overcome these difficulties I utilized the Distance Weighted Discrimination (DWD) [3] method which was designed to deal with
HDLSS settings. My winning entry was very simple: I used all of the available features, ran ten fold cross-validation to determine the penalty parameter and then fitted the DWD model.


### 2. Feature Selection

3rd place winning entry used no feature selection.

I also tried implementing a number of unsupervised feature selection methods: removing features with low variance, doing Principal Component Analysis (PCA), removing highly correlated features; But these approaches
gave me lower private and public scores.

Most likely explanation is that the main focus of HDLSS scenarios should be good generalization and feature selection is one possible source of overfitting [4].

### 3. Modeling Techniques and Training

#### 3.1 Distance Weighted Discrimination.

Distance weighted discrimination was used as single base clasifier. Below is a short and naive introduction of this method. For a more complete and formal description the reader is refered to [3]

HDLSSS data occupy only a subspace of the whole feature space. The result of this is the existance of linear projection vectors that have the so called "data-pilling" property [5].
This means that it is possible to project all the samples of both classes onto a two distinct points: one for each class. These kind of projection will surely classify the dataset at hand perfectly.
But we should not expect this result to generalize well since the particular constalation of sample points in the feature space can be determined by the generalization of our sample. Yet the majority
of learning methods (even the simple linear ones) are affected by this phenomenon since the functions they try to optimize often reaches maximum when the separating hyperplane finds such a projection.
(As an example consider Fishers linear discriminant criterion, which tries to maximize between-class variance and minimize within-class variance.).

One natural choice here can be the usage of support vector machines [6]. But even SVM's can be affected by partial data-piling since they base the parameters of the hyperplane on a few "support vector" samples.

The DWD approach is to take all of the distances from samples to the separating hyperplane into account instead of maximizing the margin between farthest samples (support vectors). Simple way of allowing
these distances to in fluence the separation boundary is to minimize the sum of the inverse distances. This gives high significance to those points that are close to the hyperplane, with little impact
from points that are farther away.

$$ \mu_1 = (\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}, ..., \frac{1}{\sqrt{n}}) $$

In this case the solution that has more points farther away from the support vectros and separating hyperplane is preferable over the same margin-size solution with data points pilled close to support vectors.

#### 3.2 Implementation.

The DWD method uses a "second-order cone programming" (SOCP) optimization. It has implementation in matlab and R. Here I used R implementation [7].

The implementation had one parameter C - the penalty cost associated with missclassification (similar to the one in SVM). Authors stated that they don't have clear guidelines for selection of this parameter
and that it is a possible object of further studies.

In my entry I ussed cross-validation to determine the value of this parameter. I selected several values (ranging from 1 to 1000) and ran 100 itterations of 10-fold cross validation for each. Then it became
clear that lower values had lower classification accuracies and the roc-area reached it's maximum at about C=300. After that point it saturated and remained unchained up to a 1000. Therefore 300 was selected
for the submission.

### 4. Code Description

All the code needed to generate the solution is assembled into one R-script file. It was devided into 6 clearly-defined sections:

1. Sources: This loads all the libraries and sets the main directory path.
2. Load Data: This loads the .csv data files from "input" directory (data that was made available to download from Kaggle.com website).
3. Prep Data: This prepares the data by comining both sets of features and writing labels into a single table. (Train and Test subsets separately).
4. Cross-Validation: This step was used to select the penalty cost parameter. This is the most time-consuming step and can be skipped when running the code.
5. Fit Model: Fits the DWD model.
6. Write Scores: Writes the scores into the "output" directory.

### 5. Dependencies
To succesfully run the code the following R libraries are required: "DWD", "verification".

### 6. How To Generate The Solution
I compiled all the code into a single R-script file. The first line of the script sets the working directory. Further it is assumed that there are 2 subdirectories within: "input" and "output".
"input" directory had all the input files that were available to download for this competition from the Kaggle website. "output" directory can be left emtpy since it was only used to write
the submission file.

The most lenghty step is cross-validation. This step can be skipped to save time.


### 7. Additional Comments and Observations

Arguably the hardest part in this competition was not overfitting. With small amount of available samples it becomes hard to track the true error of misclassification. Cross validation may be
non-reliable [8] especially if used multiple times on the same data with different models. If no a-priori information is available then simple and highly regularized models become the method of choice [9].

In the competition I tried following this philosophy of simple and regularized methods. Among the submited models were: "smartly" regularized linear discriminant analysis (LDA), distance weighted discrimination
(DWD) and some variants of non-supervised feature selection followed by diagonal LDA. My goal in this competition was to compare the regularized LDA with DWD (with the hope that my LDA variant can outperform
DWD) but in the end it could not beat out-of-the box DWD neighter in my internal cross-validation results nor on the leaderboard.

Data subset made available for training had an unequal number of cases and controls. If the final test subset also had this propery then one potential improvement could be taking the class-imbalance into account.

### 8. References

[1] Peter Hall, J. S. Marron, Amnon Neeman. Geometric representation of high dimension, low sample size data. Journal of the Royal Statistical Society: Series B (Statistical Methodology) Volume 67, Issue 3, pages 427-444, June 2005.

[2] S. Raudys, Anil K. Jain. Small Sample Size Effects in Statistical Pattern Recognition: Recommendations for Practitioners. IEEE Transactions on pattern analysis and machine intelligence Vol. 13, No. 3, Match 1991.

[3] J. S. Marron, Michael Todd, Jeongyoun Ahn. Distance Weighted Discrimination. Journal of the American Statistical Association; Volume 102, Issue 480, 2007.

[4] S. Raudys. Feature Over-Selection. Structural, Syntactic, and Statistical Pattern Recognition Lecture Notes in Computer Science Volume 4109, 2006, pp. 622-631.

[5] Jeongyoun Ahn, J. S. Marron. The maximal data piling direction for discrimination. Biometrika Volume 97, Issue 1 Pp. 254-259. 

[6] V.N. Vapnik. The Nature of Statistical Learning Theory. Springer-Verlag New York, New York, USA, 1995.

[7] Hanwen Huang, Xiaosun Lu, Yufeng Liu, Perry Haaland, J.S. Marron. R/DWD: distance-weighted discrimination for classification, visualization and batch adjustment. Bioinformatics. Apr 15, 2012; 28(8): 1182-1183.

[8] UM Braga-Neto, ER Dougherty. Is cross-validation valid for small-sample microarray classification? Bioinformatics, Volume 20, Issue 3, Pp. 374-380.

[9] Trevor Hastie, Robert Tibshirani, Jerome Friedman. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Second Edition. February 2009. Springer-Verlag.




