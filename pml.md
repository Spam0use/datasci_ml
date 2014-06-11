Weight lifting technique classification
========================================================

```
## Loading required package: cluster
## Loading required package: foreach
## Loading required package: lattice
## Loading required package: plyr
## Loading required package: reshape2
```

```
## Warning: package 'doParallel' was built under R version 3.0.3
```

```
## Loading required package: iterators
## Loading required package: parallel
```

## Data acquisition
Load data from csv file, elimate row numbers and timestamps, confirm features are the same

```r
dattr=read.csv('pml-training.csv')[,-c(1,3:5)]  #eliminate row numbers & timestamps
dattst=read.csv('pml-testing.csv')[,-c(1,3:5)] 
all.equal(colnames(dattst)[-ncol(dattst)],colnames(dattr)[-ncol(dattr)])
```

```
## [1] TRUE
```

set aside validation set:


```r
library(caret)
set.seed(123)
tridx=createDataPartition(dattr$classe,p=.8,list=FALSE)
tr=dattr[tridx,]
val=dattr[-tridx,]
```


eliminate near zero variance predictors:


```r
nzv=nearZeroVar(tr)
tr=tr[,-nzv]
val=val[,-nzv]
dattst=dattst[,-nzv]
```

eliminate highly correlated predictors:


```r
numcols=which(sapply(tr,is.numeric))  #numeric columns
corcols=findCorrelation(cor(tr[,numcols],use='pairwise'),cutoff=.95)  #columns with correlation > .95 with another column
rmcor=numcols[corcols]  #indices of columns to remove
tr=tr[,-rmcor]
val=val[,-rmcor]
dattst=dattst[,-rmcor]
numcols=which(sapply(tr,is.numeric))  #numeric columns 
```

At this point, 85 predictors remain, but 34 of them contain missing values (and 98% of rows contain a missing value).  Is it reasonable to impute these missing values?  Try building models with imputation and with exclusion of columns containing missing values & compare performance on validation set.


```r
trc=trainControl(method='cv',number=5,selectionFunction='oneSE')  #5-fold xval, bias toward 'smaller' models
nona=apply(is.na(tr),2,any)
trnn=tr[,!nona]
valnn=tr[,!nona]
rfnn=train(trnn[,-ncol(trnn)],trnn$classe,'rf',tuneGrid=data.frame(.mtry=c(6,12,24)),trControl=trc,ntree=200) #use somewhat slimmed down training to speed things up
```

```r
pp=preProcess(tr[,numcols],method='knnImpute')
trimp=tr
trimp[,numcols]=predict(pp,tr[,numcols])
valimp=val
valimp[,numcols]=predict(pp,val[,numcols])
rfimp=train(trimp[,-ncol(trimp)],trimp$classe,'rf',tuneGrid=data.frame(.mtry=c(10,20,30)),trainControl=trc,ntree=200)
```


```r
confusionMatrix(predict(rfimp,valimp[,-ncol(valimp)]),valimp[,ncol(valimp)])
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
## Loading required package: class
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    1    0    1
##          B    0  757    2    0    0
##          C    0    1  681    3    0
##          D    0    0    0  639    0
##          E    0    0    0    1  720
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.995, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.996    0.994    0.999
## Specificity             0.999    0.999    0.999    1.000    1.000
## Pos Pred Value          0.997    0.997    0.994    1.000    0.999
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.285    0.193    0.175    0.163    0.184
```

```r
confusionMatrix(predict(rfnn,valnn[,-ncol(valnn)]),valnn[,ncol(valnn)])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
```

Validation set error is lower for nonimputed data (i.e. using only features with nonmissing data).  Therefore use nonimputed model for test set predictions.  

## Final model
Retrain no-missing-value model using all training data with appropriately adjusted training parameters:


```r
trc=trainControl(method='cv',number=10,selectionFunction='oneSE')  #10-fold xval, bias toward 'smaller' models
dattrnn=rbind(trnn,valnn)
rfnnall=train(dattrnn[,-ncol(dattrnn)],dattrnn$classe,'rf',tuneGrid=data.frame(.mtry=c(3,6,12)),trControl=trc)
```

predictions for test set:

```r
dattstnn=dattst[,colnames(dattrnn)[-ncol(dattrnn)]]
answers=predict(rfnnall,dattstnn)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
pml_write_files = function(x){  #output function
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(as.character(answers))
```


