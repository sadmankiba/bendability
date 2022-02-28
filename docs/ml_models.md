# ML Models

## Correlation 
Correlation of K-mer counts and hel sep with C0 was measured in different libraries. Max corrs were around 0.2. Helsep pairs were more consistent than dinucs across diff libraries. See this [doc](https://docs.google.com/document/d/1WM0VoBn-2Az33w3PMWKPp7xLdan5hTHv/edit?usp=sharing&ouid=113564166501064431161&rtpof=true&sd=true) 

## C0 Regression 

### K-mer + helsep

#### Research Questions 
- Does linear models really reach r = 0.55 as claimed by Basu et al? 
- How much performance can we achieve with machine learning models?
#### All features

| Train  | Test   | K    | HS | C0x| Model       | #Feat| BP   |Test pear,r2  |Train pear,r2| Cmnt |
| ------ | ------ | ---- | -- | -- | ----------  | ---- | ---- | ------------ | ----------- | ---- |
| TL/10k | RL/10k | 2    | Y  |  1 | Linear      | 152  | 1-50 |-0.02,-5e+27| 0.56,0.31   |      |
| TL/all | RL/all | 2    | Y  |  1 | Linear      | 152  | 1-50 |0.02,-4e+25 | 0.56,0.31   |      |
| TL/all | RL/all | 2    | Y  |  1 | Ridge,a=5   | 152  | 1-50 | 0.4,-13.6  | 0.56,0.31   | |
| TL/all | RL/all | 2    | Y  |  1 | Ridge,a=20  | 152  | 1-50 | 0.4,-13.6  | 0.56,0.31   | | 
| TL/all | RL/all | 2    | Y  |  1 | Ridge,a=100 | 152  | 1-50 | 0.4,-13.6  | 0.56,0.31   | |
| TL/all | RL/all | 2    | Y  |  1 |Ridge,a=1e3  | 152  | 1-50 | 0.4,-13.2  | 0.56,0.31   | best ridge|
| TL/all | RL/all | 2    | Y  |  1 |Ridge,a=1e4  | 152  | 1-50 | 0.4,-10.6  | 0.56,0.3    | |
| TL/all | RL/all | 2    | Y  |  1 |Ridge,a=1e5  | 152  | 1-50 | 0.39,-3.1  | 0.54,0.2    | |
| TL/all | RL/all | 2    | Y  |  1 |Ridge,a=1e6  | 152  | 1-50 | 0.38,0.11  | 0.51,0.06   | |
| TL/all |CHRV/all| 2    | Y  |  1 |Ridge,a=1e3  | 152  | 1-50 | 0.38,-9.4  | 0.56,0.31   | |
| TL/10k |RL/all  | 2    | Y  |  1 | SVR,C=0.1   | 152  | 1-50 | 0.02,-0.04 | 0.69,0.42   | |
| TL/10k |RL/all  | 2    | Y  |  1 | SVR,C=10    | 152  | 1-50 | 0.02,-0.2  | 0.99,0.96   | |
| TL/10k |RL/all  | 2    | Y  |  1 | SVR,C=10    | 152  | 1-50 | 0.02,-0.2  | 0.99,0.96   | |
| TL/all |RL/all  | 2    | Y  |  1 |Lin SVR,C=0.1| 152  | 1-50 | 0.41,-11.5 | 0.55,0.30   | |
| TL/all |RL/all  | 2    | Y  |  1 |Lin SVR,C=0.1,iter=1e4| 152  | 1-50 | 0.41,-11.6 | 0.55,0.3| best SVR+all|
| TL/all |RL/all  | 2    | Y  |  1 |Lin SVR,C=5,iter=1e4| 152  | 1-50 | 0.4,-12.4 | 0.54,0.29| |
| TL/10k |RL/all  | 2    | Y  |  1 |RF_reg,n=25,dep=32| 152  | 1-50 | 0.24,-1.0  | 0.97,0.88   | |
| TL/all |RL/all  | 2    | Y  |  1 |RF_reg,n=25,dep=32| 152  | 1-50 | 0.26,-1.3  | 0.97,0.88   | |
| TL/all |RL/all  | 2    | Y  |  1 |RF_reg,n=50,dep=16,leaf=2| 152  | 1-50 | 0.29,-1.0  | 0.84,0.63 | best RF|
| TL/all |RL/all  | 2    | Y  |  1 |HGB_reg,lr=0.01,dep=32,leaf=5,iter=500| 152  | 1-50 | 0.33,-0.8  | 0.61,0.35   | |
| TL/all |CHRV/all| 2    | Y  |  1 |HGB_reg,lr=0.01,dep=64,l2_reg=1,leaf=1,iter=500| 152  | 1-50 | 0.33,-0.8  | 0.61,0.35   | |
| TL/all |CHRV/all| 2    | Y  |  1 |HGB_reg,lr=0.1,dep=64,l2_reg=1,leaf=1,iter=500| 152  | 1-50 | 0.3,-9.4  | 0.76,0.55   | |

#### With Correlation Feature selector (r > 0.05). 

| Train  | Test   | K     | C0 scale | Model      | Features | BP     | Test r2 | Train r2 |
| ------ | ------ | ----- | -------- | ---------- | -------- | ------ | ------- | -------- |
| TL/50k | RL/10k | 2,3,4 | 20       | SVR C=0.01 | 126      | 1 - 50 | 0.0     | 0.28     |
| TL/50k | RL/10k | 2,3,4 | 20       | SVR C=1    | 126      | 1 - 50 | 0.02    | 0.43     |
| TL/20k | RL/5k  | 2     | 20       | SVR C=0.01 | 48       | 1 - 50 | 0.01    | 0.22     |
| TL/20k | RL/5k  | 2     | 20       | SVR C=1    | 48       | 1 - 50 | 0.05    | 0.36     |

#### Observation
- Maximum corr achieved by linear model with l2 regularization (Ridge), r = 0.4 and Linear SVR, r = 0.41. 
- Linear model (Ridge) gives better prediction (more r) than other models, such as gradient boost, random forest etc. 
- Correlation feature selector has worse performance than all feature selector. Maybe because some features excluded carried non-linear signal for bendability.


### NN with DNA Shape

| Hidden Layer | Dataset | Shape Feature | Base Pair | Training acc. | Test Acc. |
| ------------ | ------- | ------------- | --------- | ------------- | --------- |
| 100 (d)      | CNL     | ProT          | all (46)  | 0.577         | 0.0       |
| 100 (d)      | RL      | ProT          | all (46)  | 0.663         | -0.360    |

## C0 Classification 

#### CNN on DNA Shape values

| Library | C0 Class range   | Shape | OHE/normal | Architecture               | BP      | Accuracy |
| ------- | ---------------- | ----- | ---------- | -------------------------- | ------- | -------- |
| CNL     | (0.25, \_, 0.25) | ProT  | normal     | (f=64,k=8), mx 2x2, 50, 3  | 1 - 50  | 0.77     |
| RL      | (0.2, 0.6, 0.2)  | HelT  | normal     | (f=32,k=8), mx 2x2, 50, 3  | 1 - 50  | 0.6      |
| RL      | (0.2, 0.6, 0.2)  | ProT  | OHE        | (f=64,k=4), mx 2x2, 100, 3 | 1 - 50  | 0.51     |
| RL      | (0.2, 0.6, 0.2)  | HelT  | OHE        | (f=32,k=4), mx 2x2, 50, 3  | 1 - 50  | 0.6      |
| CNL     | (0.2, 0.6, 0.2)  | HelT  | OHE        | (f=32,k=3), mx 2x2, 50, 3  | 11 - 40 | 0.59     |
| CNL     | (0.2, 0.6, 0.2)  | ProT  | OHE        | (f=32,k=4), mx 2x2, 50, 3  | 11 - 40 | 0.6      |

#### k-mer counts with feature selection by Boruta

| Library | C0 Class range    | K-mers  | Overlap count | Perc/Iter | Sel. feat. | BP     | Accuracy |
| ------- | ----------------- | ------- | ------------- | --------- | ---------- | ------ | -------- |
| CNL     | (0.25, \_ , 0.25) | 2,3,4,5 | True          | 90/50     | 326        | 1 - 50 | 0.6      |
| CNL     | (0.25, \_ , 0.25) | 2,3,4   | True          | 90/30     | 144        | 1 - 50 | 0.6      |
| RL      | (0.2, 0.6, 0.2)   | 2       | False         | 90/40     | 16         | 1 - 50 | 0.77     |
| RL      | (0.2, 0.6, 0.2)   | 2,3,4,5 | False         | 90/40     | 1360       | 1 - 50 | 0.76     |

#### k-mer counts and distance with manual feature selection

| Library | C0 Class range  | K-mers+dist | Bal. | Sel. feat. | BP   | Ts a. | Tr a. | Comment     |
| ------- | --------------- | ----------- | ---- | ---------- | ---- | ----- | ----- | ----------- |
| CNL     | (0.2, \_, 0.2)  | 2,3,4 + 2   | Yes  | 944        | 1-50 | 0.57  | 0.97  | RF (md=32)  |
| CNL     | (0.2, 0.6, 0.2) | 2,3,4 + 2   | Yes  | 572        | 1-50 | 0.51  | 0.98  | RF (md=inf) |
| CNL     | (0.2, 0.6, 0.2) | 2,3,4 + \_  | Yes  | 106        | 1-50 | 0.49  | 0.98  | RF (md=inf) |

## SKlearn Models

### Classification

- Logistic Regression
- Support Vector Machine: Kernelized
- Gradient boost
- Neural network / MLPClassifier
- K-nearest neighbour (not good choice)

- Random forest
- Naive Bayes
  - GaussianNB (k-mer + distance)
  - MultinomialNB (distance)
- Support Vector Machine: Linear

### Regression

- Linear regression
- Ridge regression
- Lasso
- Kernelized SVM (SVR)
- Neural network / MLPRegressor

- K-neighbours (not good choice)
- Random forest
- Gradient boost

## Model archi

```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 1, 40, 32)         288
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 1, 20, 32)         0
_________________________________________________________________
flatten (Flatten)            (None, 640)               0
_________________________________________________________________
dense (Dense)                (None, 50)                32050
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 153
=================================================================
Total params: 32,491
Trainable params: 32,491
Non-trainable params: 0
_________________________________________________________________
```
