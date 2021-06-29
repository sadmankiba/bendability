# Notes
Changed headers in each dataset to separate `Sequence #` from `Sequence` by tab.

## Results
**NN with DNA Shape**
Hidden Layer    | Dataset |  Shape Feature |   Base Pair   |  Training acc.   |     Test Acc.
 100 (d)             CNL        ProT             all (46)          0.577              0.0
 100 (d)             RL          ProT            all (46)         0.663             -0.360     


**Classification of C0 with CNN on DNA Shape values**


|Library|   C0 Class range |Shape | OHE/normal |     Architecture           |   BP   |  Accuracy |   
|-------| -----------------|------|------------|----------------------------|--------|-----------|
|  CNL  |  (0.25, _, 0.25) | ProT | normal     | (f=64,k=8), mx 2x2, 50, 3  | 1 - 50 |    0.77   |
|  RL   |  (0.2, 0.6, 0.2) | HelT | normal     | (f=32,k=8), mx 2x2, 50, 3  | 1 - 50 |    0.6    |
|  RL   |  (0.2, 0.6, 0.2) | ProT | OHE        | (f=64,k=4), mx 2x2, 100, 3 | 1 - 50 |    0.51   |
|  RL   |  (0.2, 0.6, 0.2) | HelT | OHE        | (f=32,k=4), mx 2x2, 50, 3  | 1 - 50 |    0.6    |
|  CNL  |  (0.2, 0.6, 0.2) | HelT | OHE        | (f=32,k=3), mx 2x2, 50, 3  |11 - 40 |    0.59   |
|  CNL  |  (0.2, 0.6, 0.2) | ProT | OHE        | (f=32,k=4), mx 2x2, 50, 3  |11 - 40 |    0.6    |


**Classification of C0 with classifier on k-mer counts with feature selection by Boruta**

|Library|   C0 Class range |  K-mers | Overlap count|Perc/Iter | Sel. feat. |   BP   |  Accuracy |   
|-------| -----------------|---------|--------------|----------|------------|--------|-----------|
|  CNL  |  (0.25, _ , 0.25)| 2,3,4,5 | True         |  90/50   |  326       | 1 - 50 |  0.6      |
|  CNL  |  (0.25, _ , 0.25)| 2,3,4   | True         |  90/30   |  144       | 1 - 50 |  0.6      |  
|  RL   |  (0.2, 0.6, 0.2) | 2       | False        |  90/40   |  16        | 1 - 50 |  0.77     |
|  RL   |  (0.2, 0.6, 0.2) | 2,3,4,5 | False        |  90/40   |  1360      | 1 - 50 |  0.76     |

**Classification of C0 with classifier on k-mer counts and distance with manual feature selection**

|Library|C0 Class range  |K-mers+dist| Bal.|Sel. feat.|  BP  |Ts a.|Tr a.| Comment   | 
|-------|----------------|-----------|-----| -------- |------|-----|-----|-----------|
|  CNL  |(0.2, _, 0.2)   |2,3,4 + 2  | Yes |  944     | 1-50 |0.57 |0.97 |RF (md=32) |
|  CNL  |(0.2, 0.6, 0.2) |2,3,4 + 2  | Yes |  572     | 1-50 |0.51 |0.98 |RF (md=inf)|
|  CNL  |(0.2, 0.6, 0.2) |2,3,4 + _  | Yes |  106     | 1-50 |0.49 |0.98 |RF (md=inf)|


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
