# Notes 
Changed headers in each dataset to separate `Sequence #` from `Sequence` by tab.

## Results 
**NN with DNA Shape**
Hidden Layer    | Dataset |  Shape Feature |   Base Pair   |  Training acc.   |     Test Acc.
 100 (d)             CNL        ProT             all (46)          0.577              0.0
 100 (d)             RL          ProT            all (46)         0.663             -0.360     


**Classification of C0 with CNN on DNA Shape values** 


|Library|   C0 Class range |  Shape   |      Architecture             |   BP   |  Accuracy |   
|-------| -----------------|----------|-------------------------------|--------|-----------|
|  RL   |  (0.2, 0.6, 0.2) |   ProT   |  (f=64,k=4), mx 2x2, 100, 3   | 1 - 50 |    0.51   |
|  RL   |  (0.2, 0.6, 0.2) |   HelT   | (f=32,k=4), mx 2x2, 50, 3     | 1 - 50 |    0.6    |  
|  CNL  |  (0.2, 0.6, 0.2) |   HelT   | (f=32,k=3), mx 2x2, 50, 3     |11 - 40 |    0.59   |
|  CNL  |  (0.2, 0.6, 0.2) |   ProT   | (f=32,k=4), mx 2x2, 50, 3     |11 - 40 |    0.6    |

Classification of C0 with classifier on k-mer counts with feature selection

|Library|   C0 Class range |  K-mers | # of sel. features |   BP   |  Accuracy |   
|-------| -----------------|---------|--------------------|--------|-----------|
|  RL   |  (0.2, 0.6, 0.2) | 2,3,4,5 |                    | 1 - 50 |       |
|  RL   |  (0.2, 0.6, 0.2) | 2,3,4,5 |                    | 1 - 50 |        |  
|  CNL  |  (0.2, 0.6, 0.2) | 2,3,4,5 |                     |11 - 40 |       |
|  CNL  |  (0.2, 0.6, 0.2) | 2,3,4,5 |                     |11 - 40 |       |
