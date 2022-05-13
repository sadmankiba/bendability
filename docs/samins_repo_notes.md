# `DNABendabilityModels` Repository

## Setup

- Install packages

```sh
pip3 install opencv-python opencv-contrib-python
pip3 install tensorflow_addons
pip3 install tf-slim
```

## Run

### **Running `backprop_contribution5.py`**

Copy a few rows from bendability data to `DNABendabilityModels/data/dataset_9_top.txt`. Then `python3 backprop_contribution5.py`.

### **Running a model**
Add `data/` directory in `src_bendability/`. Add bendability dataset in this dir. 


Test model 

```sh
cd src_bendability/
python3 test_model.py --parameter-file parameters/parameter8.txt --model model35 --model-weights model_weights/model35_parameters_parameter_274 --test-dataset ../data/dataset_9_top.txt
```

## Generating motif logos learned by model

```sh
cd src_bendability/
mkdir logos/
python3 generate_motif_logos.py model_weights/model35_parameters_parameter_274 parameters/parameter8.txt png
```

## Model Performance

**Model 35, parameter 8, weights 274**

| Test lib | Pearson | Spearman | R2   | 
|----------|---------|----------|------|
| RL       |  0.88   |   0.87   | 0.76 |
| ChrVL    | 0.75    | 0.74     | 0.56 |    
| TL       | 0.87    | 0.86     | 0.75 |


## Model Architecture 

**Model 35, parameter 8** 

| Layer (type)            |       Output Shape    |     Param #  |   Connected to   |                  
|-------------------------|-----------------------|--------------|------------------|
 forward (InputLayer)           [(None, 50, 4)]      0           []                               
                                                                                                  
 reverse (InputLayer)           [(None, 50, 4)]      0           []                               
                                                                                                  
 convolution_layer (Convolution  (None, 50, 256)     8448        ['forward[0][0]',                
 Layer)                                                           'reverse[0][0]']                
                                                                                                  
 re_lu (ReLU)                   (None, 50, 256)      0           ['convolution_layer[0][0]']      
                                                                                                  
 re_lu_1 (ReLU)                 (None, 50, 256)      0           ['convolution_layer[1][0]']      
                                                                                                  
 conv1d (Conv1D)                (None, 50, 2)        20482       ['re_lu[0][0]',                  
                                                                  're_lu_1[0][0]']                
                                                                                                  
 maximum (Maximum)              (None, 50, 2)        0           ['conv1d[0][0]',                 
                                                                  'conv1d[1][0]']                 
                                                                                                  
 re_lu_2 (ReLU)                 (None, 50, 2)        0           ['maximum[0][0]']                
                                                                                                  
 conv1d_1 (Conv1D)              (None, 1, 1)         101         ['re_lu_2[0][0]']                
                                                                                                  
 flatten (Flatten)              (None, 1)            0           ['conv1d_1[0][0]']               
                                                                                                  
-------------------------------------------------------------------------------------------
Total params: 29,031
Trainable params: 29,031
Non-trainable params: 0



