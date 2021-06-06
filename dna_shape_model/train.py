from model import nn_model
import time
import sys
import random
import numpy as np
# import tensorflow as tf
from data_preprocess_shape import preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# get dictionary from text file
def get_parameters(file_name):
    dict = {}
    with open(file_name) as f:
        for line in f:
           (key, val) = line.split()
           dict[key] = val

    # change string values to integer values
    dict["filters"] = int(dict["filters"])
    dict["kernel_size"] = int(dict["kernel_size"])
    dict["epochs"] = int(dict["epochs"])
    dict["batch_size"] = int(dict["batch_size"])

    return dict
    

def main(argv = None):
    if argv is None:
        argv = sys.argv
        #input argszw
        file = argv[1]
        #e.g. data/41586_2020_3052_MOESM4_ESM.txt
        rc_file = argv[2]
        # parameter_file = argv[2]
        #e.g. parameter1.txt
        readout_file = argv[3]
        parameter_file = argv[4]

    ## excute the code
    start_time = time.time()
    #Reproducibility
    seed = random.randint(1,1000)
    np.random.seed(seed)
    # tf.random.set_seed(seed)

    params = get_parameters(parameter_file)
    
    dim_num = (-1, 47, 5)
    nn = nn_model(dim_num=dim_num, filters=params["filters"], kernel_size=params["kernel_size"], pool_type=params["pool_type"], regularizer=params["regularizer"],
            activation_type=params["activation_type"], epochs=params["epochs"], batch_size=params["batch_size"], loss_func=params["loss_func"], optimizer=params["optimizer"])
    model = nn.create_model()

    prep = preprocess(file, rc_file, readout_file)

    data = prep.augment()

    readout = np.asarray(data["readout"])
    fw_fasta = np.asarray(data["forward"])
    rc_fasta = np.asarray(data["reverse"])

    np.set_printoptions(threshold=sys.maxsize)

    
    # 90% Train, 10% Test
    x1_train, x1_test, y1_train, y1_test = train_test_split(fw_fasta, readout, test_size=0.1, random_state=seed)
    x2_train, x2_test, y2_train, y2_test = train_test_split(rc_fasta, readout, test_size=0.1, random_state=seed)

    # change from list to numpy array
    y1_train = np.asarray(y1_train)
    y1_test = np.asarray(y1_test)
    y2_train = np.asarray(y2_train)
    y2_test = np.asarray(y2_test)


    # Without early stopping
    history = model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=params["epochs"], batch_size=params["batch_size"], validation_split=0.1)

    # # Early stopping
    # #callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    # #callback = EarlyStopping(monitor='val_spearman_fn', min_delta=0.0001, patience=3, verbose=0, mode='max', baseline=None, restore_best_weights=False)
    # #history = model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks = [callback])

    history2 = model.evaluate({'forward': x1_test, 'reverse': x2_test}, y1_test)
    pred = model.predict({'forward': x1_test, 'reverse': x2_test})

    print("Seed number is {}".format(seed))
    print('metric values of model.evaluate: '+ str(history2))
    print('metrics names are ' + str(model.metrics_names))

    model.save_weights('model_weights.h5', save_format='h5')

    # # folder = '.'
    # # with open(folder+'/weights', 'w') as f:
    # #     layer = model.layers[2]
    # #     config = layer.get_config()
    # #     print(config, file=f)
    # #     weights = layer.get_weights()
    # #     w = np.array(weights[0])
    # #     w = tf.transpose(w, [2,0,1])
    # #     # w = ConvolutionLayer.call(weights)
    # #     print(w, file=f)
        
    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    sys.exit(main())


