#Adaptation of both.py with redundant functions removed, for the purpose of retraining previously trained models
#GPL-3.0 licensed
#As this file is highly similar to the both.py file, few comments have been added
#Uncertainties regarding uncommented code, should be answered by the comments in both.py

import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
from layers import Erosion2D, Dilation2D, SpatialMorph, SpectralMorph

import cnn2d
import mcnn

import gc
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import load_model

from tensorflow.keras.utils import to_categorical as keras_to_categorical
import numpy as np
import re

from tensorflow.keras.callbacks import ModelCheckpoint as tf_ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy

from datetime import datetime
from zoneinfo import ZoneInfo
import os


def set_params(args):
    args.batch_size = 1; args.epochs = 5
    return args

def main():
    print("\n\n\n\n")
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--spatialsize', default=11, type=int, help='windows size')
    parser.add_argument('--wdecay', default=0.02, type=float, help='apply penalties on layer parameters')
    parser.add_argument('--preprocess', default="standard", type=str, help='Preprocessing')
    parser.add_argument('--random_state', default=None, type=int, 
                    help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--tr_percent', default=0.03, type=float, help='samples of train set')                      #Percentage of training for the pre-trained model

    parser.add_argument('--re_tr_percent', default=[0.03], type=float, nargs='+')                                   #Percentage of the new set, that will be used for retraining

    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')
    parser.add_argument('--verbosetrain', action='store_true', help='Verbose train')

    parser.add_argument('--models', default = ["CNN"], type = str, nargs='+')                                       #Determines the pre-trained model to be loaded and retrained
    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--batch_size', default=1, type=int, help='Number of training examples in one forward/backward pass.')  #limited batch size and epochs to 1 setting
    parser.add_argument('--epochs', default=5, type=int, help='Number of full training cycle on the training set')
    #########################################

    args = parser.parse_args()
    
    pixels2, labels2, num_class2 = \
    mydata.loadData("UP", num_components=args.components, preprocessing=args.preprocess)                            #Hardcoded to exclusively retrain on the UP set
    
    pixels2, labels2 = mydata.createImageCubes(pixels2, labels2, windowSize=args.spatialsize, removeZeroLabels = True)
    pixels2 = np.pad(pixels2, ((0,0), (0,0), (0,0), (0,200-103)), mode = 'constant')                                #The cubes are padded with the bands that the IP set has more of compared to UP
    
    for model in args.models:
        for tr_percent in args.re_tr_percent:
            
            timeStart = datetime.now(ZoneInfo("Europe/Amsterdam"))
            stringTimeStart = timeStart.strftime("%Y-%m-%d")
            training_size = tr_percent*100
            retrain = "_retrain_UP"

            os.makedirs(f"results{stringTimeStart}", exist_ok = True)
            os.makedirs(f"results{stringTimeStart}/retrain", exist_ok = True)                                       #Make a filename stating that it is retrained, besides the relevant stats
            of = open(f"results{stringTimeStart}/retrain/{model}_{"IP"}_{args.tr_percent*100}{retrain}_runs={args.repeat}_training_size={training_size}_{timeStart.strftime('%H-%M-%S')}.txt", "w")
            
            print(f"\nStart time = {timeStart.strftime('%H:%M:%S')}", file = of, flush = True)
            print(f"Runs = {args.repeat}, epochs = {200} \n", file = of, flush = True)

            stats = np.ones((args.repeat, 9+3)) * -1000.0 # OA, AA, K, Aclass

            for pos in range(args.repeat):
                print(f"training run {pos+1}", file = of, flush = True)
                print(f"training run {pos+1}")
                
                rstate = args.random_state+pos if args.random_state != None else None

                x_train2, x_test2, y_train2, y_test2 = \
                        mydata.split_data(pixels2, labels2, tr_percent, "sklearn", rand_state=rstate)               #Hardcoded to use the sklearn splitmethod
                x_val2, x_test2, y_val2, y_test2 = \
                    mydata.split_data(x_test2, y_test2, args.val_percent, rand_state=rstate)

                for ix in range(num_class2):
                    print(f"label {ix}, train: {np.count_nonzero(y_train2 == ix)}, test: {np.count_nonzero(y_test2 == ix)}", file = of, flush = True)
                inputshape = x_train2.shape[1:]
                if model =="CNN":                                                           #Load the pre-trained model, based on the model provided, and args.tr_percentage, being the percentage it was pre-trained on 
                    clf = load_model(f"/saved_models/CNN/{args.tr_percent}/{pos}.keras")    #Additionally the pos, meaning the current run, is used, resulting in different models being used for each of the testing runs
                else:                                                                       #This method does result in the same pre-trained models being used, between all experiments, though within each experiment they differ
                    clf = load_model(                                                      
                                f"/saved_models/MCNN/{args.tr_percent}/{pos}.keras",        #As the result of custom layers, it needs a custom loader
                                    custom_objects={
                                        "Erosion2D": Erosion2D,
                                        "Dilation2D": Dilation2D,
                                        "SpatialMorph": SpatialMorph,
                                        "SpectralMorph": SpectralMorph,
                                    }
                                    )
                if model =="CNN":
                    clf = cnn2d.get_model_compiled(inputshape, num_class2, w_decay=args.wdecay)
                elif model == "MCNN":
                    clf = mcnn.get_compiled_model(inputshape, num_class2)
                else:
                    print("Error: model unknown")
                    return
                
                valdata2 = (x_val2, keras_to_categorical(y_val2, num_class2)) if args.use_val else (x_test2, keras_to_categorical(y_test2, num_class2))
                

                x = clf.layers[-2].output                                                                           #The output layer differs based on number of classes, which changes between IP and UP set 
                new_output = tf.keras.layers.Dense(num_class2, activation = 'softmax', name = "retrain_output")(x)
                clf2 = tf.keras.Model(inputs = clf.input, outputs = new_output)                                     #New output layer is created, and appended on the pre-trained model, after which it is compiled
                del clf
                if model == "CNN":
                    clf2.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'], jit_compile = True)
                else:
                    clf2.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1), optimizer='adam', metrics=['accuracy'], jit_compile = False)


                clf2.fit(x_train2, keras_to_categorical(y_train2, num_class2),
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                verbose=args.verbosetrain,
                                validation_data=valdata2,
                                callbacks = [tf_ModelCheckpoint("/tmp/best_model.keras", monitor='val_accuracy', verbose=0, save_best_only=True)])
                del clf2; K.clear_session(); gc.collect()
                
                if model =="CNN":
                    clf2 = load_model("/tmp/best_model.keras")
                else:
                    clf2 = load_model(
                                "/tmp/best_model.keras",
                                    custom_objects={
                                        "Erosion2D": Erosion2D,
                                        "Dilation2D": Dilation2D,
                                        "SpatialMorph": SpatialMorph,
                                        "SpectralMorph": SpectralMorph,
                                    }
                                    )
                results = mymetrics.reports(np.argmax(clf2.predict(x_test2), axis=1), y_test2)[2]
                stats[pos,:] = results
                for idx in range(num_class2):                                                                       #Once again, the results after retraining are added to the file    
                    print(f"Label {idx+1}, accuracy:{results[3+idx]}", file = of, flush = True)
                print(f"OA: {results[0]}", file = of, flush = True)
                print(f"AA: {results[1]}", file = of, flush = True)
                print(f"K: {results[2]}\n", file = of, flush = True)
                del clf2, x_train2, x_test2, y_train2, y_test2, x_val2, y_val2
            
            timeEnd = datetime.now(ZoneInfo("Europe/Amsterdam"))
                                                                                                                    #Final time, duration, mean stats and deviations are added to the file
            print(f"\nFinal time = {timeEnd.strftime('%H:%M:%S')}, duration = {timeEnd - timeStart}", file = of, flush = True)
            print(f"Average OA = {np.mean(stats[:, 0])}, std = {np.std(stats[:, 0])}", file = of, flush = True)
            print(f"Average AA = {np.mean(stats[:, 1])}, std = {np.std(stats[:, 1])}", file = of, flush = True)
            print(f"Average K = {np.mean(stats[:, 2])}, std = {np.std(stats[:, 2])}", file = of, flush = True)
            of.close()


if __name__ == '__main__':
    main()
