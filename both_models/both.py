#Initial code provided by https://github.com/mhaut/hyperspectral_deeplearning_review
#GPL-3.0 licensed

#This file contains the main function, and functionality for the experiments and training runs

#Edited for our purposes, adding functionality for the MorphConvHyperNet from https://github.com/max-kuk/MorphConvHyperNet
#Additionaly, added functionality for the tests we required
#Most edits are paired with comments, explaining the changes


import argparse
import auxil.mydata as mydata
import auxil.mymetrics as mymetrics
from layers import Erosion2D, Dilation2D, SpatialMorph, SpectralMorph

import cnn2d
import mcnn

import gc
import keras.backend as K
from tensorflow.keras.models import load_model

from tensorflow.keras.utils import to_categorical as keras_to_categorical
import numpy as np
import re

from tensorflow.keras.callbacks import ModelCheckpoint as tf_ModelCheckpoint

from datetime import datetime
from zoneinfo import ZoneInfo
import os
from itertools import product

#import matplotlib.pyplot as plt                                    #Uncomment this line if you intend to produce a residual map, it does require a pip install matplotlib


def set_params(args):
    args.batch_size = 1; args.epochs = 5
    return args


def continue_existing_file(stats, path, class_num = 16):            #Functionality to continue from files that were unfinished, or that need to have their runs increased
    loaded_run = []
    insights = []
    of = open(path, "r")
    labelCount=0
    resultCount=0
    index = 0
    for line in of:
        line = line.strip()
        match = re.search(r".*? accuracy:(\d+(?:\.\d+)?)", line)    #Search in each line whether it contains "accuracy:" and save the number afterwards
        if match:
            loaded_run.append(float(match.group(1)))                #Append if there was a number
            labelCount+=1
        elif line.startswith(("OA", "AA", "K")):                    #If the AA, OA or K are found, append these
            insights.append(float(line.split(":")[1]))
            resultCount+=1
        if labelCount==class_num and resultCount == 3:              #If the results of one full run have been reached, add them to stats
            stats[index, :] = (insights + loaded_run)
            insights = []
            loaded_run = []
            labelCount = 0
            resultCount = 0
            index+=1
        if labelCount > class_num or resultCount > 3:               #If the file was not formed properly, and the labels or results exceed their respective limits, print error
            print(f"Error, labels {labelCount} or results {resultCount} incorrect\n\n\n\n")

    of.close()
    return stats, index

def add_gaussian_noise(patch, shape, st_dev=0.5):                   #Add gaussian noise to a patch, with provided standard deviation
    return patch+np.random.normal(loc=0, scale = st_dev, size = shape)

def add_salt_pepper_noise_3d(patch, img_size, noise_min, noise_max, percentage = 0.1):  #Add spectrally uncorrelated salt and pepper noise
    noiseNum=int(img_size*percentage)                                                   #Img size provided is height*width*bands, img size * noise percentage = number of noisy pixels
    random_indices = np.random.choice(img_size, noiseNum, replace = False)              #For the number of noisy pixels, random indices are created
    patch.flat[random_indices] = np.random.choice([noise_min, noise_max], noiseNum)     #The values at random indices are changed into min or max
    return patch

def add_salt_pepper_noise_2d(patch, H, W, D, noise_min, noise_max, percentage = 0.1):   #In case of 2D noise, spectrally correlated noise is added
    img_size = H*W                                                                      #Size is equal to height*width
    noiseNum=int(img_size*percentage)                                                   #Size is multiplied by noise percentage to create noisy pixels, which are carried through all bands
    random_indices = np.random.choice(img_size, noiseNum, replace = False)
    flat_patch = patch.reshape(-1, D)
    flat_patch[random_indices, :] = np.random.choice([noise_min, noise_max], noiseNum).reshape((-1,1))
    return flat_patch.reshape(H,W,D)


def main():
    print("\n\n\n\n")
    parser = argparse.ArgumentParser(description='Algorithms traditional ML')
    parser.add_argument('--dataset', type=str, required=True, \
            choices=["IP", "UP", "SV", "UH", "DIP", "DUP", "DIPr", "DUPr"], \
            help='dataset (options: IP, UP, SV, UH, DIP, DUP, DIPr, DUPr)')
    parser.add_argument('--repeat', default=1, type=int, help='Number of runs')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--spatialsize', default=11, type=int, help='windows size')
    parser.add_argument('--wdecay', default=0.02, type=float, help='apply penalties on layer parameters')
    parser.add_argument('--preprocess', default="standard", type=str, help='Preprocessing')
    parser.add_argument('--splitmethod', default="sklearn", type=str, help='Method for split datasets')     #For tr_pixelmin above 0, custom2 needs to be used
    parser.add_argument('--random_state', default=None, type=int, 
                    help='The seed of the pseudo random number generator to use when shuffling the data')
    parser.add_argument('--tr_percent', default=[0.03], type=float, nargs='+', help='samples of train set') #tr_percent allows for a list of training percentages to be provided
    parser.add_argument('--tr_pixelmin', default = 0, type=int)                                             #In combination with splitmethod custom2, a minimum number of pixels can be set
                                                                                                            #If pixelmin was not set, and method is not sklearn, tr_percent will be used as the set number of pixels
    parser.add_argument('--retrain', action='store_true')                                                   #In case you wish to prepare trained models to use for retraining

    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')
    parser.add_argument('--verbosetrain', action='store_true', help='Verbose train')
    parser.add_argument('--continue_file', action='store_true')                                             #If you want to continue from an old file of a crashed session, or that needs to have more runs
    parser.add_argument('--file_path', default = "", type=str )                                             #Provide the file path to the to be continued file
    parser.add_argument('--gauss', action='store_true')                                                     #In case Gauss noise tests need to be performed
    parser.add_argument('--gauss_dev', default = 0.5, type = float)                                         #Optionally set the deviation of the Gauss noise
    parser.add_argument('--salt_pepper', action='store_true')                                               #When tests need to be run with salt and pepper noise
    parser.add_argument('--sp_percent', default = [0.1], type = float, nargs='+')                           #A list of salt and pepper percentages can be provided
    parser.add_argument('--sp_3d', action='store_true')                                                     #When the 3d type (spectrally uncorrelated) salt and pepper needs to be tested, 
    parser.add_argument('--models', default = ["CNN"], type = str, nargs='+')                               #Set whether MCNN, CNN or both need to be tested
    parser.add_argument('--residual_map', action = 'store_true')                                            #Set the residual map to be created for the last run of each model

    
    #########################################
    parser.add_argument('--set_parameters', action='store_false', help='Set some optimal parameters')
    ############## CHANGE PARAMS ############
    parser.add_argument('--batch_size', default=[1], type=int, nargs='+', help='Number of training examples in one forward/backward pass.')
    parser.add_argument('--epochs', default=[5], type=int, nargs='+', help='Number of full training cycle on the training set')
    #########################################

    args = parser.parse_args()

    if args.set_parameters: args = set_params(args)

    if args.splitmethod =="sklearn" and args.tr_pixelmin>0:
        print("For a tr_pixelmin above 0, splitmethod custom2 has to be used.")
        return 0
    
    rawpixels, rawlabels, num_class = \
                    mydata.loadData(args.dataset, num_components=args.components, preprocessing=args.preprocess)
    print(f"numclasses = {num_class}")
    print(f"unique labels ={np.unique(rawlabels)}")                                                            #Shows the number of unique labels and number of classes
    print("Pixels shape:", rawpixels.shape)                                                                    #Shows whether the IP set has the proper 200 number of bands, or 220 which should be edited by user 
    print("Ground truth shape:", rawlabels.shape)
    pixels, labels = mydata.createImageCubes(rawpixels, rawlabels, windowSize=args.spatialsize, removeZeroLabels = True)
    
    combinations = list(product(args.tr_percent, args.epochs, args.batch_size))                             #List of combinations of training percentages, epochs and batch sizes is created, preventing a triple nested loop
    
    shape = (pixels[0].shape)
    extra_tag = ""
    if args.gauss:                                                                                          #If gauss noise is added, edit pixel cubes to include gauss noise. 
        extra_tag = "_gauss"                                                                                #Add gauss label, for file name later
        pixels = np.array(list(map(lambda patch: add_gaussian_noise(patch, shape, args.gauss_dev), pixels)))

    if args.salt_pepper:                                                                                    #If salt and pepper noise is added, get the shape of the first patch and pass to respective function
        sp_pixels = []
        H = shape[0]
        W = shape[1]
        D = shape[2]
        img_size= shape[0]*shape[1]*shape[2]
        noise_min = int(np.min(pixels)) - 2
        noise_max = int(np.max(pixels)) + 2
        extra_tag = "_salt_pepper"
        for index, sp_ratio in enumerate(args.sp_percent):                                                  #Store list of all pixels with differing salt and pepper percentages, not practical for limited RAM, or many percentages
            if args.sp_3d:
                sp_pixels.append(np.array(list(map(lambda patch: add_salt_pepper_noise_3d(patch, img_size, noise_min, noise_max, sp_ratio), pixels))))
            else:
                sp_pixels.append(np.array(list(map(lambda patch: add_salt_pepper_noise_2d(patch, H, W, D, noise_min, noise_max, sp_ratio), pixels))))
        combinations = [(sp, tr, ep, bs) for sp, (tr, ep, bs) in list(product(range(len(args.sp_percent)), combinations))]      #Due to a list of salt and pepper percentages being possible, add these to combinations

    print(f"combinations = {combinations}")                                                                 #Show the number of combinations, to allow the user to see the number of experiments
    pixel_save = pixels

    for model in args.models:
        for combo in combinations:
            if args.salt_pepper:                                                                            #Unpack the combinations, depending on whether salt and pepper was added
                sp_index, tr_percent, epochs, batch_size = combo 
                pixels = sp_pixels[sp_index]
            else:
                tr_percent, epochs, batch_size = combo

            timeStart = datetime.now(ZoneInfo("Europe/Amsterdam"))
            stringTimeStart = timeStart.strftime("%Y-%m-%d")
            os.makedirs(f"results{stringTimeStart}", exist_ok = True)                                       #Create a folder based on current date, for result storage
            if args.splitmethod == "sklearn" or tr_percent < 1:                                             #If tr_percent is >=1, then it will be used as the set number of pixels
                training_size = tr_percent*100
            else:
                training_size = int(tr_percent)
            
            if args.tr_pixelmin==0:
                min_amount = ""
            else:
                min_amount = f"_min={args.tr_pixelmin}"
            retrain = ""
            if args.retrain:
                retrain = "_for_retrain_UP"                                 #This previous section is to provide tags in the file name, allowing one to see from the file name what experiment was performed
            of = open(f"results{stringTimeStart}/{model}_{args.dataset}{retrain}_runs={args.repeat}_training_size={training_size}{min_amount}_{timeStart.strftime('%H-%M-%S')}_epochs={epochs}_batchSize={batch_size}{extra_tag}.txt", "w")
                                                                            #File name is quite long, but allows for viewing the exact parameters of an experiment, without having to open the file
            if args.gauss:                                                                                  #Writing within the file whether noise has been added
                print(f"\ngauss noise added with standard deviation = {args.gauss_dev}", file = of, flush = True)
            if args.salt_pepper:
                print(f"\nsalt and pepper noise added with percentage = {args.sp_percent[sp_index]}", file = of, flush = True)

                
            print(f"\nStart time = {timeStart.strftime('%H:%M:%S')}", file = of, flush = True)              #Writing start time, number of runs and epochs in the file
            print(f"Runs = {args.repeat}, epochs = {epochs} \n", file = of, flush = True)
            
            stats = np.ones((args.repeat, num_class+3)) * -1000.0 # OA, AA, K, Aclass   

            old_runs=0     
            if (len(combinations) == 1 and len(args.models) == 1  and args.continue_file):                  #Only in the case of 1 single run, will it allow for continuing from an old file 
                stats, old_runs = continue_existing_file(stats, args.file_path, num_class)                  #Load the already existing stats, where old_runs is the number of runs performed
                print(f"Continuing from: {args.file_path}", file = of, flush = True)                        #Write in the new file, that we are reusing an old file
                for run in range(0,old_runs):                                                               #Print the results of all old runs to the new file   
                    results = stats[run,:]
                    print(f"training run {run+1}", file = of, flush = True)
                    for idx in range(num_class):
                        print(f"Label {idx+1}, accuracy:{results[idx+3]}", file = of, flush = True)
                    print(f"OA: {results[0]}", file = of, flush = True)
                    print(f"AA: {results[1]}", file = of, flush = True)
                    print(f"K: {results[2]}\n", file = of, flush = True)
            
            for pos in range(old_runs, args.repeat):                                                        #Continue for the number of remaining runs
                print(f"training run {pos+1}", file = of, flush = True)
                print(f"training run {pos+1}")

                rstate = args.random_state+pos if args.random_state != None else None
                if args.dataset in ["UH", "DIP", "DUP", "DIPr", "DUPr"]:                                    #In case one is using disjointed sets
                    x_train, x_test, y_train, y_test = \
                        mydata.load_split_data_fix(args.dataset, pixels)    
                else:
                    if args.splitmethod == "sklearn":
                        x_train, x_test, y_train, y_test = \
                            mydata.split_data(pixels, labels, tr_percent, args.splitmethod, rand_state=rstate)
                    elif args.tr_pixelmin==0:                                                               #If we are using the tr_percent as the set number of pixels, multiply it by num_class and provide to split
                        x_train, x_test, y_train, y_test = \
                            mydata.split_data(pixels, labels, [int(tr_percent)] * num_class, args.splitmethod, rand_state=rstate)
                    else:                                                                                   #If we do have a pixelmin above 0, we use the tr percent, unless the resulting number of pixels goes under pixelmin
                        train_nums = []
                        for label_num in range(num_class):
                            train_nums.append(max(args.tr_pixelmin, int(np.count_nonzero(labels == label_num) * tr_percent)))
                        x_train, x_test, y_train, y_test = \
                            mydata.split_data(pixels, labels, train_nums, args.splitmethod, rand_state=rstate)

                if args.use_val:
                    x_val, x_test, y_val, y_test = \
                        mydata.split_data(x_test, y_test, args.val_percent, rand_state=rstate)
                print(f"y_train shape: {y_train.shape}, unique: {np.unique(y_train)}")                      #Show the number of labels in train and test, and the number of unique labels, which should be equal
                print(f"y_test shape: {y_test.shape}, unique: {np.unique(y_test)}")
                
                for ix in range(num_class):                                                                 #Store in the file what the division is, useful for later reviewing
                    print(f"label {ix}, train: {np.count_nonzero(y_train == ix)}, test: {np.count_nonzero(y_test == ix)}", file = of, flush = True)
                inputshape = x_train.shape[1:]
                if model =="CNN":
                    clf = cnn2d.get_model_compiled(inputshape, num_class, w_decay=args.wdecay)
                elif model == "MCNN":
                    clf = mcnn.get_compiled_model(inputshape, num_class)
                else:
                    print("Error: model unknown")
                    return

                valdata = (x_val, keras_to_categorical(y_val, num_class)) if args.use_val else (x_test, keras_to_categorical(y_test, num_class))
                clf.fit(x_train, keras_to_categorical(y_train, num_class),
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=args.verbosetrain,
                                validation_data=valdata,
                                callbacks = [tf_ModelCheckpoint("/tmp/best_model.keras", monitor='val_accuracy', verbose=0, save_best_only=True)])  #Save best model, based on validation accuracy
                del clf; K.clear_session(); gc.collect()
                
                if model =="CNN":                                                                           #Loading the model differs, as the MCNN has custom layers
                    clf = load_model("/tmp/best_model.keras")
                else:
                    clf = load_model(
                                "/tmp/best_model.keras",
                                    custom_objects={
                                        "Erosion2D": Erosion2D,
                                        "Dilation2D": Dilation2D,
                                        "SpatialMorph": SpatialMorph,
                                        "SpectralMorph": SpectralMorph,
                                    }
                                    )
                print("PARAMETERS", clf.count_params())

                if args.residual_map:
                    prediction = np.argmax(clf.predict(pixel_save), axis=1)
                    if model == "CNN":
                        cnn_prediction = prediction
                    else:
                        mcnn_prediction = prediction

                if args.retrain:
                    print("saving for retrain")                                                             #If we are saving for retraining, store in the proper folders
                    save_dir = f"/saved_models/{model}/{tr_percent}"
                    os.makedirs(save_dir, exist_ok=True)
                    os.makedirs(f"saved_models/{model}", exist_ok=True)
                    os.makedirs(f"saved_models/{model}/{tr_percent}", exist_ok=True)
                    new_file_path = f"{save_dir}/{pos}.keras"
                    clf.save(new_file_path)
                    print(os.path.exists(new_file_path))
                    
                
                results = mymetrics.reports(np.argmax(clf.predict(x_test), axis=1), y_test)[2]
                stats[pos,:] = results
                for idx in range(num_class):                                                                #Write results into the file, to be used later
                    print(f"Label {idx+1}, accuracy:{results[3+idx]}", file = of, flush = True)
                print(f"OA: {results[0]}", file = of, flush = True)
                print(f"AA: {results[1]}", file = of, flush = True)
                print(f"K: {results[2]}\n", file = of, flush = True)
            
            timeEnd = datetime.now(ZoneInfo("Europe/Amsterdam"))                                            #Find endtime, once again write whether noise was added
            if args.gauss:
                print(f"\ngauss noise added with standard deviation = {args.gauss_dev}", file = of, flush = True)
            if args.salt_pepper:
                print(f"\nsalt and pepper noise added with percentage = {args.sp_percent[sp_index]}", file = of, flush = True)
            
            print(f"\nFinal time = {timeEnd.strftime('%H:%M:%S')}, duration = {timeEnd - timeStart}", file = of, flush = True)  #Print final time and duration, and the mean results and deviations
            print(f"Average OA = {np.mean(stats[:, 0])}, std = {np.std(stats[:, 0])}", file = of, flush = True)
            print(f"Average AA = {np.mean(stats[:, 1])}, std = {np.std(stats[:, 1])}", file = of, flush = True)
            print(f"Average K = {np.mean(stats[:, 2])}, std = {np.std(stats[:, 2])}", file = of, flush = True)
            of.close()
    
    if args.residual_map and len(args.models)>1:                                                            #If the user wishes to make a residual map (do uncomment the matplotlib import at the top)
        height, width, _ = rawpixels.shape
        residual_map = np.zeros((height, width, 3))                                                         #Set all pixels to black, as the background
        x_coords, y_coords = np.where(rawlabels!= 0)                                                        #Select the x and y coords for all non-background labels
        indices = list(zip(x_coords, y_coords))
        combinations = zip(indices, cnn_prediction, mcnn_prediction, labels)                                #Zip all the essential variables, to loop over
        overlap_counter = 0
        correct_count = 0
        for (x,y), gt_val, cnn_pred, mcnn_pred in combinations:
            cnn_correct = cnn_pred == gt_val
            mcnn_correct = mcnn_pred == gt_val

            if (cnn_correct and mcnn_correct):                                                              #If both models produce the correct result, turn the pixel white
                correct_count+=1
                residual_map[x,y] = (1,1,1)
            elif cnn_correct:                                                                               #Otherwise, if the CNN was correct, the MCNN was wrong, and color the pixel red
                residual_map[x,y] = (1,0,0)
            elif mcnn_correct:
                residual_map[x,y] = (0,0,1)                                                                 #Likewise, is the MCNN was correct, the CNN was wrong, color the pixel blue
            else:
                residual_map[x,y] = (0,1,0)                                                                 #If both models are wrong, we have overlap, and make the pixel green
                overlap_counter += 1
        stringTimeEnd = timeEnd.strftime('%H_%M_%S')
        plt.imsave(f"residual_map_{stringTimeEnd}_CNN=Blue_MCNN=Red_Both=Green_Overlap={overlap_counter}.png", residual_map)        #Store image, the OA and AA for the image will be stored in the usual way
        print(f"correct: {correct_count}")


if __name__ == '__main__':
    main()

