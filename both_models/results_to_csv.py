#Code created for the purpose of making result.csv files, for importing into google sheets. 
#GPL-3.0 licensed
#Takes the OA, AA and deviations from all files within a folder, and places them in a results file.
#Goes through all subdirectories from a provided directory

import os
import numpy as np
import re
import argparse
import csv



def process_file(stats, path, class_num = 16):
    if "UP" in path:
        class_num = 9
    loaded_run = []
    of = open(path, "r")
    labelCount=0
    resultCount=0
    for line in of:
        line = line.strip()
        match = re.search(r".*? accuracy:\s?(\d+(?:\.\d+)?)", line)                 #Increment the label count, to check whether the file is properly formed
        if match:
            labelCount+=1

        elif line.startswith(("OA", "AA", "K")):                                    #Only OA, AA and K values are saved
            loaded_run.append(float(line.split(":")[1]))
            resultCount+=1

        if labelCount==class_num and resultCount == 3:                              #If we have found the values of a complete run, add to the stats
            stats.append(loaded_run)
            loaded_run = []
            labelCount = 0
            resultCount = 0
        
        if labelCount > class_num or resultCount > 3:                               #If a file is not shaped properly, it will discontinue the results from this file, but continue with others
            print(f"Error, labelcount = {labelCount}, resultcount = {resultCount}")
            break
    of.close()
    return

def final_dir(path):                                                                #If we have reached a final directory, with no subdirectories
    if "MNN" in path or "MCNN" in path:
        model = "MCNN_"
    else:
        model = "CNN_"
    final_results = []
    deviations = []
    for filename in sorted(os.listdir(path)):                                       #For each filename within this directory, as long as it ends with .txt, but is not a result_list.txt
        if (not filename.endswith(".txt")) or filename.endswith("result_list.txt"):
            continue
        stats = []
        filepath = os.path.join(path, filename)
        percentage = filename.split("training_size=")[1].split("_")[0]              #Extract the training percentage from the file name
        process_file(stats, filepath)                                               #Process the file, taking out the stats
        stats = np.array(stats)
        #Here we append the percentage, along with the data string, as a tuple, allowing for sorting
        final_results.append((percentage, [f"{percentage}\t{np.mean(stats[:,0])}\t{np.mean(stats[:,1])}\t{np.mean(stats[:,2])}"]))      #Append the percentage, and the mean OA, AA and K to the final results 
        deviations.append((percentage, [f"{percentage}\t{np.std(stats[:,0])}\t{np.std(stats[:,1])}\t{np.std(stats[:,2])}"]))            #To the deviations, we add the percentage, and then the standard deviation of OA, AA and K
    final_results.sort(key = lambda x: float(x[0]))                                 #Sort the data based on the percentage
    deviations.sort(key = lambda x: float(x[0]))
    
    of = open(os.path.join(os.path.dirname(filepath), "results.csv"), "w")          #Open a new file, and add the data in a table shape, with tabs as separators
    csv.writer(of).writerow([path])
    csv.writer(of).writerow([f"tr_percent\t{model}OA\t{model}AA\t{model}K"])
    csv.writer(of).writerows(line for _, line in final_results)
    
    csv.writer(of).writerow([])
    csv.writer(of).writerow(["Deviations:"])
    csv.writer(of).writerows(line for _, line in deviations)

    of.close()
    print(f"results written in path: {path}")                                       #Print to the terminal that results have been added in said path
    return

def process_dir(path):                                                              #Processes all directories
    guard = 1
    if len(os.listdir(path)) == 0:                                                  #If it is an empty directory, return
        return
    
    for filename in sorted(os.listdir(path)):
        new_path = os.path.join(path,filename)
        if os.path.isdir(new_path):                                                 #If the current path + filename points to another directory, recursively process that
            guard = 0
            process_dir(new_path)
        
    if guard:                                                                       #If no subdirectories have been found, process the current directory as a final directory
        final_dir(path)
    return

def main():
    print("\n\n\n\n")
    parser = argparse.ArgumentParser(description='Result parser')
    parser.add_argument('--dir_path', default = "", type=str )                      #Directory for results to be processed. Allows for directories with subdirectories

    args = parser.parse_args()
    
    process_dir(args.dir_path)
    


if __name__ == '__main__':
    main()
