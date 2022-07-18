import json
import os
from os.path import exists


def saveContentToFile(path, file, content):
    if os.path.isfile(path) and os.access(path, os.R_OK):
        # checks if file exists
        print("File exists and is readable")
    else:
        print("Either file is missing or is not readable, creating file...")
        with open(os.path.join(path, file), 'w') as fout:
            json.dump(content, fout)


def getPathToFile(sen_len, num_ep, filter_ds=False):
    model_dir = '../model/' + str(sen_len)
    if filter_ds:
        model_filename = '/model_2layers_' + str(num_ep) + 'e_filtered'
    else:
        model_filename = '/model_2layers_' + str(num_ep) + 'e_unfiltered'

    path_to_file = model_dir + model_filename
    loop_counter = 1
    if exists(path_to_file + ".h5"):
        while exists(path_to_file + str(loop_counter) + ".h5"):
            loop_counter += 1
        path_to_file = path_to_file + str(loop_counter)
    path_to_file = path_to_file + ".h5"
    print("Model is saved at: " + path_to_file)
    return path_to_file

