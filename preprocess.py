# -*- coding: utf-8 -*-
import csv
from common import load_track, GENRES
import sys
import numpy as np
from math import pi
from pickle import dump
import os
from optparse import OptionParser

# TRACK_COUNT = 1000
TRACK_COUNT = 6454 # number of songs clasified

def create_folders(dataset_path):
    with open('first_genres.csv', "rt", encoding="utf8") as csvfile:
        os.chdir(dataset_path) # raw string
        #create directories
        for gen in GENRES:
            if not os.path.exists(gen):
                os.makedirs(gen)
        # read rows
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            name = row[0].lower().replace('-', '')
            name = ''.join([i for i in name if not i.isdigit()])
            genre = row[1]
            # moving files
            for file in os.listdir():
                name_or = os.path.join(file)
                name_file = name_or[:-4].lower().replace('-', '')
                name_file = ''.join([i for i in name_file if not i.isdigit()])
                if name_file == name:
                    print(name_file)
                    os.rename(name_or, genre + "/" + name_or)


def get_default_shape(dataset_path):
    tmp_features, _ = load_track(os.path.join(dataset_path,
        'Bolero Son/Juramento - Rey Caney.mp3.wav'))
    return tmp_features.shape

def collect_data(dataset_path):
    '''
    Collects data from the GTZAN dataset into a pickle. Computes a Mel-scaled
    power spectrogram for each track.

    :param dataset_path: path to the GTZAN dataset directory
    :returns: triple (x, y, track_paths) where x is a matrix containing
        extracted features, y is a one-hot matrix of genre labels and
        track_paths is a dict of absolute track paths indexed by row indices in
        the x and y matrices
    '''
    default_shape = get_default_shape(dataset_path)
    x = np.zeros((TRACK_COUNT,) + default_shape, dtype=np.float32)
    y = np.zeros((TRACK_COUNT, len(GENRES)), dtype=np.float32)
    track_paths = {}
    counter = 0

    for (genre_index, genre_name) in enumerate(GENRES):
        curr_path = dataset_path+"/"+genre_name+"/"
        os.chdir(curr_path)
        for file in os.listdir():
            print('Processing', file)
            path = os.path.join(file)
            x[counter], _ = load_track(path, default_shape)
            y[counter, genre_index] = 1
            track_paths[counter] = os.path.abspath(path)
            counter+=1

    return (x, y, track_paths)

if __name__ == '__main__':
    # parser = OptionParser()
    # parser.add_option('-d', '--dataset_path', dest='dataset_path',
    #         default=os.path.join(os.path.dirname(__file__), 'data/genres'),
    #         help='path to the GTZAN dataset directory', metavar='DATASET_PATH')
    # parser.add_option('-o', '--output_pkl_path', dest='output_pkl_path',
    #         default=os.path.join(os.path.dirname(__file__), 'data/data.pkl'),
    #         help='path to the output pickle', metavar='OUTPUT_PKL_PATH')
    # options, args = parser.parse_args()
    #
    # (x, y, track_paths) = collect_data(options.dataset_path)
    #
    # data = {'x': x, 'y': y, 'track_paths': track_paths}
    # with open(options.output_pkl_path, 'wb') as f:
    #     dump(data, f)

    starting_dir = os.getcwd()
    
    dataset_path="D:/Documents/DL/FinalProject/SalsaClass/fragmentos-drive/fragmentos"
    # dataset_path="D:/Documents/DL/FinalProject/SalsaClass/fragmentos-drive"
    # create_folders(dataset_path)
    # get_default_shape(dataset_path)
    (x, y, track_paths) = collect_data(dataset_path)

    os.chdir(starting_dir)
    print(os.getcwd())

    data = {'x': x, 'y': y, 'track_paths': track_paths}
    with open('data/data.pkl', 'wb') as f:
        dump(data, f)
