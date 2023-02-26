import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf 
from statistics import mode

class DataClass:
    """
    Class to load, parse, preprocess, and encode data
    """
    def __init__(self, meta_filename, filepath, \
        use_composition = None, use_key = True, \
        use_movement = True, use_ensemble = True, modulation = True) -> None:
        """
        Constructor method to prepare data.

        params:
        meta_filename is the file name of the file containg metadata
        filepath is the path of the folder that contains all the data(labels)
        use_* indicates whether to use those meta features as the data feature 
        """
        self.use_composition = use_composition
        self.use_key = use_key
        self.use_movement = use_movement
        self.use_ensemble = use_ensemble
        #load meta-data
        meta_data = pd.read_csv(meta_filename)

        #load music
        music_data = self.load_music_data(filepath)

        #merge two data
        all_data = meta_data.merge(music_data, on = 'id')
        all_data = all_data.drop(['source', 'transcriber', 'catalog_name', 'composer'], axis=1)

        all_data['note_value'], self.note_value_ref = self.one_hot_encoding(all_data['note_value'])

        #specific parameters

        if use_key:
            all_data['key'] = self.find_keys(all_data)
            all_data['note_degree'] = self.calculate_degree(all_data['note'], self.find_keys(all_data))
        if use_composition != None:
            all_data = all_data[all_data['composition'] == use_composition]

        if use_movement:
            all_data['movement'], self.movement_ref = self.one_hot_encoding(all_data['movement'])
        else:
            all_data.drop(['movement'], axis=1)

        if use_ensemble:
            all_data['ensemble'], self.ensemble_ref = self.one_hot_encoding(all_data['ensemble'])
        else:
            all_data.drop(['ensemble'])

        if modulation:
            all_data['note'] = self.modulate(all_data['note'], all_data['key'])

        self.data = all_data

    def modulate(self, notes, keys):
        """
        modulate all the notes to the same key(C major)
        """
        modulated_notes = []
        for i in range(len(notes)):
            modulated_notes.append(notes[i] - keys[i])
        return np.array(modulated_notes)

    def encode_nn_ready(self):
        """
        make the pandas data frame a tensor to be passed into the neural network
        """
        X = []
        for index, data in self.data.iterrows():
            cur = []
            cur.append(data['seconds'])
            cur.append(data['start_time'])
            cur.append(data['end_time'])
            cur.append(data['instrument'])
            cur.append(data['note'])
            cur.append(data['start_beat'])
            cur.append(data['end_beat'])
            cur.extend(data['note_value'])
            if self.use_movement:
                cur.extend(data['movement'])
            if self.use_ensemble:
                cur.extend(data['ensemble'])
            if self.use_key:
                cur.append(data['note_degree'])
            X.append(np.array(cur, dtype='float32'))

        return np.array(X, dtype='float32')

    def decode_to_datafrom(self, data):
        df = []
        for d in data: 
            cur = {}
            ind = 0
            cur['seconds'] = int(d[ind])
            ind += 1
            cur['start_time'] = int(d[ind])
            ind += 1
            cur['end_time'] = int(d[ind])
            ind += 1
            cur['instrument'] = int(d[ind])
            ind += 1
            cur['note'] = int(d[ind])
            ind += 1
            cur['start_beat'] = d[ind]
            ind += 1
            cur['end_beat'] = d[ind]
            ind += 1
            if self.use_movement:
                cur['movement'] = self.movement_ref[np.argmax(d[ind:ind + len(self.movement_ref)])]
                ind += len(self.movement_ref)
            if self.use_ensemble:
                cur['ensemble'] = self.ensemble_ref[np.argmax(d[ind:ind + len(self.ensemble_ref)])]
                ind += len(self.ensemble_ref)
            df.append(cur)
        
        return pd.DataFrame.from_records(df)

    def load_music_data(self, filepath):
        """
        load the music data from files
        filepath should be the path of the folder containing all metadata and label data
        """
        all_filenames = glob.glob(filepath + "/*.csv")
        music_data = []
        for filename in all_filenames:
            temp = pd.read_csv(filename)
            temp['id'] = os.path.basename(filename[:-4])
            music_data.append(temp)
        music_data = pd.concat(music_data, axis=0, ignore_index=True)
        music_data['id'] = music_data['id'].astype(str).astype(int)

        return music_data

    def one_hot_encoding(self, array):
        """
        one hot encode categorical feature

        params: an 1-d array of categorical data

        return: a 2-d array of one hot encoded given data, and a dictionary to 
        transform generated feature back into one of the categories.
        """
        value_dict = {value : index for index, value in enumerate(np.unique(array))}
        ref_dict = {index : value for index, value in enumerate(np.unique(array))}
        array_trans = [value_dict[value] for value in array]
        ohe = tf.one_hot(array_trans, len(value_dict))
        return list(ohe), ref_dict

    def calculate_degree(self, notes, keys):
        """
        calculate the tonal degree of each term

        params: a 1-d array of notes and a 1-d of keys of the piece where the note
        came from. They should have the same length

        return: array of note degree
        """
        re = []
        for (n, k) in zip(notes, keys):
            diff = n % 12 - k
            diff = diff + 12 if diff < 0 else diff
            re.append(diff)
        return np.array(re)

    def find_keys(self, all_data):
        """
        find the key each note is in. If the composition specifies the key, use 
        it, otherwise we assume the most common note in the peice to be the key.

        params: data frome of the complete data

        return: array of keys of each input note
        """
        self.ref = {'C':0, 'C-sharp':1, 'D-flat':1, 'D':2, 'D-sharp':3, 'E-flat':3, 'E':4, \
            'F':5, 'F-sharp':6, 'G-flat':6, 'G':7, 'G-sharp':8, 'A-flat':8, 'A':9, 
            'A-sharp':10, 'B-flat':10, 'B':11}

        data = np.array(all_data['composition'])
        unresolved = []

        for i, d in enumerate(data):
            if d.split()[-2] in self.ref:
                data[i] = self.ref[d.split()[-2]]
            else:
                if d not in unresolved:
                    unresolved.append(d)
        
        unresolved_dict = {}
        for composition in unresolved:
            notes_in_piece = all_data[all_data['composition'] == composition]['note']
            notes_in_piece = np.array(notes_in_piece) % 12
            #we assume the key is the most common note in the piece
            unresolved_dict[composition] = mode(notes_in_piece)
        for i, d in enumerate(data):
            if d in unresolved_dict:
                data[i] = unresolved_dict[d]
        
        return np.array(data)
        
            
