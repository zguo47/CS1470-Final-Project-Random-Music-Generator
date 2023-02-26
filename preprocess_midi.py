import numpy as np
import tensorflow as tf 
import pandas as pd
import glob
from mido import MidiFile
from mido import MidiTrack
from mido import Message
import os

def preprocessing(folder_path):
    """
    main function to prepare data
    load the data from the given folder path, set it up into notes. 
    """
    all_tracks = load_midi(folder_path)
    tunes = []
    for k, v in all_tracks.items():
        if 'Right' in k:
            new_tunes = parse_notes(v)
            if len(new_tunes) > 0:
                tunes.append(pd.DataFrame(new_tunes[0]))

    phrase_len = 60
    X = []
    y = []
    for t in tunes:
        for i in range(len(t) - phrase_len):
            if various(t.iloc[i:i + phrase_len, 1]):
                X.append(t.iloc[i:i + phrase_len, :3])
                y.append(t.iloc[i + phrase_len, :3])
    X = np.array(X)
    y = np.array(y)

    X = X.astype(int)
    y = y.astype(int)

    return X, y
    
def various(notes):
    flag = True
    for i in range(8, len(notes)):
        flag = len(np.unique(notes[i-8:i])) > 2
        if not flag:
            break
    return flag

def load_midi(folder_path):
    """
    load the midi files in the folder_path, create a dictionary of all midi tracks
    !!"Bach" folder contains midi tracks in the wrong format, so we ignore those data
    
    params: the path of the folder containing midid files, should be ../musicnet_midis
    returns: all the midi_tracks
    """
    midi_traks = {}
    name = None
    all_filepaths = glob.glob(folder_path+'/*/*.mid')
    for filepath in all_filepaths:
        if 'Bach' in filepath: #Bach causes EOF error
            continue
        #print(filepath)
        midi_file = MidiFile(filepath, clip=True)
        for j in range(len(midi_file.tracks)):
            if j == 0:
                name = midi_file.tracks[j].name + ': '
            else:
                midi_traks[name + midi_file.tracks[j].name] = midi_file.tracks[j]
    return midi_traks

def get_key(s):
    k = None
    if 'key' in s:
        k = s[33:35]
        if k[-1] == "m" or k[-1] == "'":
            k = k[:-1]
    return k

def parse_notes(track):
    key = 'C'
    tunes = []
    new_tune = []
    note_dict = {}
    for i in track:
        
        if i.is_meta:
            new_key = get_key(str(i))
            if new_key is not None:
                key = new_key
            if len(tunes) > 0:
                tunes.append(new_tune)
                new_tune = []
                
        elif i.type == 'note_on' or i.type == 'note_off':
            if i.type == 'note_on' and i.dict()['velocity'] > 0 and i.dict()['time'] > 0:
                note_dict['time'] = i.dict()['time']
                note_dict['note'] = i.dict()['note']
                note_dict['velocity'] = i.dict()['velocity']
                note_dict['channel'] = i.dict()['channel']
            elif i.type == 'note_off' or i.type == 'note_on' and i.dict()['velocity'] == 0:
                if note_dict:
                    note_dict['pause'] = i.dict()['time']
                    note_dict['key'] = key
                    new_tune.append(note_dict)
                    note_dict = {}
    tunes.append(new_tune)
    return tunes

def tune_to_midi(tune, midi_name='new_tune', debug_mode=False):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for note in tune:
        if debug_mode:
            track.append(Message('note_on', note=note, time=64))
            track.append(Message('note_off', note=note, time=128))
        else:
            track.append(Message('note_on', note=note['note'], velocity=note['velocity'], time=note['time']))
            track.append(Message('note_off', note=note['note'], time=note['pause']))

    mid.save(midi_name + '.mid')