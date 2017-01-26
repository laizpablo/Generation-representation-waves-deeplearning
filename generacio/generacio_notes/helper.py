from datetime import datetime
import fnmatch
import os
import re
import threading

import scipy
import numpy as np
import random as rn

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def find_audios(num_files):
    pattern='*.wav'
    files = []
    for root, dirnames, filenames in os.walk('corpus_notes/'):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))   
    return files[:num_files]


def find_models():
    pattern = '*.ckpt*'
    models = []
    
    for root, dirnames, filenames in os.walk('logdir_notes/train/2017-01-09T19-41-13'):
        for filename in fnmatch.filter(filenames, pattern):
            if filename[-5:] != '.meta':
                models.append(os.path.join(root, filename))
    return models


def find_generation(_dir_folder):
    pattern='*.wav'
    files = []
    for root, dirnames, filenames in os.walk(_dir_folder):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

import scipy.io.wavfile as wav

def load_generic_audio(filename):
    (rate,audio) = wav.read(filename)
    return audio


def show_seed(audio):
    print len(audio)
    if not len(audio) % 16000 == 1:
        fig, ax = plt.subplots(figsize=(18,1))
        plt.plot(audio[:2000])
        plt.title("Seed Generacio")
        plt.show()
        return True
    else: 
        print "No seed"
        return False
    

def show_audios (_list_gen): 
    duration = len(_list_gen[0])/16000.
    rate = 16000
    time = np.linspace(0,duration, num=rate*duration)
    
    window = 4000
    seed_len = 8000
    
    if show_seed(_list_gen[0]):
        cont = 1
        for _gen in _list_gen:
            fig, ax = plt.subplots(figsize=(18,1))
            plt.plot( _gen[-window:])
            iteracio = (cont-1)*10000
            plt.title('Iteracio: '+ str(iteracio))
            cont+=1
            plt.show()
    else:
        cont = 1
        for _gen in _list_gen:
            fig, ax = plt.subplots(figsize=(18,1))
            plt.plot( _gen[-window:])
            iteracio = (cont-1)*10000
            plt.title('Iteracio: '+ str(iteracio))
            cont+=1
            plt.show()
            
def show_audios_complete (_list_gen): 
    duration = len(_list_gen[0])/16000.
    rate = 16000
    time = np.linspace(0,duration, num=rate*duration)
    
    window = 4000
    seed_len = 8000
    
    if show_seed(_list_gen[0]):
        cont = 1
        for _gen in _list_gen:
            fig, ax = plt.subplots(figsize=(18,1))
            plt.plot( _gen[seed_len:])
            iteracio = (cont-1)*10000
            plt.title('Iteracio: '+ str(iteracio))
            cont+=1
            plt.show()
    else:
        cont = 1
        for _gen in _list_gen:
            fig, ax = plt.subplots(figsize=(18,1))
            plt.plot( _gen)
            iteracio = (cont-1)*10000
            plt.title('Iteracio: '+ str(iteracio))
            cont+=1
            plt.show()


def show_list_audios(_list): 
    for audio, _file in _list:
        fig, ax = plt.subplots(figsize=(18,1))
        plt.plot(audio[:16000])
        plt.title(_file)
        plt.show()

def show_one_complete_audio(audio, windows):
    _buffer = audio
    fragmen = 0
    while len(_buffer) > 0:
        
        fig, ax = plt.subplots(figsize=(18,1))
        plt.plot(range(fragmen, fragmen+windows) , _buffer[:windows])
        plt.show()
        _buffer = _buffer[windows:]
        fragmen = fragmen+windows
    
        
        