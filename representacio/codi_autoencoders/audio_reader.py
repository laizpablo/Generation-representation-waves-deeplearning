import fnmatch
import os
import re

import librosa
import numpy as np
import tensorflow as tf
import random as rnd
import helper as hp


# Recursively finds all files matching the pattern.
# Input:
#   - directory: where the audios will be
#   - patters: audio type
# Output:
#   - files: list with audio directories
def find_files(directory, pattern='*.wav'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    rnd.shuffle(files, rnd.random)
    return files


# Generator that yields audio waveforms from the directory.
# Input:
#   - directory: where the audios will be
#   - sample_rate: number of samples in one second
# Output:
#   - audio: numpy list with the samples
#   - filename: file name of the directory
def load_generic_audio(filename, sample_rate):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    return audio, filename


# Generic background audio reader that preprocesses audio files
# and enqueues them into a queue.
class AudioReader(object):

    # Inputs: 
    #   - audio_dir: directory where the data is
    #   - sample_rate: samples in one second
    #   - sample_size: max samples to train
    def __init__(self, audio_dir, sample_rate=16000, sample_size=1024, size_queue = 500):

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.queue = []
        self.size_queue = size_queue
        self.buffer_ = np.array([])

        if not find_files(audio_dir):
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

        self.files = find_files(self.audio_dir)

        self.size_test = int(len(self.files)/10.)
        self.files_train = self.files[:-self.size_test]
        self.files_test = self.files[-self.size_test:]

        len_audio = len(load_generic_audio(self.files[0], self.sample_rate)[0])
        self.total_data = ((len(self.files)) - self.size_test) * int(len_audio/1024)
        
        while len(self.queue) < self.size_queue and len(self.files_train) > 0:
            audio, filename = load_generic_audio(self.files_train.pop(), self.sample_rate)
            
            self.buffer_ = np.append(self.buffer_, audio)
            while len(self.buffer_) > self.sample_size:
                    piece = self.buffer_[:1024]
                    self.queue.append(piece)
                    self.buffer_ = self.buffer_[self.sample_size:]
        
        if len(self.files_train) <= 0:
            self.files_train = self.files[:-self.size_test]

    def get_batch_train(self, batch_size):
        if len(self.files_train) <= 0:
            self.files_train = self.files[:-self.size_test]
            
        while len(self.queue) < self.size_queue and len(self.files_train) > 0:

            audio, filename = load_generic_audio(self.files_train.pop(), self.sample_rate)
            
            self.buffer_ = np.append(self.buffer_, audio)
            while len(self.buffer_) > self.sample_size:
                    piece = self.buffer_[:self.sample_size]
                    self.queue.append(piece)
                    self.buffer_ = self.buffer_[self.sample_size:]
             
        rnd.shuffle(self.queue, rnd.random)
        batch = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        return batch
    
    
    def get_batch_test(self, size = int(1e6)):
        buffer_ = []
        queue = []
        while len(self.files_test) > 0 and len(queue)< size:

            audio, filename = load_generic_audio(self.files_test.pop(), self.sample_rate)
            
            buffer_ = np.append(buffer_, audio)
            while len(buffer_) > self.sample_size:
                    piece = buffer_[:1024]
                    queue.append(piece)
                    buffer_ = buffer_[self.sample_size:]
        
        rnd.shuffle(queue, rnd.random)
        return queue
    
    def reset_data(self):
        self.files_train = self.files[:-self.size_test]
        self.queue = []

# Generic background audio reader that preprocesses Specgram files
# and enqueues them into a queue.
class AudioReaderSpectrogram(object):

    # Inputs: 
    #   - audio_dir: directory where the data is
    #   - sample_rate: samples in one second
    #   - sample_size: max samples to train
    def __init__(self, audio_dir, sample_rate=16000, sample_size=1024, size_queue = 500):

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.queue = []
        self.size_queue = size_queue
        self.buffer_ = np.array([])
        
        if not find_files(audio_dir):
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

        self.files = find_files(self.audio_dir)

        self.size_test = int(len(self.files)/10.)
        self.files_train = self.files[:-self.size_test]
        self.files_test = self.files[-self.size_test:]

        len_audio = len(load_generic_audio(self.files[0], self.sample_rate)[0])
        self.total_data = ((len(self.files)) - self.size_test) * int(len_audio/1024)

        
        while len(self.queue) < self.size_queue and len(self.files_train) > 0:
            audio, filename = load_generic_audio(self.files_train.pop(), self.sample_rate)
            
            self.buffer_ = np.append(self.buffer_, audio)
            while len(self.buffer_) > self.sample_size:
                piece = self.buffer_[:self.sample_size]
                # Spectrogram
                spec, freqs, t  = hp.specgram(piece)
                self.queue.append([ spec, freqs, t, piece])
                self.buffer_ = self.buffer_[self.sample_size:]
        
        if len(self.files_train) <= 0:
            self.files_train = self.files[:-self.size_test]

    def get_batch_train(self, batch_size):
        if len(self.files_train) <= 0:
            self.files_train = self.files[:-self.size_test]
            
        while len(self.queue) < self.size_queue and len(self.files_train) > 0:

            audio, filename = load_generic_audio(self.files_train.pop(), self.sample_rate)
            
            self.buffer_ = np.append(self.buffer_, audio)
            while len(self.buffer_) > self.sample_size:
                piece = self.buffer_[:self.sample_size]
                # Spectrogram
                spec, freqs, t  = hp.specgram(piece)
                self.queue.append([ spec, freqs, t, piece])
                self.buffer_ = self.buffer_[self.sample_size:]
        
        
        rnd.shuffle(self.queue, rnd.random)
        batch = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        return [spec[0].reshape(-1) for spec in batch]
    
    
    
    def get_batch_test(self, size = int(1e6)):
        buffer_ = []
        queue = []
        while len(self.files_test) > 0 and len(queue) < size:

            audio, filename = load_generic_audio(self.files_test.pop(), self.sample_rate)
            
            buffer_ = np.append(buffer_, audio)
            while len(buffer_) > self.sample_size:
                    piece = buffer_[:1024]
                    #Spectrogram
                    spec, freqs, t  = hp.specgram(piece)
                    queue.append([spec, freqs, t, piece])
                    buffer_ = buffer_[self.sample_size:]
        
        rnd.shuffle(queue, rnd.random)
        return [spec[0].reshape(-1) for spec in queue], queue
    
    def reset_data(self):
        self.files_train = self.files[:-self.size_test]
        self.queue = []