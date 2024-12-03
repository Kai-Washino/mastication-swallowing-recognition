import numpy as np
import pathlib
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .audio import Audio
from .stft import STFT


class DataSetSTFT():
    def __init__(self, num_samples, scale = 661, time_range = 65, num_class=3, silence_recognition = False):
        self.time_range = time_range        
        self.scale = scale
        self.data = np.zeros((num_samples, self.time_range, self.scale))
        self.labels = np.zeros((num_samples, num_class)) 
        self.max_cols = 0
        self.silence_recognition = silence_recognition

    def add_to_dataset(self, i, data, label):        
        if type(data) == tuple:
            spectrogram = np.abs(data)        
        else:
            spectrogram = data         
            
        if len(spectrogram) == 0:
            print("Warning: No data available for FFT.") 
            print(i)
            print(spectrogram)
            return 
            
        scaler_X = MinMaxScaler()        
        if spectrogram.ndim == 1:
            spectrogram = spectrogram.reshape(-1, 1)  # 1次元配列を2次元配列に変換               
        normalized_spectrogram = scaler_X.fit_transform(spectrogram)
                                 
        data = self.trim_or_pad(normalized_spectrogram)  
        data = data.reshape(self.time_range, self.scale)       
            
        self.data[i] = data
        self.labels[i] = label

    def trim_or_pad(self, data):
        current_length = data.shape[1]        
        if current_length > self.time_range:
            # time_range以上の場合はトリミング            
            trimmed_data = data[:, :self.time_range]       
            return trimmed_data
        elif current_length < self.time_range:
            # time_range未満の場合はパディング
            padding_length = self.time_range - current_length
            padded_data = np.pad(data, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
            return padded_data
        else:
            # すでにtime_rangeの場合はそのまま返す
            return data  
        
    def folder_to_dataset(self, folder_name, label, start_num):        
        file_names = self.get_wav_files(folder_name)
        for i, file_name in enumerate(file_names):
            wav = Audio(folder_name / file_name, silence_recognition=self.silence_recognition)
            wavdata = STFT(wav.sample_rate, wav.trimmed_data, )
            spectrogram = wavdata.generate_spectrogram()
            self.add_to_dataset(start_num + i, spectrogram, label)

    def get_wav_files(self, directory):
        wav_files = []
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                wav_files.append(filename)
        return wav_files

if __name__ == "__main__":    
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/swallowing/dataset/washino/voice/voice12.wav')
    wav1 = Audio(path)
    swallowing1 = STFT(wav1.sample_rate, wav1.trimmed_data, )
    spectrogram =  swallowing1.generate_spectrogram()
    print(spectrogram.shape)
    data = DataSetSTFT(15)
    label = np.array([0, 1, 0])
    data.add_to_dataset(0, spectrogram, label)
    print(len(data.data))
    print(len(data.data[0]))
    print(len(data.data[0][2]))
    print(data.labels)

    directory_path = pathlib.Path('C:\\Users\\S2\\Documents\\デバイス作成\\2024測定デバイス\\swallowing\\dataset')   
    train_voice_folder = directory_path / 'shibata' / 'voice'
    data.folder_to_dataset(train_voice_folder, label, 1)
    print(len(data.data))
    print(len(data.data[0]))
    print(len(data.data[0][5]))
