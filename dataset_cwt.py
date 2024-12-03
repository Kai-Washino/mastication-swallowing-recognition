import numpy as np
import pathlib
import os
import cv2

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .audio import Audio
from .wavelet import Wavelet


class DataSetCWT():
    def __init__(self, num_samples, img_height=224, img_width=224, channels=3, num_class=3, silence_recognition = False):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.num_class = num_class                
        self.data = np.zeros((num_samples, img_height, img_width, channels))
        self.labels = np.zeros((num_samples, num_class))   
        self.silence_recognition = silence_recognition     

    def add_to_dataset(self, i, coefficients, label):        
        spectrogram = np.abs(coefficients)
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val)
        resized_spectrogram = cv2.resize(normalized_spectrogram, (self.img_width, self.img_height))
        resized_spectrogram_uint8 = (resized_spectrogram * 255).astype(np.uint8)

        # グレースケール画像をRGBに変換
        resized_spectrogram_rgb = cv2.cvtColor(resized_spectrogram_uint8, cv2.COLOR_GRAY2RGB)
    
        # データセットに追加
        self.data[i] = resized_spectrogram_rgb
        self.labels[i] = label
        
    def folder_to_dataset(self, folder_name, label, start_num, ):        
        file_names = self.get_wav_files(folder_name)
        for i, file_name in enumerate(file_names):
            wav = Audio(folder_name / file_name, silence_recognition = self.silence_recognition)
            wavdata = Wavelet(wav.sample_rate, wav.trimmed_data, )
            coefficients, _ =  wavdata.generate_coefficients()
            self.add_to_dataset(start_num + i, coefficients, label)

    def get_wav_files(self, directory):
        wav_files = []
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                wav_files.append(filename)
        return wav_files

if __name__ == "__main__":    
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/swallowing/dataset/washino/voice/voice12.wav')
    wav1 = Audio(path)
    swallowing1 = Wavelet(wav1.sample_rate, wav1.trimmed_data, )
    coefficients, _ =  swallowing1.generate_coefficients()
    print(coefficients)
    data = DataSetCWT(15)
    label = np.array(0)
    data.add_to_dataset(0, coefficients, label)
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
