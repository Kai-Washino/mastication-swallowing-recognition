import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

class Audio:
    def __init__(self, path):
        self.path = path
        self.sample_rate, self.original_data = wav.read(path)
        data = self.original_data
        if len(data.shape) > 1:
            data = data.mean(axis=1)        
        start, end = self.find_start_end(self.sample_rate, data)
        self.length = end - start
        self.trimmed_data = self.original_data[start:end]

    def find_start_end(self, sample_rate, data):
        # 最大音量の10%を計算
        max_vol = np.max(np.abs(data))
        threshold = 0.1 * max_vol

        # 開始位置を見つける
        start_idx = np.where(np.abs(data) >= threshold)[0][0]

        # 186ミリ秒のサンプル数を計算
        silence_length = int(0.186 * sample_rate)

        # 終了位置を見つける
        end_idx = len(data)
        for i in range(start_idx + silence_length, len(data)):
            if np.all(np.abs(data[i - silence_length:i]) < threshold / 10):
                end_idx = i - silence_length
                break

        return start_idx, end_idx
        
    @staticmethod
    def plot_waveform(data, title):
        plt.figure(figsize=(10, 4))
        plt.plot(data, color = 'black')
        plt.title(title)
        plt.ylim(-30000, 30000)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

    def original_plot(self):
        Audio.plot_waveform(self.original_data, "Original")

    def trimmed_plot(self):
        Audio.plot_waveform(self.trimmed_data, "Trimmed")

if __name__ == "__main__":
    import pathlib      
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/swallowing/dataset/washino/swallowing/swallowing1.wav')
    wav1 = Audio(path)
    wav1.trimmed_plot()
    wav1.original_plot()    
    plt.show()