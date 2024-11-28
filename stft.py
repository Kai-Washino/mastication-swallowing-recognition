import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

class STFT:
    def __init__(self, sample_rate, data):
        self.sample_rate = sample_rate
        self.data = data
        self.window_length = int(0.03 * self.sample_rate)
        self.hop_length = int(self.window_length / 2)

    def generate_spectrogram(self):
        # データが複数のチャネルを持つ場合は平均を取る
        if len(self.data.shape) > 1:
            self.data = self.data.mean(axis=1)
        
        # スペクトログラムを格納するリスト
        spectrogram = []
        
        # ウィンドウごとにFFTを実行
        for start in range(0, len(self.data) - self.window_length, self.hop_length):
            windowed_data = self.data[start:start + self.window_length]
            Y = fft(windowed_data)
            power = np.abs(Y[:int(self.window_length / 2)])**2
            spectrogram.append(power)
        
        self.spectrogram = np.array(spectrogram).T
        # スペクトログラムを時間に沿った配列に変換
        return  self.spectrogram
    

    def plot_spectrogram(self):
        # 時間軸の長さを計算
        time_axis = np.arange(self.spectrogram.shape[1]) * self.hop_length / self.sample_rate
        # 周波数軸の長さを計算
        frequency_axis = np.linspace(0, self.sample_rate / 2, num=self.spectrogram.shape[0])

        plt.figure(figsize=(10, 4))
        # スペクトログラムをプロット（対数スケールでの強度表示）
        plt.pcolormesh(time_axis, frequency_axis, 10 * np.log10(self.spectrogram), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram')
        plt.colorbar(label='Intensity [dB]')
        plt.ylim([0, self.sample_rate / 2])  # 周波数の表示範囲を設定
        plt.show()
    
    def plot_spectrogram1(self):
        freq = np.linspace(0, self.sample_rate / 2, int(self.window_length / 2), endpoint=True)
        plt.figure(figsize=(10, 4))
        plt.plot(freq, self.power)
        plt.title('FFT Result')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import pathlib
    import scipy.io.wavfile as wav
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/swallowing/dataset/washino/chewing/chewing1.wav')
    sample_rate, data = wav.read(path)
    swallowing1 = STFT(sample_rate, data)
    spectrogram = swallowing1.generate_spectrogram()
    print(spectrogram.shape)
    print(spectrogram.min())
    swallowing1.plot_spectrogram()

        