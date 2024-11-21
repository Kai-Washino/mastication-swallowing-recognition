import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

class Wavelet:
    """
    Waveletクラスは、wavファイルから取得されたサンプリングレートと音声データに対してウェーブレット変換を行います。
    このクラスは、gaus5（Complex Gaussian mother wavelet）を変換窓として使用します。

    Attributes:
        sample_rate (int): サンプリング周波数。
        data (np.ndarray): 音声データの配列。

    Methods:
        generate_coefficients(): ウェーブレット変換の係数と周波数を計算し、返します。
        plot_spectrogram(): 音声データのウェーブレット変換スペクトログラムをプロットします。
    """
    
    def __init__(self, sample_rate, data):
        """
        Waveletクラスのコンストラクタ。

        Args:
            sample_rate (int): サンプリング周波数。
            data (np.ndarray): 音声データの配列。
        """
        self.sample_rate = sample_rate
        self.data = data

    def generate_coefficients(self):
        """
        ウェーブレット変換を実行し、変換係数と対応する周波数を返します。

        Returns:
            np.ndarray: ウェーブレット変換係数の配列。
            np.ndarray: 対応する周波数の配列。
        """
        # ステレオ音声の場合、モノラルに変換
        if len(self.data.shape) > 1:
            self.data = self.data.mean(axis=1)

        # ウェーブレット変換（Complex Gaussian mother wavelet、gaus5を使用）
        wavelet = 'gaus5'
        self.scales = np.arange(1, 128)

        # 連続ウェーブレット変換を実行
        self.coefficients, self.frequencies = pywt.cwt(self.data, self.scales, wavelet, 1.0 / self.sample_rate)
        return self.coefficients, self.frequencies
    
    def plot_spectrogram(self):
        """
        ウェーブレット変換による音声データのスペクトログラムをプロットします。
        """
        plt.figure(figsize=(10, 6))
        # plt.imshow(np.abs(self.coefficients), extent=[0, len(self.data) / self.sample_rate, 1, self.scales[-1]], cmap='PRGn', aspect='auto',
        #            vmax=abs(self.coefficients).max(), vmin=-abs(self.coefficients).max())

        plt.imshow(np.abs(self.coefficients), extent=[0, 1, 1, 150], cmap='PRGn', aspect='auto',
                   vmax=50000, vmin=-50000)
        
        plt.ylabel('Scale')
        plt.xlabel('Time (sec)')
        plt.title('Wavelet Transform (Spectrogram) of Audio')
        plt.colorbar(label='Magnitude')
        plt.show()

if __name__ == "__main__":
    import pathlib
    path = pathlib.Path('C:/Users/S2/Documents/デバイス作成/2024測定デバイス/swallowing/dataset/washino/swallowing/swallowing1.wav')
    sample_rate, data = wav.read(path)
    swallowing1 = Wavelet(sample_rate, data)
    coefficients  = swallowing1.generate_coefficients()
    print(type(coefficients))
    print(type(coefficients) == tuple)
    swallowing1.plot_spectrogram()
