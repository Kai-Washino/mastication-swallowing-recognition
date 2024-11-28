import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Masking, Flatten, Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class CNN(): 
    def __init__(self, scale=661, time_range=65, num_class=3, start_filter=8):
        self.num_class = num_class
        self.scale = scale
        self.time_range = time_range                
        self.model = tf.keras.models.Sequential([
            Masking(mask_value=0.0, input_shape=(time_range, scale)),            
            Conv1D(start_filter, 3, activation='relu'),  # 第1畳み込み層
            MaxPooling1D(2),  # 第1プーリング層
            Conv1D(start_filter * 2, 3, activation='relu'),  # 第2畳み込み層
            MaxPooling1D(2),  # 第2プーリング層
            Conv1D(start_filter * 2 * 2, 3, activation='relu'),  # 第3畳み込み層
            MaxPooling1D(3),  # 第3プーリング層
            Conv1D(start_filter * 2 * 2 * 2, 3, activation='relu'),  # 第4畳み込み層
            MaxPooling1D(1),  # 第4プーリング層
            Flatten(),  # データのフラット化
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.num_class, activation='softmax')  # 修正ポイント
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def training(self, train_data, train_labels, epochs, batch_size, early_stopping = None, model_checkpoint = None):
        if early_stopping == None and model_checkpoint == None:
            self.model.fit(train_data, train_labels, epochs=epochs, validation_split=0.1, batch_size= batch_size)
        elif early_stopping == None:
            self.model.fit(train_data, train_labels, epochs=epochs, validation_split=0.1, batch_size= batch_size, callbacks=[model_checkpoint])
        elif model_checkpoint == None:
            self.model.fit(train_data, train_labels, epochs=epochs, validation_split=0.1, batch_size= batch_size, callbacks=[early_stopping])
        else:
            self.model.fit(train_data, train_labels, epochs=epochs, validation_split=0.1, batch_size= batch_size, callbacks=[early_stopping, model_checkpoint])
        
    def evaluate(self, test_data, test_labels):
        self.test_loss, self.test_accuracy = self.model.evaluate(test_data, test_labels)
        print("Test accuracy: ", self.test_accuracy)
        self.predictions = self.model.predict(test_data)
        self.predicted_classes = np.argmax(self.predictions, axis=1)
        self.true_classes = np.argmax(test_labels, axis=1)
        self.correctly_classified = self.predicted_classes == self.true_classes        
        self.correct_indices = np.where(self.correctly_classified)[0]
        self.incorrect_indices = np.where(~self.correctly_classified)[0]

        print("正しく分類されたサンプルのインデックス:", self.correct_indices)
        print("誤って分類されたサンプルのインデックス:", self.incorrect_indices)
        for i in self.incorrect_indices:
            print(f"サンプル {i}: 正解 = {self.true_classes[i]}, 予測 = {self.predicted_classes[i]}")

    def evaluate_print(self):
        print(self.predictions)
        print(self.predicted_classes)
        print(self.true_classes)
        print(self.correctly_classified)
        print(self.correct_indices)
        print(self.incorrect_indices)

    def save(self, file_name):
        self.model.save(file_name)

if __name__ == "__main__":
    from .dataset_stft import DataSetSTFT
    import pathlib
    import numpy as np
    directory_path = pathlib.Path('C:\\Users\\S2\\Documents\\デバイス作成\\2024測定デバイス\\swallowing\\dataset')
   
    train_voice_folder = directory_path / 'washino' / 'voice'
    train_cough_folder = directory_path / 'washino' / 'cough'
    train_swallowing_folder = directory_path / 'washino' / 'swallowing'    

    test_voice_folder = directory_path / 'shibata' / 'voice'
    test_cough_folder = directory_path / 'shibata' / 'cough'
    test_swallowing_folder = directory_path / 'shibata' / 'swallowing'    
    
    # train_data = VariableDataSet()
    test_data = DataSetSTFT(num_samples=28)

    # train_data.folder_to_dataset(train_swallowing_folder, np.array(0))
    # train_data.folder_to_dataset(train_cough_folder, np.array(1))    
    # train_data.folder_to_dataset(train_voice_folder, np.array([1, 0, 0]), 2)
    # train_data.print_label()
    test_data.folder_to_dataset(test_swallowing_folder, np.array(0), 0)
    test_data.folder_to_dataset(test_cough_folder, np.array(1), 14)

    # test_data.folder_to_dataset(test_voice_folder, np.array([1, 0, 0]), 2)

    model = CNN()
    # model.training(train_data.data, train_data.labels, 1, 32)
    model.training(test_data.data, test_data.labels, 2, 32)
    # model.evaluate(test_data.data, test_data.labels)
    # model.save('20240116_159datasets.keras')