from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import numpy as np

class DenseNet:
    def __init__(self, num_class=3):
        self.num_class = num_class        
        self.base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3))
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_class, activation='softmax')(x)  # num_class 分類
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        # モデルのコンパイル
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
    from .dataset import DataSet
    import pathlib
    directory_path = pathlib.Path('C:\\Users\\S2\\Documents\\デバイス作成\\2024測定デバイス\\swallowing\\dataset')
   
    test_voice_folder = directory_path / 'shibata' / 'voice'
    test_cough_folder = directory_path / 'shibata' / 'cough'
    test_swallowing_folder = directory_path / 'shibata' / 'swallowing'    

    # train_data = DataSet(200, 224, 224, 3, 2)
    test_data = DataSet(42)

    test_data.folder_to_dataset(test_swallowing_folder, np.array([1, 0, 0]), 0)
    test_data.folder_to_dataset(test_cough_folder, np.array([0, 1, 0]), 14)
    test_data.folder_to_dataset(test_voice_folder, np.array([0, 0, 1]), 28)

    model = DenseNet()
    model.training(test_data.data, test_data.labels, 2, 32)
    # model.save('20240116_159datasets.keras')