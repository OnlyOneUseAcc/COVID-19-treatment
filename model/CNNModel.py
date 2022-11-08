import tensorflow as tf
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from preprocessing import CategoricalPreprocessor


class CNNModel:

    def __init__(self, num_binary_classes, dynamic_columns, window_size=2):
        self.num_binary_classes = num_binary_classes
        self.dynamic_columns = dynamic_columns
        self.window_size = window_size
        self.stat_columns = []

        self.model = None
        self.preprocessed_pipeline = Pipeline(
            [
                ('encoder', CategoricalPreprocessor()),
                ('imputer', KNNImputer(n_neighbors=20)),
                ('scaler', MinMaxScaler())
            ]
        )

    def __build_model(self, current_feat_shape, dynamic_shape):
        current_feat_input = tf.keras.layers.Input(shape=current_feat_shape)
        dynamic_input = tf.keras.layers.Input(shape=dynamic_shape)

        current_feat_layer = tf.keras.layers.Dense(64, activation='relu')(current_feat_input)

        dynamic_layer = tf.keras.layers.Masking(input_shape=dynamic_shape)(dynamic_input)  # for using pad

        dynamic_layer = tf.keras.layers.Conv1D(64, 4, strides=2, padding='same', activation='relu')(dynamic_layer)
        dynamic_layer = tf.keras.layers.Dropout(0.1)(dynamic_layer)

        dynamic_layer = tf.keras.layers.Conv1D(32, 4, strides=2, padding='same', activation='relu')(dynamic_layer)
        dynamic_layer = tf.keras.layers.Dropout(0.1)(dynamic_layer)

        dynamic_layer = tf.keras.layers.Conv1D(16, 4, strides=2, padding='same', activation='relu')(dynamic_layer)
        dynamic_layer = tf.keras.layers.Dropout(0.1)(dynamic_layer)
        dynamic_layer = tf.keras.layers.GlobalAveragePooling1D()(dynamic_layer)
        #dynamic_layer = tf.keras.layers.Flatten()(dynamic_layer)

        concat_layer = tf.keras.layers.Concatenate()([dynamic_layer, current_feat_layer])
        concat_layer = tf.keras.layers.Dense(128, activation='relu')(concat_layer)

        output_layers = []
        for index in range(0, self.num_binary_classes):
            output_layers.append(tf.keras.layers.Dense(2, activation='softmax', name=f'out_{index}')(concat_layer))

        return tf.keras.Model(
            inputs=[current_feat_input, dynamic_input],
            outputs=output_layers
        )

    def __group_dinam_facts(self, X: pd.DataFrame):
        result = []
        data = X[self.dynamic_columns]
        for index in np.unique(data.index):
            group = data.loc[[index]]

            for start in range(0, group.shape[0]):
                group_window = group.iloc[start: start + self.window_size]
                if group_window.shape[0] < self.window_size:  # padding
                    values = np.zeros(shape=(self.window_size, group_window.shape[1]))
                    values[self.window_size - group_window.shape[0]:] = group_window.values
                    result.append(values)
                else:
                    result.append(group_window.values)

        return np.array(result)

    def get_f1(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        f1_val = 2 * (precision*recall) / (precision+recall + tf.keras.backend.epsilon())
        return f1_val

    def fit(self, X: pd.DataFrame, y, eval_data=None, epochs=10, learning_rate=0.001):
        dinam_data = self.__group_dinam_facts(X)
        self.stat_columns = list(set(X.columns).difference(set(self.dynamic_columns)))
        processed_stat_data = pd.DataFrame(
            data=self.preprocessed_pipeline.fit_transform(X[self.stat_columns], y),
            columns=self.preprocessed_pipeline[-1].get_feature_names_out(
                self.preprocessed_pipeline[0].get_feature_names_out()
            ),
            index=X.index
        )
        self.model = self.__build_model(
            processed_stat_data.shape[1],
            (self.window_size, len(self.dynamic_columns))
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer,
            loss={f'out_{i}': tf.keras.losses.BinaryCrossentropy() for i in range(0, self.num_binary_classes)},
            metrics={f'out_{i}':[self.get_f1]
                        for i in range(0, self.num_binary_classes)}
        )

        if eval_data is None:
            validation_data = None
        else:
            dinam_eval_x = self.__group_dinam_facts(eval_data[0])
            stat_eval_x = self.preprocessed_pipeline.transform(eval_data[0][self.stat_columns])
            eval_y = {f'out_{i}': eval_data[1][i] for i in range(0, self.num_binary_classes)}
            validation_data = (
                [
                    stat_eval_x,
                    dinam_eval_x.reshape((-1, self.window_size, len(self.dynamic_columns)))
                ],
                eval_y
            )

        self.model.fit(
            x=[processed_stat_data, dinam_data.reshape((-1, self.window_size, len(self.dynamic_columns)))],
            y={f'out_{i}': y[i] for i in range(0, self.num_binary_classes)},
            epochs=epochs,
            validation_data=validation_data,
            callbacks=tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        )

    def predict(self, X: pd.DataFrame):
        dinam_eval_x = self.__group_dinam_facts(X)
        stat_eval_x = self.preprocessed_pipeline.transform(X[self.stat_columns])
        encoded_predictions = self.model.predict(
            [
                stat_eval_x,
                dinam_eval_x.reshape((-1, self.window_size, len(self.dynamic_columns)))
            ]
        )
        result = np.empty(shape=(self.num_binary_classes, dinam_eval_x.shape[0]))
        if self.num_binary_classes == 1:
            result[0] = np.argmax(encoded_predictions, axis=1)
            return result

        for index, class_prediction in enumerate(encoded_predictions):
            result[index] = np.argmax(class_prediction, axis=1)
        return result
