import tensorflow as tf
from tensorflow import keras

train_dataset = keras.preprocessing.text_dataset_from_directory(
    "my_data_dir/train",
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=1337,
)

test_dataset = keras.preprocessing.text_dataset_from_directory(
    "my_data_dir/train",
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)

max_features = 20000
maxlen = 400

tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_dataset)

x_train = tokenizer.texts_to_sequences(train_dataset)
x_test = tokenizer.texts_to_sequences(test_dataset)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = train_dataset.labels
y_test = test_dataset.labels

model = keras.Sequential([
    keras.layers.Embedding(max_features, 128),
    keras.layers.LSTM(64, dropout=0.1, recurrent_dropout=0.5, return_sequences=True),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
