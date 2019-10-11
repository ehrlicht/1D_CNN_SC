"""
Process:

Before running the following steps the model was initially fit with random hyper-parameters within the "train" function
in order to establish an approximate best case scenario:

for i in range (20):

        # Random Hyper-parameter Testing

        batch_size = random.choice([8, 16, 32, 64, 128])
        embedding_dims = random.randint(10, 100)
        filters = random.choice([25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500])
        filter_size = 3
        hidden_dims = random.randint(50, 500)
        dropout = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
        lr = random.choice([0.001, 0.0001])

        [training code]

Following the above scenario, hyper-parameter tuning was then applied to the "train" function to define the best
possible parameters:

    # Hyper-parameter Fine-Tuning

    batch_sizes_lst = [250, 500, 1000]
    embedding_dims_lst = range(22, 30, 1)
    filters_lst = range(350, 450, 25)
    filter_size = 3
    hidden_dims_lst = range(300, 400, 25)
    dropout_list = np.arange(0.1, 0.4, 0.1)
    lr_lst = [0.001, 0.0001]

    for batch_size in batch_sizes_lst:
        for embedding_dims in embedding_dims_lst:
            for filters in filters_lst:
                for hidden_dims in hidden_dims_lst:
                    for dropout in dropout_list:
                        for lr in lr_lst:

                            [training code]

Higher batch sizes were used at first to accelerate initial fitting. The resulting tests were then compared
manually to determine the best overall hyper-parameter configurations.

Once the best configuration was found. The batch size was reduced down to 8 and the hyper-parameters were once again
slightly tweaked to improve the model.

A last final pass was then applied with the final hyper-parameters to train the model over 10 iterations for 50 epochs
each with early stopping enabled in order to retrieve optimal weights and accuracy.

Requirements:

- TensorFlow
- Keras
- NumPy
"""

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy as np

# Training, Validation & Test set creation

max_words = 5000
maxlen = 400

(train_data, train_labels), (val_data, val_labels) = imdb.load_data(
    num_words=max_words,
    skip_top=0,
    seed=1753,
    start_char=1,
    oov_char=2,
    index_from=3)

val_data, test_data = np.split(val_data, 2)
val_labels, test_labels = np.split(val_labels, 2)

# Pad word sequences to same length

train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
val_data = sequence.pad_sequences(val_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)


def train(train_data, train_labels, val_data, val_labels, file_name):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import optimizers, callbacks
    from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Hyper-parameter settings

    batch_size = 32
    embedding_dims = 22
    filters = 425
    filter_size = 3
    hidden_dims = 325
    dropout = 0.4
    lr = 0.0001
    counter = 0

    # Model training for 10 iterations (50 epochs each, with early-stopping enabled) in order to extract the best
    # possible weights for optimal accuracy.

    for i in range(10):
        counter += 1
        NAME = 'mw-{}-ml-{}-bs-{}-ed-{}-fc-{}-fs-{}-hd-{}-do-{}-lr-{}-{}'.format(max_words,
                                                                                 maxlen,
                                                                                 batch_size,
                                                                                 embedding_dims,
                                                                                 filters,
                                                                                 filter_size,
                                                                                 hidden_dims,
                                                                                 dropout,
                                                                                 lr,
                                                                                 int(
                                                                                     time.time()))
        callback_list = [
            callbacks.TensorBoard(
                log_dir='logs\{}'.format(NAME),
            ),
            callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1,
                                    mode='auto', restore_best_weights=True),
        ]

        # Model definition

        model = Sequential()

        model.add(Embedding(max_words, embedding_dims, input_length=maxlen))
        model.add(Dropout(dropout))

        model.add(Conv1D(filters, filter_size, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Conv1D(filters, filter_size, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Conv1D(filters, filter_size, activation='relu'))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(hidden_dims, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizers.Adam(lr=lr), loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(train_data, train_labels, epochs=50, batch_size=batch_size,
                            validation_data=(val_data, val_labels), callbacks=callback_list)

        # Plots the training and validation for easy model appreciation

        sns.set()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('{}-{} model accuracy'.format(counter, file_name))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(r"./plots/{}-{}_accuracy.png".format(counter, file_name))
        plt.show()


        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('{}-{} model loss'.format(counter, file_name))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(r"./plots/{}-{}_loss.png".format(counter, file_name))
        plt.show()


        val_loss, val_accuracy = model.evaluate(val_data, val_labels)

        print(val_loss, val_accuracy)

        # Save the model as reusable file in the specified location

        model.save(r"./models/{}-{}-{}.h5".format(counter, file_name, int(time.time())))


# Predicts accuracy over 100 random samples. to print samples, pass True as a parameter (False if not)

def predict(test_data, test_labels, file_name, print_samples):
    import tensorflow.keras.models as models
    import random

    # Restore Text from encoded Keras IMDB dataset (credit: https://bit.ly/2ViEB7G)

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3
    id_to_word = {value: key for key, value in word_to_id.items()}

    # Load model from its save location

    model = models.load_model(r"./models/{}.h5".format(file_name))

    # Predict a random sample review from the test_data dataset

    results = []
    accuracy = 0

    for i in range(100):
        random_sample = random.randint(0, len(test_data) - 1)

        test_prediction = model.predict(test_data)
        test_review_text = ' '.join(id_to_word[id] for id in np.trim_zeros(test_data[random_sample]))

        if print_samples:
            print('X = {}<END>'.format(test_review_text))
            print('Predicted = {}'.format(test_prediction[random_sample]))
            print('Label = {}'.format(test_labels[random_sample]))

        if (test_prediction[random_sample] > 0.5 and test_labels[random_sample] == 1) or \
                (test_prediction[random_sample] < 0.5 and test_labels[random_sample] == 0):
            results.append(1)
        else:
            results.append(0)

    accuracy = sum(results)
    print('Accuracy: {}%'.format(accuracy))


# %%

# Run once to train the model and save it (you need to specify the save location you'd like to use).
# If a model has already been saved to "./models", skip this step and run the "predict" function directly.

train(train_data, train_labels, val_data, val_labels, '1D_CNN_SC')

# %%

# Run to analyse sentiment for a random batch of 100 IMDB reviews.
# Returns sentiment score and overall batch prediction accuracy.

for i in range(10):
    predict(test_data, test_labels, '7-1D_CNN_SC-1570661370', True)
