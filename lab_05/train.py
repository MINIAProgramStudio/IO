import tensorflow
import matplotlib.pyplot as plt
import PSO
from GeneticSolver import GeneticSolver

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

import numpy as np

from PSO import PSOSolver



label_names = [
    "airplane",   # 0
    "automobile", # 1
    "bird",       # 2
    "cat",        # 3
    "deer",       # 4
    "dog",        # 5
    "frog",       # 6
    "horse",      # 7
    "ship",       # 8
    "truck"       # 9
]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')



x_train = x_train / 255
x_test = x_test / 255



x_train.min(), x_train.max()

print(y_train)

y_train_cat = to_categorical(y_train)

y_test_cat = to_categorical(y_test)

x_valid = x_train[40000:]
y_valid_cat = y_train_cat[40000:]

x_train = x_train[:40000]
y_train_cat = y_train_cat[:40000]
def model_fitness(pos):
    MODEL_SAVE_PATH = "models/pos"
    for p in pos:
        MODEL_SAVE_PATH += "_"+str(round(p,5))
    MODEL_SAVE_PATH += ".keras"
    callbacks = [
        tensorflow.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tensorflow.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_freq='epoch'
        )
    ]
    tensorflow.random.set_seed(0)
    print("training", pos)
    model = Sequential()
    model.add(Input(shape=(32,32,3)))
    model.add(Conv2D(int(2**(pos[0]*3+3)), (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(int(2**(pos[1]*3+3)), activation='relu'))
    model.add(Dropout(pos[2]/2))
    model.add(Dense(10, activation='softmax'))


    #model.summary()
    #plot_model(model, show_shapes=True)

    model.compile(optimizer=Adam(learning_rate=1e-4*pos[3]),
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy', TopKCategoricalAccuracy(k=2, name="Top2")])
    model.fit(x_train, y_train_cat, batch_size=int(2**(pos[4]*5+3)), epochs = 20, validation_data=(x_valid, y_valid_cat), callbacks=callbacks, verbose=0)
    result = model.evaluate(x_test, y_test_cat)
    tensorflow.keras.backend.clear_session()
    return result[1]/result[0]
"""
pso_for_model = PSOSolver({
    "a1": 0.2,#acceleration number
    "a2": 0.4,#acceleration number
    "pop_size": 5,#population size
    "dim": 4,#dimensions
    "pos_min": np.zeros((4)),#vector of minimum positions
    "pos_max": np.ones((4)),#vector of maximum positions
    "speed_min": -np.ones((4)),#vector of min speed
    "speed_max": np.ones((4)),#vector of max speed
}, model_fitness, seeking_min=False)

print("solution", pso_for_model.solve(20,True))
exit()"""
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    plt.figure(figsize=(12, 12))
    for i in range(10):
        ax = plt.subplot(5, 2, i + 1)
        plt.imshow(np.array(x_train[i]))
        class_index = int(y_train[i])
        plt.title(label_names[class_index], fontsize=8)
        plt.axis("off")
    plt.show()
    genetic_for_model = GeneticSolver(
        model_fitness,
        10,
        10,
        5,
        np.array([[0,1],[0,1],[0,1],[0,1],[0,1]]),
        0.05,
        0.1,
        seeking_min = False
    )
    genetic_for_model.solve_stats(25, True, show = True, epsilon_timeout=3, epsilon=1e-2)