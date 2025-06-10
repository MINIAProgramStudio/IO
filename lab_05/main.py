import tensorflow
import matplotlib.pyplot as plt
import PSO

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

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

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
plt.figure(figsize=(12, 12))
for i in range(10):
    ax = plt.subplot(5, 2, i + 1)
    plt.imshow(np.array(x_train[i]))
    class_index = int(y_train[i])
    plt.title(label_names[class_index], fontsize=8)
    plt.axis("off")
plt.show()
# The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

print(x_train.shape)
print(x_test.shape)

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

print(x_valid.shape)

print(y_valid_cat.shape)

print(x_train.shape)

print(x_test.shape)

def model_fitness(pos):
    print("training", pos)
    model = Sequential()
    model.add(Input(shape=(32,32,3)))
    model.add(Conv2D(int(2**(pos[0]*4+3)), (int(pos[1]*4)*2+1, int(pos[1]*4)*2+1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(int(2**(pos[2]*5+4)), activation='relu'))
    model.add(Dropout(pos[3]))
    model.add(Dense(100, activation='softmax'))


    #model.summary()
    #plot_model(model, show_shapes=True)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy', TopKCategoricalAccuracy(k=2, name="Top2")])
    running = True
    epochs = 0
    best = 0
    while running:
        history = model.fit(x_train, y_train_cat, batch_size=200, epochs = 5, validation_data=(x_valid, y_valid_cat))
        epochs += 5
        best = max(best, max(np.array(history.history['val_accuracy'])/np.array(history.history['val_loss']) ))
        if epochs >= 25:
            running = False
        if history.history['accuracy'][-1] + history.history['accuracy'][-2] - history.history['val_accuracy'][-1] - history.history['val_accuracy'][-2] > 0.1:
            running = False
    tensorflow.keras.backend.clear_session()
    return best
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
exit()
"""

model = Sequential()
model.add(Input(shape=(32, 32, 3)))
model.add(Conv2D(128, (7, 7), activation='relu',padding='same'))
model.add(MaxPooling2D(2))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(256, (7, 7), activation='relu',padding='same'))
model.add(MaxPooling2D(2))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(512, (7, 7), activation='relu',padding='same'))
model.add(MaxPooling2D(3))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#model = tensorflow.keras.models.load_model('cifar100.keras')

model.summary()
#plot_model(model, show_shapes=True)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy', TopKCategoricalAccuracy(k=2, name="Top2")])

history = model.fit(x_train, y_train_cat, batch_size = 100, epochs = 10, validation_data=(x_valid, y_valid_cat))

model.save("cifar100.keras")

#model = tensorflow.keras.models.load_model("cifar100.keras")
#history = model.fit(x_train, y_train_cat, batch_size=200, epochs = 250, validation_data=(x_valid, y_valid_cat))
#model.save("cifar100.keras")
plt.plot(history.history['loss'], label = "loss")
plt.plot(history.history['val_loss'], label = "val_loss")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = "accuracy")
plt.plot(history.history['val_accuracy'], label = "val_accuracy")
plt.legend()
plt.show()

model.evaluate(x_test, y_test_cat)

predictions = model.predict(x_test)



#Predict
y_prediction_cat = model.predict(x_test)
y_prediction = np.argmax(y_prediction_cat, axis=1)

print(y_test.shape)
print(y_prediction_cat.shape)

print(predictions.shape)
print(predictions[0])


print(np.argmax(predictions[0]))

print(np.argmax(y_test[0]))

print(x_test.shape)

plt.imshow(x_test[0])
plt.title('Class: ' + str(label_names[y_test[0][0]]))
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Створення Confusion Matrix
cm = confusion_matrix(y_test, y_prediction)

# 2. Відображення без чисел
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(include_values=False,  # <- не показувати числа
          xticks_rotation=90,
          cmap='Blues',
          ax=ax)
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()