import copy

import tensorflow
import matplotlib.pyplot as plt
import os
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
from tqdm import tqdm
from scipy.interpolate import griddata

import numpy as np

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
y_test_cat = to_categorical(y_test)

MODEL_DIR = 'models'
data = []
model_files = [
    os.path.join(MODEL_DIR, f)
    for f in os.listdir(MODEL_DIR)
    if f.endswith('.keras')
]

model_files.sort(key=lambda f: os.path.getctime(f))
for filename in tqdm(model_files, desc = "evaluating models"):
    if filename.endswith('.keras'):
        model_path = copy.copy(filename)
        data.append([])
        filename = str(filename)[:filename.index(".keras")]
        filename = filename[filename.index("_") + 1:]
        while "_" in filename:
            data[-1].append(float(filename[:filename.index("_")]))
            filename = filename[filename.index("_")+1:]
        data[-1].append(float(filename))
        #print(f'Loading model: {model_path}')
        model = tensorflow.keras.models.load_model(model_path)
        result = model.evaluate(x_test, y_test_cat, verbose = 0)
        for i in result:
            data[-1].append(i)
data = np.array(data)

x = range(len(model_files))
plt.plot(x, data[:, 0], label = "Conv2D")
plt.plot(x, data[:, 1], label = "Dense")
plt.plot(x, data[:, 2], label = "Dropout")
plt.plot(x, data[:, 3], label = "learning_rate")
plt.plot(x, data[:, 4], label = "batch_size")
plt.plot(x, data[:, 6], label = "accuracy")
plt.xlabel("Номер моделі")
plt.ylabel("Значення")
plt.title("Нормалізовані значення параметрів і точності")
plt.legend()
plt.show()

print(data.shape)
data[:, 0] = np.floor(2**(data[:, 0]*3+3))
data[:, 1] = np.floor(2**(data[:, 1]*3+3))
data[:, 2] = data[:, 2]/2
data[:, 3] = data[:, 3]*0.0001
data[:, 4] = np.floor(2**(data[:, 4]*5+3))

fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Фактичні значення", fontsize = 20)
axs[0].plot(data[:, 0], label="Conv2D")
axs[0].plot(data[:, 1], label="Dense")
axs[0].set_yscale("log")
axs[0].legend()
axs[0].set_title("Кількість нейронів/фільтрів в шарах")

axs[1].plot(data[:, 2] , label="Dropout")
axs[1].legend()
axs[1].set_title("Значення Dropout")

axs[2].plot(data[:, 3] , label="LR")
axs[2].legend()
axs[2].set_title("Значення learning_rate")

axs[3].plot(data[:, 4] , label="LR")
axs[3].set_yscale("log")
axs[3].legend()
axs[3].set_title("Значення batch_size")

axs[4].plot(data[:,-2], label="Точність")
axs[4].plot(data[:,-1], label="Т2")
axs[4].legend()
axs[4].set_title("Точність та втрати")

axs[5].plot(data[:,-3], label="Втрати")
axs[5].set_yscale("log")
axs[5].legend()
axs[5].set_title("Втрати")

plt.tight_layout()
plt.show()




X = data[:, 3]
Y = data[:, 4]
Z = data[:, -2]


xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 100)
xi, yi = np.meshgrid(xi, yi)


zi = griddata((X, Y), Z, (xi, yi), method='cubic')


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')
ax.scatter(X, Y, Z, color='red', s=10, label='Data points')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Landscape from (X, Y, Z) points')
plt.show()

xi = np.linspace(X.min(), X.max(), 200)
yi = np.linspace(Y.min(), Y.max(), 200)
xi, yi = np.meshgrid(xi, yi)


zi = griddata((X, Y), Z, (xi, yi), method='cubic')

eps = 1e-8
zi_log = np.log10(np.clip(zi, eps, None))


plt.figure(figsize=(10, 6))
contour = plt.contourf(xi, yi, zi_log, levels=20, cmap='plasma')
plt.scatter(X, Y, color='black', s=10, label='Data points')
plt.colorbar(contour, label='log10(Z)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Logarithmic Contour Map of Z')
plt.tight_layout()
plt.show()