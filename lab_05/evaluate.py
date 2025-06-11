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
listdir = os.listdir(MODEL_DIR)
for filename in tqdm(listdir, desc = "generating data from models"):
    if filename.endswith('.keras'):
        data.append([])
        model_path = os.path.join(MODEL_DIR, filename)
        filename = str(filename)[:filename.index(".keras")]
        filename = filename[filename.index("_") + 1:]
        while "_" in filename:
            data[-1].append(float(filename[:filename.index("_")]))
            filename = filename[filename.index("_")+1:]
        data[-1].append(float(filename))
        #print(f'Loading model: {model_path}')
        model = tensorflow.keras.models.load_model(model_path)
        result = model.evaluate(x_test, y_test_cat, verbose = 0)
        data[-1].append(result[0])
        data[-1].append(result[1])
data = np.array(data)
print(data.shape)



X = data[:, 3]
Y = data[:, 4]
Z = data[:, -2]

# Створюємо регулярну сітку
xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Інтерполяція Z-координат
zi = griddata((X, Y), Z, (xi, yi), method='cubic')

# Побудова 3D-ландшафту
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

# Інтерполюємо Z
zi = griddata((X, Y), Z, (xi, yi), method='cubic')

# Уникаємо log(0)
eps = 1e-8
zi_log = np.log10(np.clip(zi, eps, None))

# Побудова контурної мапи
plt.figure(figsize=(10, 6))
contour = plt.contourf(xi, yi, zi_log, levels=20, cmap='plasma')
plt.scatter(X, Y, color='black', s=10, label='Data points')
plt.colorbar(contour, label='log10(Z)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Logarithmic Contour Map of Z')
plt.tight_layout()
plt.show()