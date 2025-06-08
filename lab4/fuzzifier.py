import numpy as np
import matplotlib.pyplot as plt

def trapmf(x, a, b, c, d):
    x = np.asarray(x)
    return np.maximum(
        np.minimum(
            np.minimum((x - a) / (b - a + 1e-9), 1),
            (d - x) / (d - c + 1e-9)
        ),
        0
    )

def gaussmf(x, dispersion_sq_root, mean):
    return np.exp(-(x-mean)**2/(2*dispersion_sq_root**2))

def soil_wetness(x):
    normal = trapmf(x, 0.3, 0.4, 0.6, 0.7)
    dry = trapmf(x, 0.1, 0.2, 0.3, 0.4)
    very_dry = trapmf(x, -0.1, 0, 0.1, 0.2)
    wet = trapmf(x, 0.6, 0.7, 0.8, 0.9)
    very_wet = trapmf(x, 0.8, 0.9, 1, 1.1)
    return very_dry, dry, normal, wet, very_wet

def tank_fullness(x):
    tank_empty = trapmf(x, -0.1, 0, 0, 0.2)
    tank_normal = trapmf(x, 0, 0.2, 0.8, 1)
    tank_overflowing = trapmf(x, 0.9, 1, 1, 1.1)
    return tank_empty, tank_normal, tank_overflowing

def days_since_rainfall(x):
    rainfall_was_recent = gaussmf(x, 3, 0)
    return rainfall_was_recent

def days_since_too_much_water(x):
    do_not_water = gaussmf(x, 1.5, 0)
    return do_not_water
"""
space = np.linspace(0, 1, 101)
very_dry, dry, normal, wet, very_wet = soil_wetness(space)


plt.title("Вологість грунту")
plt.plot(space, very_dry, label = "Дуже сухий")
plt.plot(space, dry, label = "Сухий")
plt.plot(space, normal, label = "Нормальний")
plt.plot(space, wet, label = "Вологий")
plt.plot(space, very_wet, label = "Дуже вологий")
plt.legend()
plt.xlabel("Вологість")
plt.ylabel("Знач. функ. приналежності")
plt.show()

plt.title("Рівень води в ємності")
plt.xlabel("Рівень води")
plt.ylabel("Знач. функ. приналежності")
tank_empty, tank_normal, tank_overflowing = tank_fullness(space)
plt.plot(space, tank_empty, label = "Ємність пуста")
plt.plot(space, tank_normal, label = "В ємності нормально води")
plt.plot(space, tank_overflowing, label = "Ємність переповнена")
plt.legend()
plt.show()

days = np.linspace(0, 10, 11)
plt.title("Ф. пр. залежні від кількості днів")
plt.xlabel("Днів після події")
plt.ylabel("Знач. функ. приналежності")
rainfall_was_recent = days_since_rainfall(days)
do_not_water = days_since_too_much_water(days)
plt.plot(days, rainfall_was_recent, label = "Нещодавно був дощ")
plt.plot(days, do_not_water, label = "Нещодавно рослина отримала забагато води")
plt.legend()
plt.show()

"""