import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

from Garden import Garden
from Plant import Plant
from Waterer import Waterer
from Weather import Weather

# Лише погода
random.seed(0)
w = Weather(
    0,
    np.array([-5, 15, 27, 15]),
    np.array([0.40, 0.60, 0.50, 0.45]),
    np.array([5, 40, 30, 20]),
)

for i in tqdm(range(365), desc = "тест погоди"):
    w.daily_update(0)

plt.plot(w.temperature, label="температура C")
plt.plot(np.array(w.moisture)*100, label = "вологість %")
plt.plot(w.rainfall, label = "опади мм")
plt.legend()
plt.show()

# Погода і рослини
random.seed(0)
w = Weather(
    0,
    np.array([-5, 15, 27, 15]),
    np.array([0.40, 0.60, 0.50, 0.45]),
    np.array([5, 40, 30, 20]),
)
g = Garden(
    [Plant(1, 1.5, 0.85), Plant(2, 1.5, 0.85), Plant(10, 1.5, 0.85)],
    np.array([0.25, 1, 2]),
    np.array([[0.9, 0.05, 0.05],
              [0.05, 0.9, 0.05],
              [0.05, 0.05, 0.09]])
)
water_delta = 0
for j in tqdm(range(365), desc = "тест саду без поливу"):
    w.daily_update(water_delta)
    g.daily_update(w.get_season(), w.temperature[-1], w.moisture[-1], w.rainfall[-1])

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Сад без поливу", fontsize = 20)
axs[0].plot(w.temperature, label="Температура (°C)")
axs[0].plot(np.array(w.moisture) * 100, label="Вологість (%)")
axs[0].legend()
axs[0].set_title("Температура і вологість")

axs[1].plot(w.rainfall, label="Опади (mm)")
axs[1].plot(np.array(g.soil_wetness_history) * 100, label="Вологість ґрунту (%)")
axs[1].legend()
axs[1].set_title("Опади і вологість ґрунту")

average_health = np.mean([plant.health for plant in g.plants], axis=0)
average_water = np.mean([np.array(plant.stored_water) / plant.water_capacity() for plant in g.plants], axis=0)

axs[2].plot(average_health * 100, label="Здоров'я саду")
axs[2].plot(average_water * 100, label="Запас води в рослинах")
axs[2].legend()
axs[2].set_title("Середні здоров'я і запас води рослин")

plt.tight_layout()
plt.show()
del w
del g
# Погода, рослини і програмака-поливака
random.seed(0)
w = Weather(
    0,
    np.array([-5, 15, 27, 15]),
    np.array([0.40, 0.60, 0.50, 0.45]),
    np.array([5, 40, 30, 20]),
)
g = Garden(
    [Plant(1, 1.5, 0.85), Plant(2, 1.5, 0.85), Plant(10, 1.5, 0.85)],
    np.array([0.25, 1, 2]),
    np.array([[0.9, 0.05, 0.05],
              [0.05, 0.9, 0.05],
              [0.05, 0.05, 0.09]])
)
waterer = Waterer(True, 0, 0, {
    "tap_watering_threshold": 0.75,
    "add_water": 0.1,
})

water_delta = 0
for j in tqdm(range(365), desc = "тест саду з поливом з водогону"):
    w.daily_update(water_delta)
    waterer.daily_update(w.rainfall[-1], g)
    g.daily_update(w.get_season(), w.temperature[-1], w.moisture[-1], w.rainfall[-1])


fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Сад з водогоном", fontsize = 20)
axs[0].plot(w.temperature, label="Температура (°C)")
axs[0].plot(np.array(w.moisture) * 100, label="Вологість (%)")
axs[0].legend()
axs[0].set_title("Температура і вологість")

axs[1].plot(w.rainfall, label="Опади (mm)")
axs[1].plot(np.array(g.soil_wetness_history) * 100, label="Вологість ґрунту (%)")
axs[1].legend()
axs[1].set_title("Опади і вологість ґрунту")

average_health = np.mean([plant.health for plant in g.plants], axis=0)
average_water = np.mean([np.array(plant.stored_water) / plant.water_capacity() for plant in g.plants], axis=0)

axs[2].plot(average_health * 100, label="Здоров'я саду")
axs[2].plot(average_water * 100, label="Запас води в рослинах")
axs[2].legend()
axs[2].set_title("Середні здоров'я і запас води рослин")

axs[3].plot(waterer.tap_usage_history, label = "Використання водогону, л")
axs[3].legend()
axs[3].set_title("Використання води")

plt.tight_layout()
plt.show()


# Погода, рослини і програмака-поливака з дощем
random.seed(0)
w = Weather(
    0,
    np.array([-5, 15, 27, 15]),
    np.array([0.40, 0.60, 0.50, 0.45]),
    np.array([5, 40, 30, 20]),
)
g = Garden(
    [Plant(1, 1.5, 0.85), Plant(2, 1.5, 0.85), Plant(10, 1.5, 0.85)],
    np.array([0.25, 1, 2]),
    np.array([[0.9, 0.05, 0.05],
              [0.05, 0.9, 0.05],
              [0.05, 0.05, 0.09]])
)
waterer = Waterer(True, 10, 2000, {
    "tap_watering_threshold": 0.75,
    "add_water": 0.1,
})

water_delta = 0
for j in tqdm(range(365), desc = "тест саду з поливом з водогону"):
    w.daily_update(water_delta)
    waterer.daily_update(w.rainfall[-1], g)
    g.daily_update(w.get_season(), w.temperature[-1], w.moisture[-1], w.rainfall[-1])


fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Сад з водогоном і збирачем дощу", fontsize = 20)
axs[0].plot(w.temperature, label="Температура (°C)")
axs[0].plot(np.array(w.moisture) * 100, label="Вологість (%)")
axs[0].legend()
axs[0].set_title("Температура і вологість")

axs[1].plot(w.rainfall, label="Опади (mm)")
axs[1].plot(np.array(g.soil_wetness_history) * 100, label="Вологість ґрунту (%)")
axs[1].legend()
axs[1].set_title("Опади і вологість ґрунту")

average_health = np.mean([plant.health for plant in g.plants], axis=0)
average_water = np.mean([np.array(plant.stored_water) / plant.water_capacity() for plant in g.plants], axis=0)

axs[2].plot(average_health * 100, label="Здоров'я саду")
axs[2].plot(average_water * 100, label="Запас води в рослинах")
axs[2].legend()
axs[2].set_title("Середні здоров'я і запас води рослин")

axs[3].plot(waterer.tap_usage_history, label = "Використання водогону, л")
axs[3].plot(waterer.rain_water_usage_history, label = "Використання дощової води, л")
axs[3].plot(np.array(waterer.water)*100/waterer.water_capacity, label = "Накопичено, %")
axs[3].legend()
axs[3].set_title("Використання води")

plt.tight_layout()
plt.show()

# Погода, рослини і програмака-поливака без водогону
random.seed(0)
w = Weather(
    0,
    np.array([-5, 15, 27, 15]),
    np.array([0.40, 0.60, 0.50, 0.45]),
    np.array([5, 40, 30, 20]),
)
g = Garden(
    [Plant(1, 1.5, 0.85), Plant(2, 1.5, 0.85), Plant(10, 1.5, 0.85)],
    np.array([0.25, 1, 2]),
    np.array([[0.9, 0.05, 0.05],
              [0.05, 0.9, 0.05],
              [0.05, 0.05, 0.09]])
)
waterer = Waterer(False, 10, 2000, {
    "tap_watering_threshold": 0.75,
    "add_water": 0.1,
    "rainwater_conservation_mult": 0.25,
})

water_delta = 0
for j in tqdm(range(365), desc = "тест саду з поливом з водогону"):
    w.daily_update(water_delta)
    waterer.daily_update(w.rainfall[-1], g)
    g.daily_update(w.get_season(), w.temperature[-1], w.moisture[-1], w.rainfall[-1])


fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Сад зі збирачем дощу, але без водогону", fontsize = 20)
axs[0].plot(w.temperature, label="Температура (°C)")
axs[0].plot(np.array(w.moisture) * 100, label="Вологість (%)")
axs[0].legend()
axs[0].set_title("Температура і вологість")

axs[1].plot(w.rainfall, label="Опади (mm)")
axs[1].plot(np.array(g.soil_wetness_history) * 100, label="Вологість ґрунту (%)")
axs[1].legend()
axs[1].set_title("Опади і вологість ґрунту")

average_health = np.mean([plant.health for plant in g.plants], axis=0)
average_water = np.mean([np.array(plant.stored_water) / plant.water_capacity() for plant in g.plants], axis=0)

axs[2].plot(average_health * 100, label="Здоров'я саду")
axs[2].plot(average_water * 100, label="Запас води в рослинах")
axs[2].legend()
axs[2].set_title("Середні здоров'я і запас води рослин")

axs[3].plot(waterer.rain_water_usage_history, label = "Використання дощової води, л")
axs[3].plot(np.array(waterer.water)*100/waterer.water_capacity, label = "Накопичено, %")
axs[3].legend()
axs[3].set_title("Використання води")

plt.tight_layout()
plt.show()