import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Garden import Garden
from Plant import Plant
from Weather import Weather

w = Weather(
    0,
    np.array([-5, 15, 27, 15]),
    np.array([0.40, 0.60, 0.50, 0.45]),
    np.array([5, 20, 15, 10]),
)

for i in tqdm(range(365), desc = "тест погоди"):
    w.daily_update(0)

plt.plot(w.temperature, label="температура C")
plt.plot(np.array(w.moisture)*100, label = "вологість %")
plt.plot(w.rainfall, label = "опади мм")
plt.legend()
plt.show()

g = Garden(
    [Plant(1, 0.05, 0.5), Plant(2, 0.05, 0.5), Plant(10, 0.05, 0.5)],
    np.array([2, 4, 10]),
    np.array([[0.9, 0.05, 0.05],
              [0.05, 0.9, 0.05],
              [0.05, 0.05, 0.09]])
)
water_delta = 0
for j in tqdm(range(365), desc = "тест саду без поливу"):
    w.daily_update(water_delta)
    g.daily_update(w.get_season(), w.temperature[-1], w.moisture[-1], w.rainfall[-1])

plt.plot(w.temperature[366:], label="temp C")
plt.plot(np.array(w.moisture[366:])*100, label = "humidity %")


plt.legend()
plt.show()

plt.plot(w.rainfall[366:], label = "rainfall mm")
plt.plot(np.array(g.soil_wetness_history)*100, label = "soil wetness")

plt.legend()
plt.show()

average_health = []
for plant in g.plants:
    average_health.append(plant.health)
average_health = np.mean(np.array(average_health), axis = 0)

average_water = []
for plant in g.plants:
    average_water.append(np.array(plant.stored_water)/plant.water_capacity())
average_water = np.mean(np.array(average_water), axis = 0)

plt.plot(average_health*100, label = "Здоров'я саду")
plt.plot(average_water*100, label = "Запас води в рослинах")
plt.legend()
plt.show()