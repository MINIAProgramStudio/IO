import matplotlib.pyplot as plt
import numpy as np

from Garden import Garden
from Plant import Plant
from Weather import Weather

w = Weather(
    0,
    np.array([-5, 15, 27, 15]),
    np.array([0.40, 0.60, 0.50, 0.45]),
    np.array([5, 30, 20, 10]),
)

for i in range(365):
    w.daily_update(0)

plt.plot(w.temperature, label="temp C")
plt.plot(np.array(w.moisture)*100, label = "humidity %")
plt.plot(w.rainfall, label = "rainfall mm")
plt.legend()
plt.show()

g = Garden(
    [Plant(1, 0.05, 0.5), Plant(2, 0.05, 0.5)],
    np.array([2,4]),
    np.array([[0.9, 0.1],
              [0.1, 0.9]])
)
water_delta = 0
for j in range(365):
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

plt.plot(np.array(g.plants[0].health)*100, label = "plant health")
plt.plot(np.array(g.plants[0].stored_water)*100, label = "plant water")
plt.legend()
plt.show()