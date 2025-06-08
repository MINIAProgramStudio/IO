import numpy as np
from random import random

class Weather:
    def __init__(self, start_day, average_temperatures, average_moistures, average_rainfall, rainfall_sparcity = 3):
        self.days = [start_day]
        self.average_temperatures = average_temperatures
        self.average_moistures = average_moistures
        self.average_rainfall = average_rainfall
        self.temperature = [self.average_temperatures[self.get_season()]*(0.9 + random()*0.2)]
        self.moisture = [self.average_moistures[self.get_season()]*(0.9 + random()*0.2)]
        self.rainfall = [0]
        self.rainfall_sparcity = rainfall_sparcity

    def daily_update(self, delta_water):
        self.days.append((self.days[-1]+1)%365)

        if random() < 1/self.rainfall_sparcity:
            self.rainfall.append(self.average_rainfall[self.get_season()] * (self.rainfall_sparcity-0.5 + random()))
        else:
            self.rainfall.append(0)
        self.moisture.append(self.moisture[-1] + delta_water/10**5)
        self.moisture[-1] += (self.average_moistures[self.get_season()] - self.moisture[-1])*(-0.1 + random()*0.4)
        self.moisture[-1] += random()*0.1 - 0.05

        self.temperature.append(self.temperature[-1] - delta_water * 2 / 10 ** 5)
        self.temperature[-1] += (self.average_temperatures[self.get_season()] - self.temperature[-1]) * (
                    -0.1 + random() * 0.4)
        self.temperature[-1] += random() * 5 - 2.5

        return self.get_season(), self.rainfall[-1], self.temperature[-1], self.moisture[-1]

    def get_season(self):
        day = self.days[-1]
        if day < 60:
            return 0
        if day < 152:
            return 1
        if day < 244:
            return 2
        if day < 335:
            return 3
        return 0

