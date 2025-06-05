import numpy as np

class Waterer:
    def __init__(self, water_price, rain_collector_area, water_capacity, weights):
        self.water_price = water_price
        self.rain_collector_area = rain_collector_area
        self.water_capacity = water_capacity
        self.water = 0

    def daily_update(self, rainfall, garden):
        self.water = min(self.water_capacity, self.water+self.rain_collector_area*rainfall)

        