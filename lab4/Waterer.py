import numpy as np
import inference_engine

"""
weights = {
    "tap_watering_threshold": ,
    "add_water": ,
}
"""

class Waterer:
    def __init__(self, tap_water_avaliable, rain_collector_area, water_capacity, weights):
        self.tap_water_avaliable = tap_water_avaliable
        self.rain_collector_area = rain_collector_area
        self.water_capacity = water_capacity
        self.water = 0
        self.weights = weights

    def daily_update(self, rainfall, garden):
        if self.tap_water_avaliable:
            if self.water_capacity > 0:
                pass
            else:
                for plant_index in range(len(garden.plants)):
                    if inference_engine.watering_importance(plant_index, garden) > self.weights["tap_watering_threshold"]:
                        garden.soil_wetness[plant_index] += self.weights["add_water"]
                        garden.soil_wetness[plant_index] = min(garden.soil_wetness[plant_index], 1)

        else: # only rainwater
            self.water = min(self.water_capacity, self.water+self.rain_collector_area*rainfall)

        