import numpy as np
import inference_engine
from Garden import SOIL_CAPACITY

"""
weights = {
    "tap_watering_threshold": ,
    "add_water": ,
    "rainwater_conservation_mult": ,
}
"""



class Waterer:
    def __init__(self, tap_water_avaliable, rain_collector_area, water_capacity, weights):
        self.tap_water_avaliable = tap_water_avaliable
        self.rain_collector_area = rain_collector_area
        self.water_capacity = water_capacity
        self.water = [0]
        self.weights = weights
        self.tap_usage_history = [0]
        self.rain_water_usage_history = [0]

    def daily_update(self, rainfall, garden):
        self.water.append(self.water[-1])
        self.tap_usage_history.append(0)
        self.rain_water_usage_history.append(0)
        if self.tap_water_avaliable:
            if self.water_capacity > 0:
                self.water[-1] = min(self.water_capacity, self.water[-1] + self.rain_collector_area * rainfall)
                # generous usage
                while np.argmax(inference_engine.twarc_rule(self.water[-1]/self.water_capacity)) == 2:
                    plants_idndexes = inference_engine.plant_criticality_list(garden)
                    worst_plant_index = plants_idndexes[0]
                    garden.soil_wetness[worst_plant_index] += min(self.weights["add_water"], self.water[-1]/(garden.soil_area[worst_plant_index]*SOIL_CAPACITY))
                    self.water[-1] -= min(self.weights["add_water"]*SOIL_CAPACITY*garden.soil_area[worst_plant_index], self.water[-1])
                    self.rain_water_usage_history[-1] += min(self.weights["add_water"]*SOIL_CAPACITY*garden.soil_area[worst_plant_index], self.water[-1])

                # normal and conservative usage
                for plant_index in range(len(garden.plants)):
                    if inference_engine.watering_importance(plant_index, garden) > self.weights["tap_watering_threshold"]:
                        garden.soil_wetness[plant_index] += self.weights["add_water"]
                        garden.soil_wetness[plant_index] = min(garden.soil_wetness[plant_index], 1)
                        tap_usage_history_delta = self.weights["add_water"]*(garden.soil_area[plant_index]*SOIL_CAPACITY) - min(self.weights["add_water"]*(garden.soil_area[plant_index]*SOIL_CAPACITY), self.water[-1])
                        if np.argmax(inference_engine.twarc_rule(self.water[-1]/self.water_capacity)) == 1:
                            self.rain_water_usage_history[-1] += min(self.weights["add_water"]*(garden.soil_area[plant_index] * SOIL_CAPACITY), self.water[-1])
                            self.water[-1] -= min(self.weights["add_water"]*(garden.soil_area[plant_index]*SOIL_CAPACITY), self.water[-1])
                            self.tap_usage_history[-1] += tap_usage_history_delta
                        else:
                            self.tap_usage_history[-1] += garden.soil_area[plant_index] * SOIL_CAPACITY * self.weights[
                                "add_water"]



            else:
                for plant_index in range(len(garden.plants)):
                    if inference_engine.watering_importance(plant_index, garden) > self.weights["tap_watering_threshold"]:
                        garden.soil_wetness[plant_index] += self.weights["add_water"]
                        garden.soil_wetness[plant_index] = min(garden.soil_wetness[plant_index], 1)
                        self.tap_usage_history[-1] += garden.soil_area[plant_index]*SOIL_CAPACITY*self.weights["add_water"]

        else: # only rainwater
            self.water[-1] = min(self.water_capacity, self.water[-1] + self.rain_collector_area * rainfall)
            # generous usage
            while np.argmax(inference_engine.desert_water_usage_rule(self.water[-1] / self.water_capacity, garden)) == 2:
                plants_idndexes = inference_engine.plant_criticality_list(garden)
                worst_plant_index = plants_idndexes[0]
                garden.soil_wetness[worst_plant_index] += min(self.weights["add_water"], self.water[-1] / (garden.soil_area[worst_plant_index] * SOIL_CAPACITY))
                self.water[-1] -= min(self.weights["add_water"] * SOIL_CAPACITY * garden.soil_area[worst_plant_index], self.water[-1])
                self.rain_water_usage_history[-1] += min(
                    self.weights["add_water"] * SOIL_CAPACITY * garden.soil_area[worst_plant_index], self.water[-1])

            # normal and conservative usage
            for plant_index in inference_engine.plant_criticality_list(garden):
                if inference_engine.watering_importance(plant_index, garden) > self.weights["tap_watering_threshold"]:
                    garden.soil_wetness[plant_index] += self.weights["add_water"]
                    garden.soil_wetness[plant_index] = min(garden.soil_wetness[plant_index], 1)
                    tap_usage_history_delta = self.weights["add_water"] * (garden.soil_area[plant_index] * SOIL_CAPACITY) - min(
                        self.weights["add_water"] * (garden.soil_area[plant_index] * SOIL_CAPACITY), self.water[-1])
                    if np.argmax(inference_engine.desert_water_usage_rule(self.water[-1] / self.water_capacity, garden)) == 1: # normal
                        self.rain_water_usage_history[-1] += min(self.weights["add_water"] * (garden.soil_area[plant_index] * SOIL_CAPACITY), self.water[-1])
                        self.water[-1] -= min(self.weights["add_water"] * (garden.soil_area[plant_index] * SOIL_CAPACITY), self.water[-1])
                        self.tap_usage_history[-1] += tap_usage_history_delta
                    else: #conservative
                        self.rain_water_usage_history[-1] += min(self.weights["add_water"] * (garden.soil_area[plant_index] * SOIL_CAPACITY), self.water[-1]) * self.weights["rainwater_conservation_mult"]
                        self.water[-1] -= min(self.weights["add_water"] * (garden.soil_area[plant_index] * SOIL_CAPACITY), self.water[-1]) * self.weights["rainwater_conservation_mult"]

