from random import random

from Garden import SOIL_CAPACITY


class Plant:
    def __init__(self, mass, normal_daily_water_consumption_per_kg, water_capacity_per_kg):
        self.mass = [mass]
        self.normal_daily_water_consumption = normal_daily_water_consumption_per_kg
        self.water_density = water_capacity_per_kg
        self.stored_water = [(random()*0.5+0.5)*self.water_capacity()]
        self.health = [random()*0.2 + 0.8]
        self.days_since_overwatering = float("inf")

    def daily_update(self, season, soil_moisture, temperature, air_moisture):
        #print(soil_moisture)
        if self.health[-1] <= 0:
            self.health.append(0)
            self.stored_water.append(0)
            self.mass.append(self.mass[-1])
            return 0, 0
        else:
            self.health.append(self.health[-1])
            self.days_since_overwatering += 1

        k = 2-abs(season-2)
        water_deficit = self.normal_daily_water_consumption*self.mass[-1]*k
        #print(water_deficit)
        if air_moisture < 0.3:
            water_deficit *= 0.3/(0.3-air_moisture)
        water_deficit *= max((temperature/17), 0.1)
        #print(water_deficit)
        water_desired_proficit = self.water_capacity()-self.stored_water[-1] + self.normal_daily_water_consumption*self.mass[-1] + water_deficit
        #print(soil_moisture)
        water_avaliable_from_soil = soil_moisture*SOIL_CAPACITY/2
        #print(water_avaliable_from_soil)
        #print(water_desired_proficit)
        water_proficit = min(water_avaliable_from_soil, water_desired_proficit)

        self.stored_water.append(self.stored_water[-1] + water_proficit - water_deficit)
        self.stored_water[-1] = min(self.stored_water[-1], self.water_capacity())
        self.stored_water[-1] = max(0, self.stored_water[-1])
        self.health[-1] += 0.025
        if self.stored_water[-1] < self.water_capacity()*0.1:
            self.health[-1] -= 0.05
        if soil_moisture > 0.9:
            self.health[-1] -= 0.05
            self.days_since_overwatering = 0



        self.health[-1] = min(1, max(0, self.health[-1]))
        return self.stored_water[-1]-self.stored_water[-2]+water_deficit, water_deficit #soil water deficit and moisturised water


    def water_capacity(self):
        return self.water_density*self.mass[-1]