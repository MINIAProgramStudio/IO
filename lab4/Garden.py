import numpy as np

SOIL_CAPACITY = 2*300 # 1m^3 грунту може утримувати 150-600 літрів води, вважатимемо що ємність 300 л/м^3

class Garden:
    def __init__(self, plants, soil_area, soil_distance_matrix):
        self.plants = plants
        self.soil_distance_matrix = soil_distance_matrix
        self.soil_area = soil_area
        self.soil_wetness = np.random.rand(self.soil_area.shape[0])*0.2 + 0.4
        self.soil_wetness_history = [np.mean(self.soil_wetness)]
        self.days_since_rain = 1

    def daily_update(self, season, temperature, air_moisture, rain_mm):
        if rain_mm > 0 :
            self.days_since_rain = 0
        else:
            self.days_since_rain += 1
        self.soil_wetness += rain_mm*0.001
        #print(self.soil_wetness)
        self.soil_wetness_history.append(np.mean(self.soil_wetness))
        delta = np.zeros((self.soil_wetness.shape[0],2))
        for i in range(len(self.plants)):
            delta[i][0], delta[i][1] = self.plants[i].daily_update(season, self.soil_wetness[i], temperature, air_moisture)
            self.soil_wetness[i] -= delta[i, 0]/(self.soil_area[i]*SOIL_CAPACITY)
        soil_wetness_evaporation_delta = max(temperature*0.0001, 0.00001)*(1+self.soil_wetness_history[-1])**3
        self.soil_wetness -= soil_wetness_evaporation_delta * (1.05-np.random.rand(self.soil_area.shape[0])*0.1)
        self.soil_wetness = np.clip(self.soil_wetness, 0, 1)
        self.soil_wetness = np.matmul(self.soil_distance_matrix, self.soil_wetness)
        #print(self.soil_wetness)

        return np.sum(delta, axis = 0)[1] + soil_wetness_evaporation_delta