import numpy as np
import matplotlib.pyplot as plt
import fuzzifier

def general_watering_rule(plant_index, garden):
    return general_watering_rule_direct(garden.soil_wetness[plant_index])

def general_watering_rule_direct(soil_wetness):
    plant_soil_wetness_set = fuzzifier.soil_wetness(soil_wetness)
    value = np.sum(plant_soil_wetness_set[0:2], axis = 0)
    return np.array([np.clip(value, 0, 1)])

def general_do_not_water_rule(plant_index, garden):
    return general_do_not_water_rule_direct(garden.plants[plant_index].days_since_overwatering)

def general_do_not_water_rule_direct(days_since_overwatering):
    do_not_water_set = fuzzifier.days_since_too_much_water(days_since_overwatering)[0]
    return np.array([do_not_water_set])

# twarc -- tap water and rain collector

def twarc_rule(tank_level):
    tank_level_set = fuzzifier.tank_fullness(tank_level)
    use_tap_water = tank_level_set[0]
    use_rain_water = 1-use_tap_water
    empty_tank_to_the_most_dry_plant = tank_level_set[2]
    return np.array([use_tap_water, use_rain_water, empty_tank_to_the_most_dry_plant])

def desert_water_usage_rule(tank_level, garden):
    return desert_water_usage_rule_direct(tank_level, garden.days_since_rain)

def desert_water_usage_rule_direct(tank_level, days_since_rain):
    tank_level_set = np.asarray(fuzzifier.tank_fullness(tank_level))
    rainfall_recency = fuzzifier.days_since_rainfall(days_since_rain)[0]
    rainfall_deficit = 1 - rainfall_recency
    conservative_usage = tank_level_set[0] * (1 - rainfall_recency)
    normal_usage = np.clip(tank_level_set[1] + tank_level_set[0] * rainfall_deficit, 0, 1)
    generous_usage = tank_level_set[2]
    return np.array([conservative_usage, normal_usage, generous_usage])

space = np.linspace(0, 1, 101)
plt.plot(space, general_watering_rule_direct(space)[0], label = "Необхідність поливу рослини")
plt.title("Необхідність поливу рослини")
plt.xlabel("Вологість грунту")
plt.ylabel("Знач. функ. приналежності")
plt.legend()
plt.show()

space = np.linspace(0, 10, 11)
plt.plot(space, general_do_not_water_rule_direct(space)[0], label = "Не поливати рослину")
plt.title("Не поливати рослину")
plt.xlabel("Днів після зависокої вологості")
plt.ylabel("Знач. функ. приналежності")
plt.legend()
plt.show()


space = np.linspace(0, 1, 101)
twarc = twarc_rule(space)

plt.plot(space, twarc[0], label = "Використовувати водогін")
plt.plot(space, twarc[1], label = "Використовувати дощову воду")
plt.plot(space, twarc[2], label = "Зливати дощову воду в найсухішу рослину")
plt.title("Не поливати рослину")
plt.xlabel("Наповнення ємності")
plt.ylabel("Знач. функ. приналежності")
plt.legend()
plt.show()

for day in range(0, 11):
    data = desert_water_usage_rule_direct(space, day)[0]
    plt.plot(space, data, label = f"Дощ був {day} днів тому")
plt.title("Економити воду")
plt.xlabel("Наповнення ємності")
plt.ylabel("Знач. функ. приналежності")
plt.legend()
plt.show()

for day in range(0, 11):
    data = desert_water_usage_rule_direct(space, day)[1]
    plt.plot(space, data, label = f"Дощ був {day} днів тому")
plt.title("Нормальне використання води")
plt.xlabel("Наповнення ємності")
plt.ylabel("Знач. функ. приналежності")
plt.legend()
plt.show()

for day in range(0, 11):
    data = desert_water_usage_rule_direct(space, day)[2]
    plt.plot(space, data, label = f"Дощ був {day} днів тому")
plt.title("Злив води в сухі рослини")
plt.xlabel("Наповнення ємності")
plt.ylabel("Знач. функ. приналежності")
plt.legend()
plt.show()