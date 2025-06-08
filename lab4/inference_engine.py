import numpy as np
import fuzzifier

def general_watering_rule(plant_index, garden):
    plant_soil_wetness_set = fuzzifier.soil_wetness(garden.soil_wetness[plant_index])
    return [min(max(sum(plant_soil_wetness_set[0:2]), 0), 1)]

def general_do_not_water_rule(plant_index, garden):
    do_not_water_set = fuzzifier.days_since_too_much_water(garden.plants[plant_index].days_since_overwatering)
    return [do_not_water_set]

# twarc -- tap water and rain collector

def twarc_rule(tank_level):
    tank_level_set = fuzzifier.tank_fullness(tank_level)
    use_tap_water = tank_level_set[0]
    use_rain_water = 1-use_tap_water
    empty_tank_to_the_most_dry_plant = tank_level_set[2]
    return [use_tap_water, use_rain_water, empty_tank_to_the_most_dry_plant]

def desert_water_usage_rule(tank_level, garden):
    tank_level_set = fuzzifier.tank_fullness(tank_level)
    rainfall_recency = fuzzifier.days_since_rainfall(garden.days_since_rain)[0]
    rainfall_deficit = 1 - rainfall_recency
    normal_usage = max(tank_level_set[1] + tank_level_set[0]*rainfall_deficit, 1)
    conservative_usage = tank_level_set[0] - tank_level_set[0]*rainfall_recency
    generous_usage = tank_level_set[2] - tank_level_set[0]*rainfall_recency
    return [conservative_usage, normal_usage, generous_usage]