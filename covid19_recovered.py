import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import interpolate
# import seaborn as sns


def get_korea_data(action_data):
    data_korea = action_data[action_data['Country/Region'] == 'Korea, South']
    data_korea = data_korea.drop(columns=['Province/State', 'Country/Region', 'Intermediate Region Code'])
    val_korea = data_korea['Value'].to_numpy().astype(int)
    val_korea = np.flipud(val_korea[val_korea != 0])
    plt.plot(val_korea)
    plt.xlabel("Days")
    plt.ylabel("Confirmed cases")
    plt.title("Korea")
    plt.show()
    return val_korea


def get_germany_data(action_data):
    data_germany = action_data[action_data['Country/Region'] == 'Germany']
    value_germany = data_germany['Value'].to_numpy().astype(int)
    value_germany = np.flipud(value_germany[value_germany != 0])
    plt.plot(value_germany)
    plt.xlabel("Days")
    plt.ylabel("Confirmed cases")
    plt.title("Germany")
    plt.show()
    return value_germany


def get_italy_data(action_data):
    data_italy = action_data[action_data['Country/Region'] == 'Italy']
    value_italy = data_italy['Value'].to_numpy().astype(int)
    value_italy = np.flipud(value_italy[value_italy != 0])
    plt.plot(value_italy)
    plt.xlabel("Days")
    plt.ylabel("Confirmed cases")
    plt.title("Italy")
    plt.show()
    return value_italy


def get_spain_data(action_data):
    data_spain = action_data[action_data['Country/Region'] == 'Spain']
    value_spain = data_spain['Value'].to_numpy().astype(int)
    value_spain = np.flipud(value_spain[value_spain != 0])
    plt.plot(value_spain)
    plt.xlabel("Days")
    plt.ylabel("Confirmed cases")
    plt.title("Spain")
    plt.show()
    return value_spain


def get_france_data(action_data):
    data_france = action_data[action_data['Country/Region'] == 'France']
    value_france = data_france['Value'].to_numpy().astype(int)

    array_france = np.zeros([80], dtype=int)
    for i in range(11):
        array_france = array_france + value_france[80 * i:80 * (i + 1)].astype(int)

    array_france = np.flipud(array_france[array_france != 0])
    plt.plot(array_france)
    plt.xlabel("Days")
    plt.ylabel("Confirmed cases")
    plt.title("France")
    plt.show()
    return array_france


def get_us_data(action_data):
    data_us = action_data[action_data['Country/Region'] == 'US']
    value_us = data_us['Value'].to_numpy().astype(int)
    value_us = np.flipud(value_us[value_us != 0])
    plt.plot(value_us)
    plt.xlabel("Days")
    plt.ylabel("Confirmed cases")
    plt.title("USA")
    plt.show()
    return value_us


def get_japan_data(action_data):
    data_japan = action_data[action_data['Country/Region'] == 'Japan']
    value_japan = data_japan['Value'].to_numpy().astype(int)
    value_japan = np.flipud(value_japan[value_japan != 0])
    plt.plot(value_japan)
    plt.xlabel("Days")
    plt.ylabel("Confirmed cases")
    plt.title("Japan")
    plt.show()
    return value_japan


def get_swiss_data(action_data):
    data_swiss = action_data[action_data['Country/Region'] == 'Switzerland']
    value_swiss = data_swiss['Value'].to_numpy().astype(int)
    value_swiss = np.flipud(value_swiss[value_swiss != 0])
    plt.plot(value_swiss)
    plt.xlabel("Days")
    plt.ylabel("Confirmed cases")
    plt.title("Swiss")
    plt.show()
    return value_swiss


# get_italy_data(all_data)






