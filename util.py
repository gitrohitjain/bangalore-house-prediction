import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_pice(bhk,total_sqft,bath,balcony,location):

    x = np.zeros(len(__data_columns))
    x[0] = bhk
    x[1] = total_sqft
    x[2] = bath
    x[3] = balcony

    try:
        loc_index = __data_columns.index(location.lower())
    except:
        x[loc_index] = -1
    

    if loc_index >=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def get_location_names():
    return __locations


def load_saved_artifacts():
    print('....Loading artifacts')
    global __data_columns
    global __locations
    global __model

    with open(r'columns.json', 'r') as f :
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[4:]

    with open(r'banglore_home_prices_model.pickle', 'rb') as f :
        __model = pickle.load(f)
    
    print('...artifacts loaded')
