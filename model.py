import numpy as np
import pandas as pd

wines = pd.read_csv('data/winemag-data_first150k.csv')
wines.dropna(subset=['price'], inplace=True)

def top50():
    return wines.sort_values('points', ascending=False).head(50)