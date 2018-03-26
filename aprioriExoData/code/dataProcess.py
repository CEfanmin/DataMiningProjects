import numpy as np
import pandas as pd
'''
# load data
source_data = pd.read_csv('../data/exo_sample_data_with_targets.csv')

# columns normalization
def NormalData(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d-MIN) / (MAX-MIN)).tolist()  # ratio
    return newDataFrame

normal_data = NormalData(source_data)
# print(normal_data)
# normal_data.to_csv('../data/normal_data.csv')
'''
normal_data = pd.read_csv('../data/normal_data.csv')


def Discretion(df):
    new_data = []
    columns = df.columns.tolist()
    bins = [[0, 0.353402, 0.434448286, 0.520907257, 1],
            [0, 0.296393009, 0.537042603, 0.710138647, 1],
            [0, 0.265911005, 0.583725591, 0.826025587, 1],
            [0, 0.121382296, 0.345748298, 0.602592961, 1],
            [0, 0.188141787, 0.444103392, 0.685365825, 1],
            [0, 0.423882365, 0.82370812, 0.938094281, 1],
            [0, 0.428508489, 0.8268283, 0.938447358, 1],
            [0, 0.151034791, 0.371497124, 0.729663929, 1],
            [0, 0.144629214, 0.371539636, 0.741076602, 1]]

    labels = [['A1','A2','A3','A4'],
              ['B1','B2','B3','B4'],
              ['C1','C2','C3','C4'],
              ['D1','D2','D3','D4'],
              ['E1','E2','E3','E4'],
              ['F1','F2','F3','F4'],
              ['G1','G2','G3','G4'],
              ['H1','H2','H3','H4'],
              ['I1','I2','I3','I4']]

    for c, bi, label, in zip(columns, bins, labels):
        column = df[c]
        ser = pd.Series(np.array(column))
        cats = pd.cut(ser, bi, labels=label)
        # print(cats)
        new_data.append(cats)
    return new_data

new_data = Discretion(normal_data)
pd.DataFrame(new_data).T.to_csv('../data/new_data.csv')
