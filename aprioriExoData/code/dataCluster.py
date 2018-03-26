import pandas as pd
from sklearn.cluster import KMeans


data_file = '../data/normal_data.csv'
cluster_datafile = '../data/data_cluster.csv'
typelabel = {'roll_degree':'A', 'pitch_degree':'B','yaw_degree':'C',
            'left_pressure':'D','right_pressure':'E','left_hip_joint':'F',
            'right_hip_joint':'G','left_knee_joint':'H','right_knee_joint':'I'}

data = pd.read_csv(data_file)
keys = list(typelabel.keys())
result = pd.DataFrame()

if __name__ == '__main__':
    for i in range(len(keys)):
        print('cluster on %s...'%keys[i])
        kmodel = KMeans(n_clusters=4, n_jobs=-1)
        kmodel.fit(data[[keys[i]]])
        r1 = pd.DataFrame(kmodel.cluster_centers_, columns=[typelabel[keys[i]]])
        r2 = pd.Series(kmodel.labels_).value_counts()
        r2 = pd.DataFrame(r2 , columns=[typelabel[keys[i]]+ 'n'])
        r = pd.concat([r1, r2], axis=1).sort_values(typelabel[keys[i]])
        r.index = [1,2,3,4]
        r[typelabel[keys[i]]] = pd.rolling_mean(r[typelabel[keys[i]]], 2)
        r[typelabel[keys[i]]][1] = 0.0
        result = result.append(r.T)
    result.to_csv(cluster_datafile)

