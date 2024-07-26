from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt
data = np.load('10新能源电站.npy')[:,0].reshape(-1,24) #数据读取
scaler = MinMaxScaler(feature_range=(-1,1)) #归一化
sca_data = scaler.fit_transform(data)
data = sca_data
ap = AffinityPropagation().fit(data) # ap 聚类
## af = AffinityPropagation(preference=-50).fit(X) perference是参考度 参考度初始值影响聚类效果
labels = ap.labels_ #标签
labels_predict = ap.predict(data) #对新的数据区分到对应聚类簇
cluster_centers_indices = ap.cluster_centers_indices_ #聚类中心
davies_bouldin_score(data, labels) #指标评价