
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, cluster
import scipy
import seaborn as sns
df = pd.read_csv("/Users/Gym/Desktop/research/Benchmark_LTW_20/lrc105.csv")
X = df[['XCOORD.','YCOORD.']]
max_k = 10

distortions = []
for i in range(1, max_k+1):
    if len(X) >= i:
        model = cluster.KMeans(n_clusters=i, init='k-means++', 
        max_iter=300, n_init=10, random_state=0)
        model.fit(X)
        distortions.append(model.inertia_)

k = [i*100 for i in np.diff(distortions, 2)].index(min([i*100 for i in np.diff(distortions, 2)]))

fig, ax = plt.subplots()
ax.plot(range(1, len(distortions)+1), distortions)
ax.axvline(k, ls='--', color='red', label='k= '+str(k))
ax.set(title='The Elbow Method', xlabel='Number of Cluster', ylabel='Distortion')
ax.legend()
ax.grid(True)
# plt.show()

model = cluster.KMeans(n_clusters=k, init='k-means++')
df_X = X.copy()
df_X['cluster'] = model.fit_predict(X)

closest, distances = scipy.cluster.vq.vq(model.cluster_centers_,
                        df_X.drop('cluster', axis=1).values)

df_X['centroids'] = 0
for i in closest:
    df_X['centroids'].iloc[i] = 1

df[['cluster', 'centroids']] = df_X[['cluster', 'centroids']]
# print(df.sample(5))

fix, ax = plt.subplots()
# sns.scatterplot(x='XCOORD.', y='YCOORD.', data= df, palette=sns.color_palette('bright', k),
#                 hue='cluster', size='centroids', size_order=[1,0],
#                 legend='brief', ax=ax).set_title('Clustering (k='+str(k)+')')

# th_centroids = model.cluster_centers_
# ax.scatter(th_centroids[:,0], th_centroids[:,1], s=50, c='black', marker='x')

model = cluster.AffinityPropagation()
k=df['cluster'].nunique()
sns.scatterplot(x='XCOORD.', y='YCOORD.', data= df, palette=sns.color_palette('bright', k),
                hue='cluster', size='centroids', size_order=[1,0],
                legend='brief', ax=ax).set_title('Clustering (k='+str(k)+')')
plt.text(df["XCOORD."][0],df["YCOORD."][0],'Depot: ({}, {})'.format(df["XCOORD."][0],df["YCOORD."][0]))
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()