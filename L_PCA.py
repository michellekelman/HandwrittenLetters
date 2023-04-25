from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib as plt
def L_PCA_3D(X, y):
  pca = PCA(n_components=3).fit(X).transform(X) # n_components is the dimension 
  fig = px.scatter_3d(pca, x=0, y=1, z=2,
              color=y[0])
  fig.update_traces(marker=dict(size=2,
                              opacity = 0.5,
                              ),
                  selector=dict(mode='markers'))
  fig.show() # This is for 3d graph

def L_PCA_3D(X, y):
  pca = PCA(n_components=2).fit(X).transform(X)
  plt.scatter(pca[:,0], pca[:,1], c=y[0]) # 2d plot