from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px


def L_t_SNE_3D(X, y):
  tsne = TSNE(n_components=3).fit_transform(X)# n_components is the dimension number
  
  fig = px.scatter_3d(tsne, x=0, y=1, z=2,
              color=y[0] ) # 3d plot
  fig.update_traces(marker=dict(size=3,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
  fig.show() 

def L_t_SNE_2D(X, y):
  tsne = TSNE(n_components=3).fit_transform(X)# n_components is the dimension number
  
  plt.scatter(tsne[:,0], tsne[:,1], c=y[0]) # 2d plot