import matplotlib as plt
import numpy as np
import plotly.express as px
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class PCA():

    def Pca(self, X, num_components = 2):

        # move the center of the data to the origin
        X_meaned = X - np.mean(X, axis=0)

        # get the covariance matrix
        cov_mat = np.cov(X_meaned, rowvar=False)

        # get the eigen_values, eigen_vectors from the matrix
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        # sort the values, pick the top k (k is the dimension, 2 or 3)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

        # dot product of the vector and the recentered data (project the data to lower dimension)
        X_reduced = np.dot(eigenvector_subset.transpose(),
                           X_meaned.transpose()).transpose()
        return X_reduced

    def plot_3D(self, X, y):
        fig = px.scatter_3d(X, x=0, y=1,
                            color=y[0], )  # 3d plot
        fig.update_traces(marker=dict(size=2,
                                      opacity=0.5,
                                      ),
                          selector=dict(mode='markers'))
        return fig

    def plot_2D(self, X, y):
        fig = plt.scatter(X[:, 0], X[:, 1], c=y[0])  # 2d plot
        return fig