#Joint work with Pratyush Kaware
#https://lvdmaaten.github.io/publications/papers/AISTATS_2009.pdf
#https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold._t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px




class t_SNE():
    MACHINE_EPSILON = np.finfo(np.double).eps
    n_components = 2
    perplexity = 30
    def kl_divergence(self, q, P, degrees_of_freedom, n_samples, n_components):
        X_embedded = q.reshape(n_samples, n_components)

        # ||xi-xj||^2 for all unique combinations
        dist = pdist(X_embedded, "sqeuclidean")

        # 1+(||yi-yj||^2)*(1/v)
        dist /= degrees_of_freedom
        dist += 1.

        # (1+(||yi-yj||^2)*(1/v))^-(v+1)/2
        dist **= (degrees_of_freedom + 1.0) / -2.0

        machine_min = np.finfo(np.double).eps
        # Calculates Qij
        Q = np.maximum(dist / (2.0 * np.sum(dist)), machine_min)

        # Kullback-Leibler divergence of P and Q
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, machine_min) / Q))

        # For all pairs xi, xj
        PQd = squareform((P - Q) * dist)

        grad = np.ndarray((n_samples, n_components), dtype=q.dtype)
        for i in range(n_samples):
            # Setting order of operations to K is the fastest.
            grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                             X_embedded[i] - X_embedded)
        grad = grad.ravel()
        c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
        grad *= c

        return kl_divergence, grad

    def gradient_descent(self, Q_, model_features, it=0, n_iter=1000, momentum=0.8, learning_rate=200.0, min_gain=0.01,
                         min_grad_norm=1e-7):

        q = Q_.copy().ravel()
        # Initializing arrays with same size as q.
        update = np.zeros_like(q)
        gains = np.ones_like(q)

        error = np.finfo(float).max
        best_error = np.finfo(float).max
        best_iter = i = it

        for i in range(it, n_iter):
            error, grad = self.kl_divergence(q, *model_features)

            # Length of gradient vector
            grad_norm = linalg.norm(grad)

            # Checking how many points need to increment and decrement
            inc = update * grad < 0.0
            dec = np.invert(inc)

            # Setting amount change according to default values.
            gains[inc] += 0.2
            gains[dec] *= 0.8

            # Making sure its between min_gain and inf, so some change occurs.
            np.clip(gains, min_gain, np.inf, out=gains)

            # Multiply each value in the gradients with the gains.
            grad *= gains
            update = momentum * update - learning_rate * grad
            q += update
            print("***** %d ***** error = %.7f*****,"
                  " *****gradient norm = %.7f*****"
                  % (i + 1, error, grad_norm))

            if error < best_error:
                best_error = error
                best_iter = i

            if grad_norm <= min_grad_norm:
                break
        return q

    def fit(self,X, n_components=2):
        n_samples = X.shape[0]

        # Compute euclidean distance, squared form means whole matrix d(x1,x2) == d(x2,x1)
        distances = pairwise_distances(X, metric='euclidean', squared=True)

        # Compute joint probabilities p_ij from distances.
        P = _joint_probabilities(
            distances=distances, desired_perplexity=30, verbose=False)

        # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
        X_embedded = 1e-4 * \
            np.random.mtrand._rand.randn(
                n_samples, n_components).astype(np.float32)

        # degrees_of_freedom = n_components - 1
        degrees_of_freedom = max(n_components - 1, 1)

        X_embedded = X_embedded.ravel()

        X_embedded = self.gradient_descent(X_embedded, [P, degrees_of_freedom, n_samples, n_components])

        X_embedded = X_embedded.reshape(n_samples, n_components)

        return X_embedded

    def plot_3D(self, X, y):
        fig = px.scatter_3d(X, x=0, y=1, z=2,
                            color= y[0][:10000],)

        fig.update_traces(marker=dict(size=3,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))

    def plot_2D(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y[0][:10000])
