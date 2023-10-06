import numpy as np
import utils


class KMeans:
    def __init__(self, X, K=1, options=None):
        """
        Constructor of KMeans class
            Args:
                K (int): Number of cluster
                options (dict): dictionary with options
        """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self.centroids = np.zeros(shape=(self.K, self.X.shape[1]), dtype=float)
        self.old_centroids = np.zeros(shape=(self.K, self.X.shape[1]), dtype=float)
        self.labels = np.zeros(shape=(self.K, self.X.shape[1]), dtype=float)

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
        Args:
            X (list or np.array): list(matrix) of all pixel values
                if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                the last dimension
        """
        X = np.float64(X)
        if X.ndim == 3 and X.shape[2] == 3:
            X = X.reshape(-1, 3)
        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if "km_init" not in options:
            options["km_init"] = "first"  # first or kmeans++
        if "verbose" not in options:
            options["verbose"] = False
        if "tolerance" not in options:
            options["tolerance"] = 20
        if "max_iter" not in options:
            options["max_iter"] = np.inf
        if "fitting" not in options:
            options["fitting"] = "WCD"  # WCD, ICD or FC

        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options["km_init"].lower() == "first":
            self.centroids = np.zeros(shape=(self.K, self.X.shape[1]), dtype=float)
            i_centroid = 0
            for x in self.X:
                if not np.any(np.all(self.centroids == x, axis=1)):
                    self.centroids[i_centroid] = x
                    i_centroid += 1
                    if i_centroid == self.K:
                        break
        elif self.options["km_init"].lower() == "kmeans++":
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            for _ in range(1, self.k):
                dist_sq = np.array(
                    [
                        min([np.inner(c - x, c - x) for c in self.centroids])
                        for x in self.X
                    ]
                )

                probs = dist_sq / dist_sq.sum()
                cumulative_probs = np.cumsum(probs)
                self.centroids.append(
                    np.random.choice(
                        (np.random.rand(self.K, self.X.shape[1])), cumulative_probs
                    )
                )
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])

    def get_labels(self):
        """Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        self.labels = np.argmin(distance(self.X, self.centroids), axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids.copy()
        for i in range(self.K):
            if (self.labels == i).sum() > 0:
                self.centroids[i] = np.mean(
                    self.X[self.labels == i], axis=0, dtype=float
                )

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(
            self.centroids, self.old_centroids, rtol=self.options["tolerance"]
        )

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        iteration = 0
        self._init_centroids()
        while iteration < self.options["max_iter"]:
            self.get_labels()
            self.get_centroids()
            iteration += 1
            if np.allclose(self.centroids, self.old_centroids):
                break

    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """

        sum = 0
        self.WCD = 0
        for i in range(self.K):
            sum += np.sum(distance(self.X[self.labels == i], self.centroids[i]) ** 2)
        self.WCD = sum / self.X.shape[0]
        return self.WCD

    def interClassDistance(self):
        """
        returns the inter class distance of the current clustering
        """

        icd = 0
        num_centroids = self.centroids.shape[0]
        for i in range(num_centroids):
            for j in range(i + 1, num_centroids):
                icd += np.linalg.norm(self.centroids[i] - self.centroids[j]) ** 2

        icd /= self.X.shape[0]
        return icd

    def fisherCoefficient(self):
        """
        returns the fisher coefficient of the current clustering
        """

        mean_distances = []
        for i in range(self.centroids.shape[0]):
            for j in range(i + 1, self.centroids.shape[0]):
                dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                mean_distances.append(dist)

        mean_distance = np.mean(mean_distances)
        within_class_distance = 0
        for i in range(self.centroids.shape[0]):
            within_class_distance += np.sum(
                np.linalg.norm(self.X[self.labels == i] - self.centroids[i]) ** 2
            )

        fisher_coeff = mean_distance / within_class_distance
        return fisher_coeff

    def find_bestK(self, max_K):
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """
        tol = self.options["tolerance"]
        prevDistance = None
        for i in range(2, max_K, 1):
            distance = None

            self.K = i
            self.fit()

            if self.options["fitting"].lower() == "ICD":
                distance = self.interClassDistance()
            elif self.options["fitting"].lower() == "FC":
                distance = self.fisherCoefficient()
            else:
                # WCD
                distance = self.withinClassDistance()

            if prevDistance is not None:
                dec = (distance / prevDistance) * 100

                if 100 - dec < tol:
                    self.K -= 1
                    break

            prevDistance = distance


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    return np.sqrt(np.sum((X[:, np.newaxis] - C) ** 2, axis=2))


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    return utils.colors[np.argmax(utils.get_color_prob(centroids), axis=1)]
