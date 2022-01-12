import collections, numpy as np


beta = 2 # global beta
class matrix:

    def __init__(self, file_name):
        self.file_name = file_name
        self.array_2d = np.array([]) # empty array_2d
        self.start()
    
    def load_from_csv(self): # file_name(string)
        # reading data from file into numpy array_2d
        self.array_2d = np.genfromtxt("/Users/mac/Downloads/Python/Assignment 2021-20211225/" + self.file_name, delimiter=",") # reading from file into 2d numpy array

    def standardise(self):
        # this method is for standardising the array_2d we have in out class matrix
        # we have the details into appendix that how we can do this thing
        n, m = self.array_shape()
        standardized_matrix = np.empty([n, m], dtype=float)
        for i, j in np.ndindex(self.array_2d.shape):
            average = np.average(self.array_2d[:, j]) # average of column
            max = np.max(self.array_2d[:, j]) # max of column
            min = np.min(self.array_2d[:, j]) # min of column
            standardized_matrix[i, j] = self.array_2d[i, j] - average / max - min

        self.array_2d = standardized_matrix # copying standardized array into the numpy 2d_array

    def get_distance(self, other_matrix, weights, beta): # other_matrix(numpy_Array) | weights(numpy_Array)  | beta(int)
        # we are calculating euclidean distance in this function
        r, c = self.array_shape()

        S = np.zeros([r, 1], dtype=int) 
        for i in range(0, r):
            distances = []
            for j in range(0, len(other_matrix)):
                distances.append(self.euclidean_distance(self.array_2d[i, :], other_matrix[j], weights, beta))
                S[i, 0] = min(range(len(distances)), key=distances.__getitem__) # getting minimum distance index for us to store

        return S

    def get_count_frequency(self):
        # it works only when our matrix has only one column
        rows, cols = self.array_shape() # getting rows and cols to check that only 1 col
        if cols == 1:
            S = np.asarray(self.array_2d[:, 0])
            #This supported in latest numpy library so make sure to run it using python 3.10
            frequency_count = collections.Counter(S)
            self.start() # preparing again now for another run

            return frequency_count 
        else:
            return 0

    def start(self): # after every iteration in run_test i do start from scratch for every centroids and beta
        self.load_from_csv() # loading from csv file
        self.standardise() # normalizing data using min-max technique
        pass

    def euclidean_distance(self, p1, p2, weights, beta):
        # calculating euclidean distance between 2 points
        return (np.sum( (weights ** beta) * (p1 - p2) ** 2) )

    def array_shape(self): # returning array rows and columns
        return [self.array_2d.shape[0], self.array_2d.shape[1]]


def get_initial_weights(m): # m(integer)
    # returning a function with one row and m columns between 0-1 having sum as 1
    return np.random.dirichlet(np.ones(m), size=1)

def get_centroids(D, S, K): # D(numpy_array) | S(numpy_array) | K(integer)
    # to get k number of centroids from the D
    # K is between [2 - n-1] where n is number of rows
    r, c = D.array_shape()

    centroids_clusters = {}
    centroids = np.zeros([K, c], dtype=float)

    for k in range(0, K):
        centroids_clusters[k] = np.zeros([K, c], dtype=float)

    for i in range(0, r):
        c_index = S[i, 0]
        centroids_clusters[c_index] = np.vstack((centroids_clusters[c_index], D.array_2d[i, :]))

    for k in range(0, K):
        centroids[k] = np.average([centroids_clusters[k]], axis=1)

    return centroids

def get_groups(D, K, beta): # D(numpy_array) | K(integer) | beta(integer)
    # K is between [2 s- n-1] where n is number of rows    
    r, c = D.array_shape() # matrix rows and columns
    
    weights = get_initial_weights(c) # setting initial weights

    centroids = np.empty([K, c], dtype=float)

    S = np.zeros([r, 1], dtype=int) # our S where we will be storing our nearest centroids rows index

    # selecting random rows as K for centroids
    random_rows = np.random.choice(r, size=K, replace=False)
    random_data_rows = D.array_2d[random_rows , :]
    centroids = random_data_rows

    while True:
        # storing distances into S
        E = np.asarray(D.get_distance(centroids, weights, beta)) # storing them into S

        if (S == E).all():
            break
        else:
            S = E
            
        # now updating centroids
        centroids = get_centroids(D, S, K)

        # now getting new weights
        weights = get_new_weights(D, centroids, S)

    D.array_2d = S
    return D

def get_new_weights(D, centroids, S): # D(numpy_array) | centroids(numpy_array) | S(numpy_array)
    r, c = D.array_shape()
    weights = np.empty([1, c], dtype=float)
    
    # dispersion for a combined
    r2 = 0
    for m in range(0, c): # iterating onto columns  
        r2 += dispersion_calculation(D, centroids, r, m, S)

    for j in range(0, c): # iterating onto columns 

        r1 = dispersion_calculation(D, centroids, r, j, S) # dispersion for every column while iterating -> j
        
        if r1 == 0: # if delta j is zero then weight of that j is also zero
            weights[0, j] = 0
        else:
            weights[0, j] = 1 / ((r1 / r2) ** (1 / beta - 1))

    return weights

def dispersion_calculation(D, centroids, r, j, S):
    outer_loop = 0
    for k in range(0, len(centroids)): # iterating upto number of centroid k = 1 -> K
        inner_loop = 0
        for i in range(0, r): # iterating upto number of rows
            u_ik = 0 # uik is zero when Si == K 
            if S[i] == k: # if S(i) != 0 then zero
                u_ik = 1
                inner_loop += u_ik * (D.array_2d[i, j] - centroids[k, j]) ** 2

        outer_loop += inner_loop

    return outer_loop

def run_test():
    m = matrix("Data.csv")
    for k in range(2,5):
        for beta in range(11,25):
            S = get_groups(m, k, beta/10) 
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))


if __name__ == "__main__":
    run_test()