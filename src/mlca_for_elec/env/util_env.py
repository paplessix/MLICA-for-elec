import pandas as pd 
import scipy
from tqdm import tqdm
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
def preprocessor_wind(df, max_capacity, type):
    """Preprocess wind data"""
    df = df.clip(0,24)**3
    df = (df/df.max()*max_capacity).apply(lambda x  : type(x))
    df.rename("generation", inplace = True)
    return df

def preprocessor_solar(df, max_capacity, type):
    """Preprocess solar data"""
    df =(df/df.max()*max_capacity).apply(lambda x  : type(x))
    df.rename("generation", inplace = True)
    return df

def preprocessor_consumption(df, max_capacity, type):
    """Preprocess consumption data"""

    df = (df/df.mean()/2*max_capacity).apply(lambda x  : type(x))
    df.rename("consumption", inplace=True)
    return df

def preprocessor_spot_price(df):
    """Preprocess spot price data"""
    df.rename("spot_price", inplace=True)
    return df/1000*2 #Mean around 100

def preprocessor_fcr_price(df):
    """Preprocess fcr price data"""
    assert len(df) == 2190 
    time =pd.date_range(periods=2190,freq ="4H", start="2019-01-01 00:00:00")
    df = pd.DataFrame(df.values, index=time)
    df = df.resample("1H", closed ="right").mean().fillna(method="ffill")
    df.columns = ["fcr_price"]
    return df.reset_index(drop = True)/4000


######### BUNDLE SAMPLING #########

def greedy_sampling(origin_vector, n_samples, bounds):
    K = n_samples
    dim = len(origin_vector)
    sampled_points = [origin_vector]
    for k in tqdm(range(K)):
        def dist(x,z):
            return np.linalg.norm(x-z, ord=2)

        def objective(z):
            return -np.min([dist(z,x) for x in sampled_points])
        bnds = [(0,bounds[i]) for i in range(dim)]

        sample = scipy.optimize.minimize(objective, x0 = np.zeros(dim),  method = "L-BFGS-B", bounds=bnds)
        sampled_points.append(sample.x)
    return np.asarray(sampled_points)

def l1_ball_sampling(radius, dim):
    # See [1] G. Calafiore, F. Dabbene, and R. Tempo, “Uniform sample generation in l/sub p/ balls for probabilistic robustness analysis,” in Proceedings of the 37th IEEE Conference on Decision and Control (Cat. No.98CH36171), Dec. 1998, pp. 3335–3340 vol.3. doi: 10.1109/CDC.1998.758215.

    epsilon = scipy.stats.gennorm.rvs(beta = 1, size=dim) +1
    z= np.random.uniform(0,1)**(1/dim)
    return radius*epsilon*z/np.linalg.norm(epsilon, ord=1)

def uniform_sampling(n_samples, bounds):
    return np.random.uniform(0, bounds, size=(n_samples, len(bounds)))

def uniform_spacing_sampling(n_samples, bounds):
    U = np.random.uniform(0,1, size=(n_samples, len(bounds)))
    U = np.sort(U, axis=1)
    X = np.zeros((n_samples, len(bounds)+1))
    X[:,1:] = U 
    return (X[:,1:] - X[:,:-1])*240
    
def greedy_sampling(origin_vector, n_samples, bounds, seed=None):
    K = n_samples
    dim = len(origin_vector)
    sampled_points = [origin_vector]
    for k in tqdm(range(K)):
        def dist(x,z):
            return np.linalg.norm(x-z, ord=2)

        def objective(z):
            return -np.min([dist(z,x) for x in sampled_points])
        bnds = [(0,bounds[i]) for i in range(dim)]

        sample = scipy.optimize.minimize(objective, x0 = np.zeros(dim),  method = "L-BFGS-B", bounds=bnds)
        sampled_points.append(sample.x)
    return np.asarray(sampled_points)



def metropolis_sampling(origin_vector, n_samples, bounds,min_cons, max_cons, seed=None):

    def pi(x,V):
        '''
        Density of the target distribution, up to a constant. 
        
        x -- np array of size k
        V -- np array of size k*k
        '''
        if np.any(x > bounds) or np.any(x < 0) or np.linalg.norm(x, ord =1) < min_cons or np.linalg.norm(x, ord = 1) > max_cons:
            return 0
        else:
            lambd = 0.01
            return np.exp(-lambd*np.linalg.norm(x-origin_vector, ord=1)**2)
    
    def prop(x):
        '''
        Random proposition for the Metropolis-Hastings algorithm.
        Uses the Random Walk Metropolis formula with unit variance.
        
        x -- np array of size k
        '''
        return x + normal(size=len(x))
    
    def q(x,y):
        '''
        Probability density of transition x to y, up to a constant.
        Uses the Random Walk Metropolis formula with unit variance.
        
        x -- np array of size k
        y -- np array of size k
        '''
        dist = x-y
        return np.exp(-.5*np.dot(dist,dist))
    
    def MH(N,pi,q,prop,x0,V=np.identity(2)):
        x = x0
        trajectory = [x0]
        for i in range(1,N):
            y = prop(x)
            ratio = pi(y,V)*q(x,y)/pi(x,V)/q(y,x)
            a = np.min([1.,ratio])
            r = np.random.rand()
            if r < a:
                x = y
            trajectory += [x]
        return np.array(trajectory)
    example_V = np.abs(np.random.random((24,24)))

    traj = MH(100000,pi,q,prop,x0 = np.asarray(origin_vector),V=example_V)
    idx = np.random.choice(np.arange(len(traj)), size = n_samples, replace = False)
    samples = traj[idx]    

    return samples



def fps_sampling( n_samples, bounds):

    points = np.random.uniform(0,10, size=(n_samples*10,24))
    def fps(points, n_samples):
        """
        points: [N, 3] array containing the whole point cloud
        n_samples: samples you want in the sampled point cloud typically << N 
        """
        points = np.array(points)
        
        # Represent the points by their indices in points
        points_left = np.arange(len(points)) # [P]

        # Initialise an array for the sampled indices
        sample_inds = np.zeros(n_samples, dtype='int') # [S]

        # Initialise distances to inf
        dists = np.ones_like(points_left) * float('inf') # [P]

        # Select a point from points by its index, save it
        selected = 0
        sample_inds[0] = points_left[selected]

        # Delete selected 
        points_left = np.delete(points_left, selected) # [P - 1]

        # Iteratively select points for a maximum of n_samples
        for i in range(1, n_samples):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]
        
            dist_to_last_added_point = (
                (points[last_added] - points[points_left])).sum(-1) # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_last_added_point, 
                                            dists[points_left]) # [P - i]

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]

            # Update points_left
            points_left = np.delete(points_left, selected)

        return points[sample_inds]        
    sampled_points = fps(points, n_samples)
    return sampled_points                


if __name__=="__main__":
    print(preprocessor_wind)
    print("Test BUndle sampling")
    for i in range(10):
        sample = get_uniform_bundle(33, 100, 24)
        # print(sample)
        print(sample.sum())
