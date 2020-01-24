import numpy as np
from .util import ensure_rng


def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.register_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds, random_state=None,norm_constant = 1,noisy=False):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.noisy = noisy
        self.random_state = ensure_rng(random_state)
        self._norm_constant = norm_constant
        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        self._aug_keys = sorted(pbounds)
        # Create an array with parameters bounds
        b = []
        discrete = [0]*len(pbounds.items())
        category_id_counter = 2#count ID of category
        i =0#count the index of category
        for item in sorted(pbounds.items(), key=lambda x: x[0]):
            if(isinstance(item[1],int)):
                b.append((0.5,round(item[1])+0.499999))
                discrete[i]=1
                i+=1
            elif(isinstance(item[1][1],int)):
                b.append(item[1])
                i+=1
            elif((isinstance(item[1][1],str))):
                for j in range(item[1][0]):
                    if(j>0):
                        discrete.append(0)
                        self._aug_keys.insert(i,self._aug_keys[i-1][:-1]+str(int(self._aug_keys[i-1][-1])+1))
                    else:
                        self._aug_keys[i] = self._aug_keys[i]+'0'
                    discrete[i]=category_id_counter
                    i+=1
                    b.append((0,1))
                category_id_counter+=1
            

        self._bounds = np.array(b,dtype=np.float)
        self._discrete = np.array(discrete)
        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._normalized_target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._normalized_target)
        return len(self._normalized_target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params
    
    def aug_params(self,x=None):
        #return augmented parameters (only changes in case there are categorical variables, in which case categorycal var x is split in x1, x2... xn for n possible categories)
        #if x is None x = self.params
        if(x is not None):
            nx = x.copy()
            for c in range(2,max(self.discrete)+1):#this loop will go trough all categorical values and collapse the one hot encoder representation (e.g. [0.3,0.4,0.2] will become [1] because class 1 has highest value = 0.5)
                categories = np.where(self.discrete==c)[0]
                v = np.argmax(x[categories])
                dele = categories[:max(1,len(x[categories])-1)] - (len(x)-len(nx))
                nx = np.delete(nx,dele)
                nx[dele[0]] = v
        else:
            xs = self.params
            nx = []
            for x in xs:
                nx0 = x.copy()
                for c in range(2,max(self.discrete)+1):#this loop will go trough all categorical values and collapse the one hot encoder representation (e.g. [0.3,0.4,0.2] will become [1] because class 1 has highest value = 0.5)
                    categories = np.where(self.discrete==c)[0]
                    v = np.argmax(x[categories])
                    dele = categories[:max(1,len(x[categories])-1)] - (len(x)-len(nx0))
                    nx0 = np.delete(nx0,dele)
                    nx0[dele[0]] = v
                nx.append(nx0)
        return nx

    @property
    def normalized_target(self):
        return self._normalized_target
    @property
    def real_target(self):
        return self._normalized_target*self._norm_constant

    @property
    def dim(self):
        return len(self._bounds)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    @property
    def discrete(self):
        return self._discrete

    def params_to_array(self, params):
        try:
            assert set(params) == set(self._aug_keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self._aug_keys)
            )
        return np.asarray([params[key] for key in self._aug_keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self._aug_keys))
            )
        return x

    def register(self, params, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        y : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        x = self._as_array(params)
        # if x in self:
        #     raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._normalized_target = np.concatenate([self._normalized_target, [target]])

    def probe(self, params):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Notes
        -----
        If x has been previously seen returns a cached value of y.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)
        #compress categorical parameters
        nx = self.aug_params(x=x)
        if(self.noisy):
            params = dict(zip(self._keys, nx))
            target = self.target_func(**params)
            self.register(x, target)
        else:
            try:
                target = self._cache[_hashable(x)]
                self.register(x, target)
            except KeyError:
                params = dict(zip(self._keys, nx))
                target = self.target_func(**params)
                self.register(x, target)
        return target

    def random_sample(self):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
            if(self._discrete[col] == 1):
                data.T[col]=data.T[col].round()
        return data.ravel()

    def normalized_max(self):
        """Get maximum target value found and corresponding parametes."""
        try:
            res = {
                'target': self.normalized_target.max(),
                'params': dict(
                    zip(self._aug_keys, self.params[self.normalized_target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def real_max(self):
        """Get maximum target value found and corresponding parametes."""
        try:
            res = {
                'target': self.real_target.max(),
                'params': dict(
                    zip(self._aug_keys, self.params[self.real_target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def real_res(self):
        """Get all target values found and corresponding parametes (un-normalized, i.e. multiplied by normalizarion constant)."""
        params = [dict(zip(self._aug_keys, p)) for p in self.params]

        return [
            {"target": target*self._norm_constant, "params": param}
            for target, param in zip(self.normalized_target, params)
        ]
    def normalized_res(self):
        """Get all target values found and corresponding parametes (normalized)."""
        params = [dict(zip(self._aug_keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.normalized_target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters

        Alo used to change the type of parameter (contious, discrete or categorical)
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                if(len(new_bounds[key])==1 or isinstance(new_bounds[key][1],str)):
                    self._bounds[row]=([0.5,round(new_bounds[key][0])+0.499999])
                    self._discrete[row]=1
                else:
                    self._bounds[row] = new_bounds[key]
                
