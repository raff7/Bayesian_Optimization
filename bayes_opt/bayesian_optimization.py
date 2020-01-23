import warnings
import numpy as np
from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger

from .util import UtilityFunction, ensure_rng,augKernel, Acquisitor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from time import time
#TODO s: -check if normalized value mess up the "save and load" process
        # - check if the way categorical variables are set up works in all different conditions
        # -should i save categorical values after the one hot encoding is collapsed to the highest value? or as continous numbers?
        # -Maybe make an adaprive range that changes the scaling factor to fit the data he saw (or maybe introducing it as previous knowledge is better.)
    #     -Bloody implement the blooody EI for noisy variables.
class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback == None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, yrange=1, random_state=None, verbose=2,alpha=1e-6,nu=2.5,noisy=False,parameter_normalizer=1.5,parall_option=1,print_timing = False):
        #parameter normalizer is a parameter that decided within how many standard deviation the y limists should be, advised to be between 1 and 3 (lower is better as it avoids getting stuck due to low uncertainty where no data is avaiable)
        #parall_option is to chose the level of parallilazation of the optimizer: 0 is no parallelization, 1 is low level (only valid for NEI, parallel montecarlo estimation) and 2 is high level (when NEI parallel montecarlo estimation during grid search, and then swich to parallelize optimizer)
        """"""
        self._random_state = ensure_rng(random_state)
        self.parall_option = parall_option
        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        
        self.print_timing = print_timing
        # queue
        self._queue = Queue()
        if(isinstance(yrange,int)):#set an expected confidence interval
            _yrange = yrange
        else:
            _yrange = yrange[1]-yrange[0]


        self._norm_constant = _yrange/(parameter_normalizer*2)#1.96 = 95% confidence intervall (0.05 probability of getting elements outside this range)
        self._original_f = f
        self._space = TargetSpace(self.normalizedF, pbounds, random_state,norm_constant=self._norm_constant,noisy=noisy)
        # lambda **params:f(**params)/self._norm_constant
        if(not noisy):
            try:
                assert alpha<0.0001
            except AssertionError:
                raise ValueError(
                    "When working with non-noisy functions use an alpha smaller then 0.00001 (used {} instead) do ".format(alpha) 
                )
        else:#normalize GP noise with respect to the expecter y range.
          alpha = alpha/self._norm_constant

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel= augKernel(nu=nu,discrete=self._space._discrete),
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    def normalizedF(self,**params):
        return self._original_f(**params)/self._norm_constant

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        mx = self._space.max()
        mx['target'] = mx['target']*self._norm_constant
        return mx

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTMIZATION_STEP)

    def suggest(self, utility_function,optimizer_best_trials,optimizer_random_trials,optimizer_n_warmups):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        acquisitor = Acquisitor(
            ac=utility_function.utility,
            opt=self,
            random_state=self._random_state
        )
        if(self.parall_option==2):
            tp2=time()
            suggestion = acquisitor.parall_acq_max(n_best_iter=optimizer_best_trials,n_rand_iter=optimizer_random_trials,n_warmup=optimizer_n_warmups)
            tp2 = time()-tp2
            if(self.print_timing):
                print(tp2)
        else:
            t = time()
            if(self.parall_option==0):
                suggestion = acquisitor.acq_max(low_level_parall=False,n_best_iter=optimizer_best_trials,n_rand_iter=optimizer_random_trials,n_warmup=optimizer_n_warmups)
            elif(self.parall_option==1):
                suggestion = acquisitor.acq_max(low_level_parall=True,n_best_iter=optimizer_best_trials,n_rand_iter=optimizer_random_trials,n_warmup=optimizer_n_warmups)
            else:
                raise ValueError(
                    "Parallelization option should be 0 (none) 1 (low level) or 2 (high level) (used {} instead) do ".format(self.parall_option) 
                )
            t = time()-t
            if(self.print_timing):
                print(t)
        return suggestion

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTMIZATION_START, _logger)
            self.subscribe(Events.OPTMIZATION_STEP, _logger)
            self.subscribe(Events.OPTMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 N_QMC = None,
                 optimizer_best_trials= 4,
                 optimizer_random_trials = 10,
                 optimizer_n_warmups = 10000,
                 **gp_params):
        """Mazimize your function
        inputs:
        xi = offset of the expected improvement. would suggest not to tuch.
        kappa = decides the ratio of mean to standard deviation for UCB
        N_QMC = for NEI, decides the number of iterations of monte carlo simulation (higher has better precision but will take longer)
        optimizer_best_trials = how many trials to let the optimizer from the best results of the grid search
        optimizer_random_trials = how many trials to let the optimizer from randomly selected points
        """

        self._prime_subscriptions()
        self.dispatch(Events.OPTMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)
        # if(self.space.noisy):
            # try:
            #     assert acq == 'nei'
            # except AssertionError:
            #     raise ValueError(
            #         "If working with a noisy function, please use 'nei' acquisition (using {} instead) do ".format(acq) 
            #     )
            
        util = UtilityFunction(kind=acq, kappa=kappa, xi=xi,N_QMC=N_QMC)
        self.util = util
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                x_probe = self.suggest(util,optimizer_best_trials,optimizer_random_trials,optimizer_n_warmups)
                iteration += 1

            self.probe(x_probe, lazy=False)

        self.dispatch(Events.OPTMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)
