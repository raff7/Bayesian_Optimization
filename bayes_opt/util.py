import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import sobol_seq
from line_profiler import LineProfiler
from joblib import Parallel, delayed
import multiprocessing
from time import time
from copy import deepcopy

class augKernel(Matern):
    def __init__(self, discrete=None ,length_scale=1.0, length_scale_bounds=(1e-5, 1e5),nu=1.5):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds,nu=nu)
        self.discrete = discrete
    def __call__(self, X, Y=None, eval_gradient=False):

        if(self.discrete is None):
            return super().__call__(X,Y,eval_gradient)
        else:
            nX = X.copy()
            nY = Y.copy() if Y is not None else None
            idX = np.where(self.discrete==1)[0]
            for i in range(len(nX)):
                nX[i][idX] = nX[i][idX].round()
                for c in range(2,max(self.discrete)+1):
                    categories = np.where(self.discrete==c)[0]
                    v = np.argmax(nX[i][categories])
                    nX[i][categories]= [0]*len(categories)
                    nX[i][categories[v]]=1
            if nY is not None:
                for i in range(len(nY)):
                    nY[i][idX] = nY[i][idX].round()
                    for c in range(2,max(self.discrete)+1):
                        categories = np.where(self.discrete==c)[0]
                        v = np.argmax(nY[i][categories])
                        nY[i][categories]= [0]*len(categories)
                        nY[i][categories[v]]=1
            return super().__call__(nX,nY,eval_gradient)


class Acquisitor():
    def __init__(self,ac, opt, random_state):
        self.ac = ac
        self.opt =opt
        self.random_state=random_state

    def acq_max(self, n_warmup, n_best_iter,n_rand_iter,low_level_parall = False):
        """
        A function to find the maximum of the acquisition function

        It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
        optimization method. First by sampling `n_warmup` (1e5) points at random,
        and then running L-BFGS-B from `n_iter` (250) random starting points.

        Parameters
        ----------
        :param ac:
            The acquisition function object that return its point-wise value.

        :param gp:
            A gaussian process fitted to the relevant data.

        :param y_max:
            The current maximum known value of the target function.

        :param bounds:
            The variables bounds to limit the search of the acq max.

        :param random_state:
            instance of np.RandomState random number generator

        :param n_warmup:
            number of times to randomly sample the aquisition function

        :param n_iter:
            number of times to run scipy.minimize

        Returns
        -------
        :return: x_max, The arg max of the acquisition function.
        """
        # Warm up with random points
        bounds = self.opt._space.bounds
        discrete = self.opt._space.discrete
        if(low_level_parall):
            num_cores = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(num_cores)
        else:
            pool = None
        # x_tries = self.random_state.uniform(bounds[:, 0], bounds[:, 1],
        #                                size=(n_warmup, bounds.shape[0]))
        x_tries = (sobol_seq.i4_sobol_generate(bounds.shape[0], n_warmup) * (bounds[:,1]-bounds[:,0]))+bounds[:,0]
        x_tries = discretize(x_tries,discrete,unique_shuffle=True)
        ys = self.ac(x_tries, opt=self.opt,parall=low_level_parall,pool=pool)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
        counter = 0
        x_bests = x_tries[(-ys).argsort()[:min(n_best_iter*10,len(ys))]]
        x_IDS = self.random_state.uniform(1+0.5,len(x_bests)-0.51,size=(n_best_iter-1))
        x_bests_seeds = [x_bests[0]]
        for i in x_IDS:
            x_bests_seeds.append(x_bests[int(i)])


        for x_try in x_bests_seeds:
            t = time()
            counter +=1
            res = minimize(lambda x: -self.ac(x.reshape(1, -1), opt=self.opt,parall=low_level_parall,pool=pool),
                            x0=x_try.reshape(1, -1),
                            bounds=bounds,
                            method="L-BFGS-B") # SLSQP

            # See if success
            if res.success and (max_acq is None or -res.fun[0] >= max_acq):
                x_max = res.x
                max_acq = -res.fun[0]
                # print("BESTED_>")
            if(self.opt.print_timing):
                print("done {}/{} in {}".format(counter,n_rand_iter+n_best_iter,time()-t))
        # Explore the parameter space more throughly
        x_seeds = self.random_state.uniform(bounds[:, 0], bounds[:, 1],
                                        size=(n_rand_iter, bounds.shape[0]))
        
        x_seeds = discretize(x_seeds,discrete,unique_shuffle=True)

        for x_try in x_seeds:
            counter +=1
            t = time()
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -self.ac(x.reshape(1, -1), opt=self.opt,parall=low_level_parall,pool=pool),
                        x0=x_try.reshape(1, -1),
                        bounds=bounds,
                        method="L-BFGS-B") # SLSQP

            # See if success
            if not res.success:
                continue

            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
                # print("BESTED_>")
            if(self.opt.print_timing):
                print("done {}/{} in {}".format(counter,n_rand_iter+n_best_iter,time()-t))
        if(low_level_parall):
            pool.close()
        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        x_max = np.array(discretize([x_max],discrete)[0])
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])

    def acq_step(self,x_try):
            bounds=self.opt._space.bounds
            # Find the minimum of minus the acquisition function
            t = time()
            res = minimize(lambda x: -self.ac(x.reshape(1, -1), opt=self.opt),
                        x_try.reshape(1, -1),
                        bounds=bounds,
                        method="L-BFGS-B")

            # See if success
            if(self.opt.print_timing):
                print("Done an iteration in parallel in {}".format(time()-t))
            if  res.success:
                return res.x, -res.fun[0]

    def parall_acq_max(self, n_warmup, n_best_iter,n_rand_iter):
        # Warm up with random points
        bounds = self.opt._space.bounds
        discrete = self.opt._space.discrete
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
        # x_tries = self.random_state.uniform(bounds[:, 0], bounds[:, 1],
        #                                size=(n_warmup, bounds.shape[0]))
        x_tries = (sobol_seq.i4_sobol_generate(bounds.shape[0], n_warmup) * (bounds[:,1]-bounds[:,0]))+bounds[:,0]
        x_tries = discretize(x_tries,discrete,unique_shuffle=True)
        ys = self.ac(x_tries, opt=self.opt,parall=True,pool=pool)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()

        # Explore the parameter space more throughly
        x_bests = x_tries[(-ys).argsort()[:min(n_best_iter*10,len(ys))]]
        x_IDS = self.random_state.uniform(1+0.5,len(x_bests)-0.51,size=(n_best_iter-1))
        x_bests_seeds = [x_bests[0]]
        for i in x_IDS:
            x_bests_seeds.append(x_bests[int(i)])
        args = [(x_try) for x_try in x_bests_seeds]
        results = pool.map(self.acq_step,args)
        # if(option == 1):
        #     results = Parallel(n_jobs=num_cores,backend="loky")(delayed(self.acq_step)(x_try) for x_try in x_tries[(-ys).argsort()[:n_best_iter]])
        # if(option == 2):
        #     results = Parallel(n_jobs=num_cores,backend="multiprocessing")(delayed(self.acq_step)(x_try) for x_try in x_tries[(-ys).argsort()[:n_best_iter]])
        
        for re in results:
            if max_acq is None or (re is not None and re[1] >= max_acq):
                x_max = re[0]
                max_acq = re[1]
        x_seeds = self.random_state.uniform(bounds[:, 0], bounds[:, 1],
                                        size=(n_rand_iter, bounds.shape[0]))
        
        x_seeds = discretize(x_seeds,discrete,unique_shuffle=True)
        args = [(x_try) for x_try in x_seeds]
        results = pool.map(self.acq_step,args)
        pool.close()
        # if(option ==1):
        #     results = Parallel(n_jobs=num_cores,backend="loky")(delayed(self.acq_step)(x_try) for x_try in x_seeds)
        # if(option ==2):
        #     results = Parallel(n_jobs=num_cores,backend="multiprocessing")(delayed(self.acq_step)(x_try) for x_try in x_seeds)

        for re in results:
            if max_acq is None or (re is not None and re[1] >= max_acq):
                x_max = re[0]
                max_acq = re[1]
        x_max = np.array(discretize([x_max],discrete)[0])
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])

    
class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi,N_QMC=None):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa
        self.N_QMC = N_QMC
        self.xi = xi
        self.counter = 0

        if kind not in ['ucb', 'ei', 'poi','nei']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, opt,parall = False,pool=None):
        self.counter +=1
        # print(self.counter)
        if self.kind == 'ucb':
            return self._ucb(x, opt, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, opt, self.xi)
        if self.kind == 'poi':
            return self._poi(x, opt, self.xi)
        if self.kind == 'nei':
            if(self.N_QMC is None):
                e =self._Nei(x,opt,self.xi)
            else:
                # lp = LineProfiler()
                # lp_wrapper = lp(self._Nei)
                # lp_wrapper(x,opt,self.xi,N_QMC=self.N_QMC)
                # lp.print_stats()
                # t = time()
                if( not parall):
                    e =self._Nei(x,opt,self.xi,N_QMC=self.N_QMC)
                if(parall):
                    e = self.parall_Nei(x,opt,pool,self.xi,N_QMC=self.N_QMC)
                # t = time()-t
                # tp1=time()
                
                # tp1= time()-tp1
                # print((t-tp1))
                # tp2=time()
                # e2 = self.parall_Nei(x,opt,self.xi,N_QMC=self.N_QMC,backend="multiprocessing")
                # tp2= time()-tp2

            return e

    @staticmethod
    def _ucb(x, opt, kappa):
        #upper confidence bound
        gp = opt._gp
        x = discretize(x,opt._space.discrete)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    def _Nei(self,x, opt, xi,N_QMC=35):
        gp = opt._gp
        discrete = opt._space.discrete
        x = discretize(x,discrete)
        pars = opt._space.params #add batches here as np.concatenate((opt._space.params,previousBatches))
        mean, cov = gp.predict(pars,return_cov = True)
        # A = gp.L_
        A = np.linalg.cholesky(cov)

        NEI = 0
        tks = sobol_seq.i4_sobol_generate(len(pars), N_QMC)
        for k in range(N_QMC):#quasi monte carlo integration (QMC)
            # print(k)
            gpk=GaussianProcessRegressor(
                kernel=augKernel(nu=opt._gp.kernel.nu,discrete=discrete),
                alpha=0.0000001,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=k,
                )
            tk = tks[k]
            Fn = (np.diag(A)*norm.ppf(tk) + mean).tolist() #generate pseudorandom noisless observations from noisy observations
            with warnings.catch_warnings():#Do not print gp warnings
                warnings.simplefilter("ignore")    
                gpk.fit(pars,Fn)#fit gaussian process with pseudorandom observations
            y_max = max(Fn)
            NEI += 1/N_QMC*self._ei(x,opt,xi,overGP=gpk,overY_max=y_max)#compute EI for gp trained on pseudorandom observations
        return NEI

    def parall_Nei(self,x, opt,pool, xi,N_QMC=35):
        gp = opt._gp
        discrete = opt._space.discrete
        x = discretize(x,discrete)
        pars = opt._space.params #add batches here as np.concatenate((opt._space.params,previousBatches))
        mean, cov = gp.predict(pars,return_cov = True)
        # A = gp.L_
        A = np.linalg.cholesky(cov)
        tks = sobol_seq.i4_sobol_generate(len(pars), N_QMC)

        args = [(opt,pars,x,discrete,k,tks,A,mean,N_QMC,xi) for k in range(N_QMC)]
        results = pool.map(self.MCstep,args)
        # results = Parallel(n_jobs=num_cores,backend=backend)(delayed(self.MCstep)(opt,pars,x,discrete,k,tks,A,mean,N_QMC,xi) for k in range(N_QMC))
        
        return sum(results)

    def MCstep(self,args):
        # for parallel NEI
        opt,pars,x,discrete,k,tks,A,mean,N_QMC,xi = args
        gpk=GaussianProcessRegressor(
            kernel=augKernel(nu=opt._gp.kernel.nu,discrete=discrete),
            alpha=0.0000001,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=k,
            )
        tk = tks[k]
        Fn = (np.diag(A)*norm.ppf(tk) + mean).tolist() #generate pseudorandom noisless observations from noisy observations
        with warnings.catch_warnings():#Do not print gp warnings
            warnings.simplefilter("ignore")    
            gpk.fit(pars,Fn)#fit gaussian process with pseudorandom observations
        y_max = max(Fn)
        pNEI = 1/N_QMC*self._ei(x,opt,xi,overGP=gpk,overY_max=y_max)#compute EI for gp trained on pseudorandom observations
        return pNEI

    @staticmethod
    def _ei(x, opt, xi,overGP=None,overY_max=None):
        #expected Improvement
        x = discretize(x,opt._space.discrete)
        if(overGP is None):
            gp = opt._gp
            y_max = opt._space.max()['target']
        else:
            gp = overGP
            y_max = overY_max
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        # z2 = (mean-y_max-xi)
        # EIn2 = [max(i,0) for i in z2] + std*norm.pdf(z2/std)- abs(z2)*norm.cdf(z2/std)
        # if((EIn2 != (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)).any()):
        #     print()
        #return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, opt, xi):
        #Probability of Iprovement 
        gp = opt._gp
        y_max = opt._space.max()['target']
        x = discretize(x,opt._space.discrete)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)


def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"]/optimizer._norm_constant,
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)

def discretize(x, discrete,unique_shuffle=False):
    discIx = np.where(discrete == 1)[0]
    for i in range(len(x)):
        x[i][discIx] = x[i][discIx].round()
    if(unique_shuffle):
        x = np.unique(x, axis=0)
        np.random.shuffle(x)

    return x