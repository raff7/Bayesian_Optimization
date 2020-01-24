from bayes_opt import BayesianOptimization
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from bayes_opt.util import augKernel
from bayes_opt import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import time
def six_hump_camel(x1,x2):
    r = (4-2.1*(x1**2)+((x1**4)/3))*(x1**2) + (x1*x2) + (-4+4*(x2**2))*(x2**2)
    return -r

def branin(x1,x2):
    a=1
    b=5.1/(4*(np.pi**2))
    c=5/np.pi
    r=6
    s=10
    t=1/(8*np.pi)
    r = a*(x2-(b*(x1**2)) +c*x1-r)**2 + s*(1-t)*np.cos(x1) + s
    return -r

def hartmann6d(x1,x2,x3,x4,x5,x6):
    xs = [x1,x2,x3,x4,x5,x6]
    alphas = [1.0,1.2,3.0,3.2]
    A = [[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]]
    P = 0.0001* np.array([[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],[2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]])
    r = 0

    for i in range(4):
        ri=0
        for j in range(6):
            ri -= A[i][j]*(xs[j]-P[i][j])**2
        r -= alphas[i]* np.exp(ri)
    return -r
def f(x,a=0):
    r = x * np.sin(x) + norm.pdf(x,loc=8.5,scale=0.35)*10
    return r

def noise(**params):
    return hartmann6d(**params) + np.random.normal(loc=0,scale=10)
def fr(a=0,b=0,c=0,d=0,x=0,y=0,z=0):
    if(isinstance(x,float)):
        x= round(x)
    else:
        x=x.round()
    r = x * np.sin(x) + norm.pdf(x,loc=5,scale=0.35)*10
    
    return r
def fd(a=0,x=0,y=0):
    if(a==1):
        r = x * np.sin(x) + norm.pdf(x,loc=5,scale=0.35)*10
    elif(a==2):
        r = x * np.sin(x) + norm.pdf(x,loc=6,scale=0.35)*10
    elif(a==3):
        r = x * np.sin(x) + norm.pdf(x,loc=7,scale=0.36)*10
    else:
        TypeError
    return r
def fc(a=0,x=0,y=0):
    if(a==0):
        r = x * np.sin(x)*1.1 + norm.pdf(x,loc=5,scale=0.35)*10
    elif(a==1):
        r = x * np.sin(x) + norm.pdf(x,loc=-9,scale=0.35)*10
    elif(a==2):
        r = x * np.sin(x) + norm.pdf(x,loc=-1,scale=0.3)*10
    else:
        TypeError
    return r

def gpFit(X,y):
    X = np.array(X).reshape(-1,1)
    y = np.array(X).reshape(-1,1)
    gp = GaussianProcessRegressor(   
        kernel=Matern(nu=2.5),
        alpha=0.00005,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=5,
        )
    gp.fit(X,y)
    x = np.atleast_2d(np.linspace(0, 10, 1000000)).T
    pp=np.array([np.array([i[0],0]) for i in x])
    y_pred, sigma = gp.predict(pp, return_std=True)
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 5 * sigma,
                            (y_pred + 5 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None')
    plt.plot(x, y_pred, 'b-')
    plt.plot(X, y, 'r.', markersize=10)
    plt.draw()
    plt.pause(0.5)
    plt.clf()

def plot(optimizer,nextPlt):
    #plotter for 1d function
    fNorm = lambda x:f(x)/optimizer._norm_constant
    X = optimizer._space.params
    X1d=X
    y = optimizer._space.normalized_target
    x = np.atleast_2d(np.linspace(-25, 25, 20000)).T
    a = [0,1,2]
    gp = optimizer._gp
    # gp = GaussianProcessRegressor(
    #         kernel=augKernel(nu=2.5),
    #         alpha=0.00001,
    #         normalize_y=True,
    #         n_restarts_optimizer=5,
    #         random_state=19,
            
    #     )
    # gp.fit(X,y)
    # pp=np.array([np.array([i[0],0]) for i in x])
    y_pred, sigma = gp.predict(x, return_std=True)
    fig, (ax1,ax2) = plt.subplots(2, 1,figsize=(15,5))
    ax1.plot(x, fNorm(x=x), 'r:')
    ax1.plot(X1d[:-1], y[:-1], 'r.', markersize=10)
    ax1.plot(X1d[-1], y[-1], 'g*', markersize=15)
    ax1.plot(x, y_pred, 'b-')
    ax1.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.5 * sigma,
                            (y_pred + 1.5 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_ylim(-2, 2)
    
    # ax2.plot(x,optimizer.util.utility(x,optimizer,overGP=optimizer._gp,overY_max=optimizer._gp.y_train_.max()),'k:')
    plt.draw()
    plt.pause(0.1)
    plt.clf()

def plot2D(optimizer):
    #plotter for 2d function
    Xa = optimizer._space.aug_params()
    y = optimizer._space.target
    x = np.atleast_2d(np.linspace(-10, 10, 1000)).T
    gp = optimizer._gp
    Xa0 = [(ai[1],yy) for ai,yy in zip(Xa[:-1],y[:-1]) if ai[0]==0]
    Xa1 = [(ai[1],yy) for ai,yy in zip(Xa[:-1],y[:-1]) if ai[0]==1]
    Xa2 = [(ai[1],yy)  for ai,yy in zip(Xa[:-1],y[:-1]) if ai[0]==2]
    XaLast = Xa[-1]

    
    y_pred0, sigma0 = gp.predict(np.concatenate((np.array([1]*len(x)).reshape(-1,1),np.array([0]*len(x)).reshape(-1,1),np.array([0]*len(x)).reshape(-1,1),x),axis=1), return_std=True)
    y_pred1, sigma1 = gp.predict(np.concatenate((np.array([0]*len(x)).reshape(-1,1),np.array([1]*len(x)).reshape(-1,1),np.array([0]*len(x)).reshape(-1,1),x),axis=1), return_std=True)
    y_pred2, sigma2 = gp.predict(np.concatenate((np.array([0]*len(x)).reshape(-1,1),np.array([0]*len(x)).reshape(-1,1),np.array([1]*len(x)).reshape(-1,1),x),axis=1), return_std=True)

    plt.clf()
    fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(15,5))
    ax1.plot(x, optimizer._space.target_func(x=x,a=0), 'r:')
    if(len(Xa0)>0):
        ax1.plot(np.array(Xa0)[:,0], np.array(Xa0)[:,1], 'r.', markersize=10)
    if(XaLast[0]==0):
        ax1.plot(XaLast[1], y[-1], 'g*', markersize=15)
    ax1.plot(x, y_pred0, 'b-')
    ax1.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred0 - 2.5 * sigma0,
                            (y_pred0 + 2.5 * sigma0)[::-1]]),
            alpha=.5, fc='b', ec='None')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_ylim(-4, 8)

    ax2.plot(x, optimizer._space.target_func(x=x,a=1), 'r:')
    if(len(Xa1)>0):
        ax2.plot(np.array(Xa1)[:,0], np.array(Xa1)[:,1], 'r.', markersize=10)
    if(XaLast[0]==1):
        ax2.plot(XaLast[1], y[-1], 'g*', markersize=15)
    ax2.plot(x, y_pred1, 'b-')
    ax2.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred1 - 2.5 * sigma1,
                            (y_pred1 + 2.5 * sigma1)[::-1]]),
            alpha=.5, fc='b', ec='None')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f(x)$')
    ax2.set_ylim(-4, 8)


    ax3.plot(x, optimizer._space.target_func(x=x,a=2), 'r:')
    if(len(Xa2)>0):
        ax3.plot(np.array(Xa2)[:,0], np.array(Xa2)[:,1], 'r.', markersize=10)
    if(XaLast[0]==2):
        ax3.plot(XaLast[1], y[-1], 'g*', markersize=15)
    ax3.plot(x, y_pred2, 'b-')
    ax3.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred2 - 2.5 * sigma2,
                            (y_pred2 + 2.5 * sigma2)[::-1]]),
            alpha=.5, fc='b', ec='None')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$f(x)$')
    ax3.set_ylim(-4, 8)
    plt.draw()
    plt.pause(0.5)
    plt.clf()


def plot2DCont(optimizer):
    #plotter for 2d function
    Xa = optimizer._space.params
    y = optimizer._space.target
    x = np.atleast_2d(np.linspace(-10, 10, 1000)).T
    gp = optimizer._gp
    # 
    Xa0 = [(ai[1],yy) for ai,yy in zip(Xa[:-1],y[:-1]) if ai[0]==1]
    Xa1 = [(ai[1],yy) for ai,yy in zip(Xa[:-1],y[:-1]) if ai[0]==2]
    Xa2 = [(ai[1],yy)  for ai,yy in zip(Xa[:-1],y[:-1]) if ai[0]==3]
    XaLast = Xa[-1]

    y_pred0, sigma0 = gp.predict(np.concatenate((np.array([1]*len(x)).reshape(-1,1),x),axis=1), return_std=True)
    y_pred1, sigma1 = gp.predict(np.concatenate((np.array([2]*len(x)).reshape(-1,1),x),axis=1), return_std=True)
    y_pred2, sigma2 = gp.predict(np.concatenate((np.array([3]*len(x)).reshape(-1,1),x),axis=1), return_std=True)
    

    plt.clf()
    fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(15,5))
    ax1.plot(x, optimizer._space.target_func(x=x,a=1), 'r:')
    if(len(Xa0)>0):
        ax1.plot(np.array(Xa0)[:,0], np.array(Xa0)[:,1], 'r.', markersize=10)
    if(XaLast[0]==1):
        ax1.plot(XaLast[1], y[-1], 'g*', markersize=15)
    ax1.plot(x, y_pred0, 'b-')
    ax1.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred0 - 2.5 * sigma0,
                            (y_pred0 + 2.5 * sigma0)[::-1]]),
            alpha=.5, fc='b', ec='None')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_ylim(-6, 8)

    ax2.plot(x, optimizer._space.target_func(x=x,a=2), 'r:')
    if(len(Xa1)>0):
        ax2.plot(np.array(Xa1)[:,0], np.array(Xa1)[:,1], 'r.', markersize=10)
    if(XaLast[0]==2):
        ax2.plot(XaLast[1], y[-1], 'g*', markersize=15)
    ax2.plot(x, y_pred1, 'b-')
    ax2.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred1 - 2.5 * sigma1,
                            (y_pred1 + 2.5 * sigma1)[::-1]]),
            alpha=.5, fc='b', ec='None')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$f(x)$')
    ax2.set_ylim(-4, 8)


    ax3.plot(x, optimizer._space.target_func(x=x,a=3), 'r:')
    if(len(Xa2)>0):
        ax3.plot(np.array(Xa2)[:,0], np.array(Xa2)[:,1], 'r.', markersize=10)
    if(XaLast[0]==3):
        ax3.plot(XaLast[1], y[-1], 'g*', markersize=15)
    ax3.plot(x, y_pred2, 'b-')
    ax3.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred2 - 2.5 * sigma2,
                            (y_pred2 + 2.5 * sigma2)[::-1]]),
            alpha=.5, fc='b', ec='None')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$f(x)$')
    ax3.set_ylim(-4, 8)
    # y_max = max(y)
    # eiY = optimizer.util.utility(x,gp,y_max)
    # ax2.plot(x, eiY, 'k:', label='Expected Improvement')
    # ax2.plot(x[np.argmax(eiY)].tolist()*1000,np.linspace(min(eiY), max(eiY), 1000),'r-')
    plt.draw()
    plt.pause(0.5)
    plt.clf()

# plt.ion()
# pbounds = {'x': (-25   , 25)}
pbounds ={'x1':(0,1),'x2':(0,1),'x3':(0,1),'x4':(0,1),'x5':(0,1),'x6':(0,1)}
expectedYbounds = (-1,200)
optimizer = BayesianOptimization(
    f=noise,
    pbounds=pbounds,
    yrange = expectedYbounds,
    verbose=2,
    random_state=2, 
    alpha=10,
    noisy=True,
    parall_option=2,
    print_timing=True,
  
)
# load_logs(optimizer, logs=["./logs.json"])
logger = JSONLogger(path="./logs_NEI.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
# optimizer.probe(params=[1],lazy=False,)
nex = None

    
optimizer.maximize(
        init_points=5,
        n_iter=50,
        acq='nei',
        xi=0.00,
        N_QMC=20,
        optimizer_best_trials=3,
        optimizer_random_trials=10,
        optimizer_n_warmups=30000
    )
# plot(optimizer,nex) 
# for i in range(100):
#     t = time.time()
#     optimizer.maximize(
#         init_points=0,
#         n_iter=1,
#         acq='ei',
#         xi=0.00,
#         N_QMC=20,
#         optimizer_best_trials=2,
#         optimizer_random_trials=10,
#         optimizer_n_warmups=10000
#     )
#     plot(optimizer,nex)
#     print("Timing: {} seconds ({} minutes)".format(time.time()-t,(time.time()-t)/60))
