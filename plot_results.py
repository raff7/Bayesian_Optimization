Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@raff7 
raff7
/
Bayesian_Optimization
Private
1
00
 Code Issues 0 Pull requests 0 Actions Projects 0 Wiki Security Insights Settings
Bayesian_Optimization/plot_results.py
@raff7 raff7 m
d4ffe63 21 minutes ago
43 lines (40 sloc)  1.27 KB
  
Code navigation is available!
Navigate your code with ease. Click on function and method calls to jump to their definitions or references in the same repository. Learn more

 Code navigation is available for this repository but data for this commit does not exist.

Learn more or give us feedback
from json import JSONDecoder
from matplotlib import pyplot as plt
import numpy as np
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
ei = []
dec = JSONDecoder()
with open("logs_EI.json") as json_file:
    for l in json_file.read().split('\n'):
        try:
            ei.append(dec.decode(l))
        except:
            continue

maxEI = []
real_max_EI = []
for i in ei:
    t = i['target']
    real_t = hartmann6d(i['params']['x1'],i['params']['x2'],i['params']['x3'],i['params']['x4'],i['params']['x5'],i['params']['x6'])
    if len(maxEI)<1 or t >maxEI[-1]:
        maxEI.append(t)
    else:
        maxEI.append(maxEI[-1])
    if len(real_max_EI)<1 or real_t >real_max_EI[-1]:
        real_max_EI.append(real_t)
    else:
        real_max_EI.append(real_max_EI[-1])

plt.plot(maxEI)
plt.plot(real_max_EI,'r')
plt.show()
print()
© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
