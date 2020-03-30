#!/usr/bin/env python

import os, sys, traceback
import numpy as np
import pylab as pl
from math import exp
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph as kn_graph

from scipy.spatial.distance import cdist, pdist
import scipy.optimize.linesearch  as ln
from cvxopt import matrix, spmatrix, solvers, printing

#import optim.fminsvn as fsvm

solvers.options['show_progress'] = False
try:
    import compiled_mod
    reload(compiled_mod)
except:
    compiled_mod=None


def get_W_L1L2(transp,labels):

    lstlab=np.unique(labels)
    W=np.zeros(transp.shape)

    for i in range(transp.shape[1]):
        for lab in lstlab:
            temp=transp[labels==lab,i]
            #W[]
            n=np.linalg.norm(temp)
            if n:
                W[labels==lab,i]=temp/n
    return W

def loss_L1L2(transp,labels):

    lstlab=np.unique(labels)
    res=0

    for i in range(transp.shape[1]):
        for lab in lstlab:
            temp=transp[labels==lab,i]
            #W[]
            res+=np.linalg.norm(temp)

    return res


def dist(x1,x2=None,metric='euclidean'):
    """Compute distance between samples in x1 and x2"""
    if x2 is None:
        return pdist(x1,metric=metric)
    else:
        return cdist(x1,x2,metric=metric)

def get_W(x,method='unif',param=None):
    """ returns the density estimation for a discrete distribution"""
    if method.lower() in ['rbf','gauss']:
        K = rbf_kernel(x,x,param)
        W = np.sum(K,1)
        W = W/sum(W)
    else:
        if not method.lower()=='unif':
            print("Warning: unknown density estimation, revert to uniform")
        W = np.ones(x.shape[0])/x.shape[0]
    return W


def dots(*args):
    return reduce(np.dot,args)


def get_sim(x,sim,**kwargs):
    if sim=='gauss':
        try:
            rbfparam=kwargs['rbfparam']
        except KeyError:
            rbfparam=1
        S=rbf_kernel(x,x,rbfparam)
    elif sim=='gaussthr':
        try:
            rbfparam=kwargs['rbfparam']
        except KeyError:
            rbfparam=1
        try:
            thrg=kwargs['thrg']
        except KeyError:
            thrg=.5
        S=np.float64(rbf_kernel(x,x,rbfparam)>thrg)
    elif sim=='gaussthr2':
        try:
            rbfparam=kwargs['rbfparam']
        except KeyError:
            rbfparam=1
        try:
            thrg=kwargs['thrg']
        except KeyError:
            thrg=.5
        S=np.float64(rbf_kernel(x,x,rbfparam)>thrg)*rbf_kernel(x,x,rbfparam)
    elif sim=='gaussclass':
        try:
            rbfparam=kwargs['rbfparam']
        except KeyError:
            rbfparam=1
        try:
            y=kwargs['labels']
        except KeyError:
            raise KeyError('sim="gaussclass" require the source labels "labels" to be passed as parameters')
        S=rbf_kernel(x,x,rbfparam)
        temp=np.tile(y.T,(y.shape[0],1))
        temp2=temp==temp.T
        S=S*temp2
    elif sim=='knn':
        try:
            num_neighbors=kwargs['nn']
        except KeyError('sim="knn" requires the number of neighbors nn to be set'):
            num_neighbors=3
        S=kn_graph(x,num_neighbors).toarray()
        S=(S+S.T)/2
    elif sim=='knnclass':
        try:
            num_neighbors=kwargs['nn']
        except KeyError('sim="knnclass" requires the number of neighbors nn to be set'):
            num_neighbors=3
        try:
            y=kwargs['labels']
        except KeyError:
            raise KeyError('sim="gaussclass" requires the source labels "labels" to be passed as parameters')
        S = kn_graph(x, num_neighbors).toarray()
        # handle unlabelled data (class=-1)
        temp = np.tile(y.T,(y.shape[0],1))
        temp2 = (temp == temp.T) * (temp != -1)
        S = (S+S.T)/2
        S = S*temp2
    return S


def compute_transport(xs, xt, method='lp', metric='euclidean', weights='unif', reg=0, solver=None, wparam=1, **kwargs):
    """
    Solve the optimal transport problem (OT)

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = \mu^s

             \gamma^T 1= \mu^t

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - Omega is the regularization term
    - mu_s and mu_t are the sample weights

    Parameters
    ----------
    xs : (ns x d) ndarray
        samples in the source domain
    xt : (nt x d) ndarray
        samples in the target domain
    reg: float()
        Regularization term >0
    method : str()
        Select the regularization term Omega

        - {'lp','None'} : classical optimization LP
            .. math::
                \Omega(\gamma)=0

        - {'qp'} : quadratic regularization
            .. math::
                \Omega(\gamma)=\|\gamma\|_F^2=\sum_{j,k}\gamma_{i,j}^2

        - {'sink','sinkhorn'} : sinkhorn (entropy) regularization
            .. math::
                \Omega(\gamma)=\sum_{i,j}\gamma_{i,j}\log(\gamma_{i,j})

        - {'labelpl1'} : group label regularization + sinkhorn
            .. math::
                \Omega(\gamma)=eta/reg\sum_{j,k}\|\gamma_{\mathcal{L}_k,j}\|_1^p+\sum_{i,j}\gamma_{i,j}\log(\gamma_{i,j})
            where L_k sesign the samples belonging to class k and p=1/2. lables
            of the source samples have to be passed as the named keyword
            'labels'.

        - {'labelpl1_qp'} : group label regularization + quadratic regularization
            .. math::
                \Omega(\gamma)=eta/reg\sum_{j,k}\|\gamma_{\mathcal{L}_k,j}\|_1^p+\sum_{i,j}\gamma_{i,j}^2
            where L_k sesign the samples belonging to class k and p<=1. lables
            of the source samples have to be passed as the named keyword
            'labels'.

        - {'laplace'} : laplacian regularization
            .. math::
                \Omega(\gamma)=(1-\alpha)/n_s^2\sum_{j,k}S_{i,j}\|T(\mathbf{x}^s_i)-T(\mathbf{x}^s_j)\|^2
                +\alpha/n_t^2\sum_{j,k}S_{i,j}^'\|T(\mathbf{x}^t_i)-T(\mathbf{x}^t_j)\|^2
            where the similarity matrices can be selected with the 'sim' parameter and 0<=alpha<=1 allow
            a fine tuning of the weights of each regularization.

            - sim='gauss'
                Gaussian kernel similarity with param 'rbfparam'
            - sim='gaussthr',rbfparam=1.,thrg=.5
                Gaussian kernel similarity with param 'rbfparam' and threshold 'thrg'
            - sim='gaussclass',rbfparam=1.,labels=y
                Gaussian kernel similarity with param 'rbfparam' intra-class only similarity
            - sim='knn',nn=3
                Knn similarity with param 'nn' (number of neighbors)
            - sim='knnclass',nn=3,labels=y
                Knn similarity with param 'nn' (number of neighbors) intra-class only similarity


        - {'laplace_traj'} : laplacian regularization on sample trajectory
            .. math::
                \Omega(\gamma)= (1-\alpha)/n_s^2\sum_{j,k}S_{i,j}\|T(\mathbf{x}^s_i)-\mathbf{x}^s_i-T(\mathbf{x}^s_j)+\mathbf{x}^s_j\|^2
                +\alpha/n_t^2\sum_{j,k}S_{i,j}^'\|T(\mathbf{x}^t_i)-\mathbf{x}^t_i-T(\mathbf{x}^t_j)+\mathbf{x}^t_j\|^2
            where the similarity matrices can be selected with the 'sim' parameter and 0<=alpha<=1 allow
            a fine tuning of the weights of each regularization.


    metric : str
        distance used for the computation of the M matrix.
        parameter can be:  'cityblock','cosine', 'euclidean',
        'sqeuclidean'.

        or any opf the distances described in the documentation of
        scipy.spatial.distance.cdist

    weights: str
        Defines the weights used for the source and target samples.


        Choose from:

        - {'unif'} :  uniform weights
            .. math::
                ,\quad\mu_k^t=1/n_t

        - {'gauss'} : gaussian kernel weights
            .. math::
                \mu_k^s=1/n_s\sum_j exp(-\|\mathbf{x}_k^s-\mathbf{x}_j^s\|^2/(2*wparam))

        - {'precomputed'} : user given weights
               then weightxs and weightxt should be given

    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters

    """

    # metric computation
    M=dist(xs,xt,metric)


    # compute weights
    if weights.lower()=='precomputed':
        w=kwargs['weightxs']
        wtest=kwargs['weightxt']
    else:
        w=get_W(xs,weights,wparam)
        wtest=get_W(xt,weights,wparam)

    # final if
    if method.lower()=='lp':
        transp=computeTransportLP(w,wtest,M,solver=solver)
    elif method.lower()=='qp':
        transp=computeTransportQP(w,wtest,M,reg,solver=solver)
    elif method.lower() in ['sink','sinkhorn']:
        transp=computeTransportSinkhorn(w,wtest,M,1./reg)
    elif method.lower() in ['sink2','sinkhorn2']:
        transp=computeTransportSinkhorn_noscaling(w,wtest,M,1./reg)
    elif method.lower() in ['labelpl1']:
        try:
            labels=kwargs['labels']
            eta=kwargs['eta']
        except KeyError:
            raise KeyError('Method "labelpl1" require the source labels "labels" and the regularization term "eta" to be passed as parameters')
        transp=computeTransportSinkhornLabelsLpL1(w,labels,wtest,M,1./reg,eta)
    elif method.lower() in ['labelpl1_qp']:
        try:
            labels=kwargs['labels']
            eta=kwargs['eta']
        except KeyError:
            raise KeyError('Method "labelpl1_qp" require the source labels "labels" and the regularization term "eta" to be passed as parameters')
        transp=computeTransportQPLabelsLpL1(w,labels,wtest,M,reg,eta,solver=solver)
    elif method.lower() in ['laplace_traj']:
        try:
            sim=kwargs['sim']
            alpha=kwargs['alpha']
        except KeyError:
            raise KeyError('Method "laplace_traj" require the similarity "sim" and the regularization term "alpha" to be passed as parameters')

        Ss=get_sim(xs,**kwargs)
        if sim=='gaussclass':
            kw=kwargs.copy()
            kw['sim']='gauss'
            St=get_sim(xt,**kw)
        else:
            St=get_sim(xt,**kwargs)

        transp = computeTransportLaplacianSymmetricTraj_fw(M,Ss,St,xs,xt,regls=reg*(1-alpha),reglt=reg*alpha,solver=solver,**kwargs)
    elif method.lower() in ['laplace']:
        try:
            sim=kwargs['sim']
            alpha=kwargs['alpha']
        except KeyError:
            raise KeyError('Method "laplace" require the similarity "sim" and the regularization term "alpha" to be passed as parameters')

        Ss=get_sim(xs,**kwargs)
        if sim=='gaussclass':
            kw=kwargs.copy()
            kw['sim']='gauss'
            St=get_sim(xt,**kw)
        elif sim=='knnclass':
            kw=kwargs.copy()
            kw['sim']='knn'
            St=get_sim(xt,**kw)
        else:
            St=get_sim(xt,**kwargs)

        transp = computeTransportLaplacianSymmetric_fw(M,Ss,St,xs,xt,regls=reg*(1-alpha),reglt=reg*alpha,solver=solver,**kwargs)
    elif method.lower() in ['laplace_sinkhorn']:
        try:
            sim=kwargs['sim']
            alpha=kwargs['alpha']
            eta=kwargs['eta']
        except KeyError:
            raise KeyError('Method "laplace_sinkhorn" requires the similarity "sim", the regularization terms "eta" and "alpha" to be passed as parameters')

        Ss=get_sim(xs,**kwargs)
        if sim=='gaussclass':
            kw=kwargs.copy()
            kw['sim']='gauss'
            St=get_sim(xt,**kw)
        elif sim=='knnclass':
            kw=kwargs.copy()
            kw['sim']='knn'
            St=get_sim(xt,**kw)
        else:
            St=get_sim(xt,**kwargs)

        transp = computeTransportLaplacianSymmetric_fw_sinkhorn(M,Ss,St,xs,xt,reg=1./reg,regls=eta*(1-alpha),reglt=eta*alpha,**kwargs)

    elif method.lower() in ['label_sinkhorn']:
        try:
            labels=kwargs['labels']
            eta=kwargs['eta']
        except KeyError:
            raise KeyError('Method "label_sinkhorn" require the source labels "labels" and the regularization term "eta" to be passed as parameters')
        transp=computeTransportL1L2_fw_sinkhorn(w,labels,wtest,M,1./reg,**kwargs)

    else:
        print('Warning: unknown method {method}. Fallback to LP'.format(method=method))
        transp=computeTransportLP(w,wtest,M,solver=solver)

    return transp



def get_K_laplace(Nini,Nfin,S,Sigma):
    """
    fonction pas efficace mais qui marche
    """

    K=np.zeros((Nini*Nfin,Nini*Nfin))

    def idx(i,j):
        return np.ravel_multi_index((i,j),(Nini,Nfin))

    # contruction de K!!!!!!
    for i in range(Nini):
        for j in range(Nini):
            for k in range(Nfin):
                for l in range(Nfin):
                    K[idx(i,k),idx(i,l)]+=S[i,j]*Sigma[k,l]
                    K[idx(j,k),idx(j,l)]+=S[i,j]*Sigma[k,l]
                    K[idx(i,k),idx(j,l)]+=-2*S[i,j]*Sigma[k,l]
    return K

def get_K_laplace2(Nini,Nfin,St,Sigmat):
    """
    fonction pas efficace mais qui marche
    """
    K=np.zeros((Nini*Nfin,Nini*Nfin))

    def idx(i,j):
        return np.ravel_multi_index((i,j),(Nini,Nfin))

    # contruction de K!!!!!!
    for i in range(Nfin):
        for j in range(Nfin):
            for k in range(Nini):
                for l in range(Nini):
                    K[idx(k,i),idx(l,i)]+=St[i,j]*Sigmat[k,l]
                    K[idx(k,j),idx(l,j)]+=St[i,j]*Sigmat[k,l]
                    K[idx(k,i),idx(l,j)]+=-2*St[i,j]*Sigmat[k,l]
    return K

def get_gradient(transp,K):
    s=transp.shape
    res=np.dot(K,transp.flatten())
    return res.reshape(s)

def get_gradient1(L,X,transp):
    """
    Compute gradient for the laplacian reg term on transported sources
    """
    return np.dot(L+L.T,np.dot(transp,np.dot(X,X.T)))

def get_gradient2(L,X,transp):
    """
    Compute gradient for the laplacian reg term on transported targets
    """
    return np.dot(X,np.dot(X.T,np.dot(transp,L+L.T)))

def get_laplacian(S):
    L=np.diag(np.sum(S,axis=0))-S
    return L

def quadloss(transp,K):
    """
    Compute quadratic loss with matrix K
    """
    return np.sum(transp.flatten()*np.dot(K,transp.flatten()))

def quadloss1(transp,L,X):
    """
    Compute loss for the laplacian reg term on transported sources
    """
    return np.trace(np.dot(X.T,np.dot(transp.T,np.dot(L,np.dot(transp,X)))))

def quadloss2(transp,L,X):
    """
    Compute loss for the laplacian reg term on transported sources
    """
    return np.trace(np.dot(X.T,np.dot(transp,np.dot(L,np.dot(transp.T,X)))))

### ------------------------------- Optimal Transport ---------------------------------------


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


########### Compute transport with an LP solver

def computeTransportLP(distribS,distribT, distances,solver=None):
    # init data
    Nini = len(distribS)
    Nfin = len(distribT)


    # generate probability distribution of each class
    p1p2 = np.concatenate((distribS,distribT))
    p1p2 = p1p2[0:-1]
    # generate cost matrix
    costMatrix = distances.flatten()

    # express the constraints matrix
    I = []
    J = []
    for i in range(Nini):
        for j in range(Nfin):
            I.append(i)
            J.append(i*Nfin+j)
    for i in range(Nfin-1):
        for j in range(Nini):
            I.append(i+Nini)
            J.append(j*Nfin+i)

    A = spmatrix(1.0,I,J)

    # positivity condition
    G = spmatrix(-1.0,range(Nini*Nfin),range(Nini*Nfin))

    sol = solvers.lp(matrix(costMatrix),G,matrix(np.zeros(Nini*Nfin)),A,matrix(p1p2),solver=solver)
    S = np.array(sol['x'])

    transp = np.reshape([l[0] for l in S],(Nini,Nfin))
    return transp


def computeTransportQP(distribS,distribT, distances,reg=0,K=None,solver=None):
    # init data
    Nini = len(distribS)
    Nfin = len(distribT)

    # generate probability distribution of each class
    p1p2 = np.concatenate((distribS,distribT))
    p1p2 = p1p2[0:-1]
    # generate cost matrix
    costMatrix = distances.flatten()

    # express the constraints matrix
    I = []
    J = []
    for i in range(Nini):
        for j in range(Nfin):
            I.append(i)
            J.append(i*Nfin+j)
    for i in range(Nfin-1):
        for j in range(Nini):
            I.append(i+Nini)
            J.append(j*Nfin+i)

    A = spmatrix(1.0,I,J)

    # positivity condition
    G = spmatrix(-1.0,range(Nini*Nfin),range(Nini*Nfin))
    if  not K==None:
        P=matrix(K)+reg*spmatrix(1.0,range(Nini*Nfin),range(Nini*Nfin))
    else:
        P=reg*spmatrix(1.0,range(Nini*Nfin),range(Nini*Nfin))

    sol = solvers.qp(P,matrix(costMatrix),G,matrix(np.zeros(Nini*Nfin)),A,matrix(p1p2),solver=solver)
    S = np.array(sol['x'])

    transp = np.reshape([l[0] for l in S],(Nini,Nfin))
    return transp


def computeTransportLaplacianSymmetric(distances,Ss,St,xs,xt,reg=0,regls=0,reglt=0,solver=None):

    distribS=np.ones((xs.shape[0],1))/xs.shape[0]
    distribT=np.ones((xt.shape[0],1))/xt.shape[0]

    Nini = len(distribS)
    Nfin = len(distribT)

    Sigmat=np.dot(xt,xt.T)
    Sigmas=np.dot(xs,xs.T)


    # !!!! MArche pas a refaire je me suis plante dans le deuxieme K
    if compiled_mod==None:
        Ks=get_K_laplace(Nini,Nfin,Ss,Sigmat)
        Kt=get_K_laplace2(Nini,Nfin,St,Sigmas)
    else:
        Ks=compiled_mod.get_K_laplace(Nini,Nfin,Ss,Sigmat)
        Kt=compiled_mod.get_K_laplace2(Nini,Nfin,St,Sigmas)

    K=(Ks*regls+Kt*reglt)

    transp= computeTransportQP(distribS,distribT, distances,reg=reg,K=K,solver=solver)

    #print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

    return transp




def computeTransportLaplacianSource(distances,Ss,xs,xt,reg=0,regl=0,solver=None):

    distribS=np.ones((xs.shape[0],1))/xs.shape[0]
    distribT=np.ones((xt.shape[0],1))/xt.shape[0]

    Nini = len(distribS)
    Nfin = len(distribT)

    Sigma=np.dot(xt,xt.T)

    if compiled_mod==None:
        K=get_K_laplace(Nini,Nfin,Ss,Sigma)
    else:
        K=compiled_mod.get_K_laplace(Nini,Nfin,Ss,Sigma)

    K=K*regl

    transp= computeTransportQP(distribS,distribT, distances,reg=reg,K=K,solver=solver)
    return transp



def computeTransportLaplacianSource_fw(distances,Ss,xs,xt,reg=0,regl=0,solver=None,nbitermax=200,thr_stop=1e-6,step='opt'):

    distribS=np.ones((xs.shape[0],1))/xs.shape[0]
    distribT=np.ones((xt.shape[0],1))/xt.shape[0]


    # compute laplacian
    L=get_laplacian(Ss)

    loop=True

    transp= computeTransportLP(distribS,distribT, distances,solver=solver)
    #transp=np.dot(distribS,distribT.T)

    niter=0;
    while loop:

        old_transp=transp.copy()

        #G=get_gradient(old_transp,K)
        G=regl*get_gradient1(L,xt,old_transp)

        transp0= computeTransportLP(distribS,distribT, distances+G,solver=solver)

        E=transp0-old_transp
        #Ge

        if step=='opt':
        # optimal step size !!!
            tau=max(0,min(1,(-np.sum(E*distances)-np.sum(E*G))/(2*regl*quadloss1(E,L,xt))))
        else:
            # other step size just in case
            tau=2./(niter+2)
        #print "tau:",tau

        transp=old_transp+tau*E

        if niter>=nbitermax:
            loop=False

        if np.sum(np.abs(transp-old_transp))<thr_stop:
            loop=False
        #print niter

        niter+=1



    return transp


def computeTransportLaplacianSymmetric_fw(distances,Ss,St,xs,xt,reg=1e-9,regls=0,reglt=0,solver=None,nbitermax=200,thr_stop=1e-8,step='opt',**kwargs):

    distribS=np.ones((xs.shape[0],1))/xs.shape[0]
    distribT=np.ones((xt.shape[0],1))/xt.shape[0]


    Ls=get_laplacian(Ss)
    Lt=get_laplacian(St)


    loop=True

    transp= computeTransportLP(distribS,distribT, distances,solver=solver)


    niter=0
    while loop:

        old_transp=transp.copy()

        G=np.asarray(regls*get_gradient1(Ls,xt,old_transp)+reglt*get_gradient2(Lt,xs,old_transp))

        transp0= computeTransportLP(distribS,distribT, distances+G,solver=solver)


        E=transp0-old_transp
        #Ge=get_gradient(E,K)


        if step=='opt':
        # optimal step size !!!
            tau=max(0,min(1,(-np.sum(E*distances)-np.sum(E*G))/(2*regls*quadloss1(E,Ls,xt)+2*reglt*quadloss2(E,Lt,xs))))
        else:
            # other step size just in case
            tau=2./(niter+2)       #print "tau:",tau


        transp=old_transp+tau*E

        #print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

        if niter>=nbitermax:
            loop=False

        if np.sum(np.abs(transp-old_transp))<thr_stop:
            loop=False
        #print niter

        niter+=1

    #print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

    return transp

def computeTransportLaplacianSymmetric_fw_sinkhorn(distances,Ss,St,xs,xt,reg=1e-9,regls=0,reglt=0,nbitermax=200,thr_stop=1e-8,**kwargs):

    distribS=np.ones((xs.shape[0],1))/xs.shape[0]
    distribS=distribS.ravel()
    distribT=np.ones((xt.shape[0],1))/xt.shape[0]
    distribT=distribT.ravel()

    Ls=get_laplacian(Ss)
    Lt=get_laplacian(St)

    maxdist = np.max(distances)

    regmax=300./maxdist
    reg0=regmax*(1-exp(-reg/regmax))

    transp= computeTransportSinkhorn(distribS,distribT,distances,reg,maxdist)

    niter=1;
    while True:
        old_transp=transp.copy()
        G=regls*get_gradient1(Ls,xt,old_transp)+reglt*get_gradient2(Lt,xs,old_transp)
        transp0= computeTransportSinkhorn(distribS,distribT,distances + G,reg,maxdist)
        E=transp0-old_transp
        # do a line search for best tau
        def f(tau):
            T = (1-tau)*old_transp+tau*transp0
            #print np.sum(T*distances),-1./reg0*np.sum(T*np.log(T)),regls*quadloss1(T,Ls,xt),reglt*quadloss2(T,Lt,xs)
            return np.sum(T*distances)+1./reg0*np.sum(T*np.log(T))+regls*quadloss1(T,Ls,xt)+reglt*quadloss2(T,Lt,xs)

        # compute f'(0)
        res = regls*(np.trace(np.dot(xt.T,np.dot(E.T,np.dot(Ls,np.dot(old_transp,xt)))))+\
                     np.trace(np.dot(xt.T,np.dot(old_transp.T,np.dot(Ls,np.dot(E,xt))))))\
             +reglt*(np.trace(np.dot(xs.T,np.dot(E,np.dot(Lt,np.dot(old_transp.T,xs)))))+\
                     np.trace(np.dot(xs.T,np.dot(old_transp,np.dot(Lt,np.dot(E.T,xs))))))


        #derphi_zero = np.sum(E*distances) - np.sum(1+E*np.log(old_transp))/reg + res
        derphi_zero = np.sum(E*distances) + np.sum(E*(1+np.log(old_transp)))/reg0 + res

        tau,cost = ln.scalar_search_armijo(f, f(0),derphi_zero,alpha0=0.99)



        if tau is None:
            break
        transp=(1-tau)*old_transp+tau*transp0

        if niter>=nbitermax or np.sum(np.fabs(E))<thr_stop:
            break
        niter+=1
    print('nbiter={}'.format(niter))
    return transp


def computeTransportL1L2_fw_sinkhorn(distribS,LabelsS, distribT, M, reg, eta=0.1,nbitermax=200,thr_stop=1e-8,**kwargs):


    Nini = len(distribS)
    Nfin = len(distribT)

    W=np.zeros(M.shape)

    maxdist = np.max(M)
    distances=M

    regmax=300./maxdist
    reg0=regmax*(1-np.exp(-reg/regmax))

    transp= computeTransportSinkhorn(distribS,distribT,distances,reg,maxdist)

    niter=1;
    while True:
        old_transp=transp.copy()

        W = get_W_L1L2(old_transp,LabelsS)
        G=eta*W
        transp0= computeTransportSinkhorn(distribS,distribT,distances + G,reg,maxdist)
        deltatransp = transp0 - old_transp
        # do a line search for best tau
        def f(tau):
            T = old_transp+tau*deltatransp
            #print np.sum(T*distances)-1./reg*np.sum(T*np.log(T))+eta*loss_L1L2(T,LabelsS)
            return np.sum(T*distances)+1./reg0*np.sum(T*np.log(T))+eta*loss_L1L2(T,LabelsS)

        # compute f'(0)
        lstlab=np.unique(LabelsS)
        res=0
        for i in range(transp.shape[1]):
            for lab in lstlab:
                temp1=old_transp[LabelsS==lab,i]
                temp2=deltatransp[LabelsS==lab,i]
                res+=np.dot(temp1,temp2)/np.linalg.norm(temp1)
        derphi_zero = np.sum(deltatransp*distances) + np.sum(deltatransp*(1+np.log(old_transp)))/reg0 + eta*res

        tau,cost = ln.scalar_search_armijo(f, f(0), derphi_zero,alpha0=0.99)
        #tau=2./(niter+2)
        #print "tau:",tau#,' cost=',cost
        if tau is None:
            break
        transp=(1-tau)*old_transp+tau*transp0

        if niter>=nbitermax or np.sum(np.fabs(deltatransp))<thr_stop:
            break
        niter+=1
    #print 'nbiter=',niter
    return transp

def computeTransportLaplacianSymmetricTraj_fw(distances,Ss,St,xs,xt,reg=0,regls=0,reglt=0,solver=None,nbitermax=200,thr_stop=1e-8,step='opt',**kwargs):

    distribS=np.ones((xs.shape[0],1))/xs.shape[0]
    distribT=np.ones((xt.shape[0],1))/xt.shape[0]

    ns=xs.shape[0]
    nt=xt.shape[0]


    Ls=get_laplacian(Ss)
    Lt=get_laplacian(St)

    Cs=-regls/ns*dots(Ls+Ls.T,xs,xt.T)
    Ct =-reglt/nt*dots(xs,xt.T,Lt+Lt.T)


    loop=True

    transp= computeTransportLP(distribS,distribT, distances,solver=solver)

    Ctot=distances+Cs+Ct

    niter=0;
    while loop:

        old_transp=transp.copy()

        G=regls*get_gradient1(Ls,xt,old_transp)+reglt*get_gradient2(Lt,xs,old_transp)

        transp0= computeTransportLP(distribS,distribT, Ctot+G,solver=solver)


        E=transp0-old_transp
        #Ge=get_gradient(E,K)


        if step=='opt':
        # optimal step size !!!
            tau=max(0,min(1,(-np.sum(E*Ctot)-np.sum(E*G))/(2*regls*quadloss1(E,Ls,xt)+2*reglt*quadloss2(E,Lt,xs))))
        else:
            # other step size just in case
            tau=2./(niter+2)       #print "tau:",tau


        transp=old_transp+tau*E

        #print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

        if niter>=nbitermax:
            loop=False

        if np.sum(np.abs(transp-old_transp))<thr_stop:
            loop=False
        #print niter

        niter+=1

    #print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

    return transp


########### Compute transport with the Sinkhorn algorithm
## ref "Sinkhorn distances: Lightspeed computation of Optimal Transport", NIPS 2013, Marco Cuturi

def computeTransportSinkhorn(distribS,distribT, M, reg,Mmax=0,numItermax = 200,stopThr=1e-9):
    # init data
    Nini = len(distribS)
    Nfin = len(distribT)


    cpt = 0

    # we assume that no distances are null except those of the diagonal of distances
    u = np.ones(Nini)/Nini
    v = np.ones(Nfin)/Nfin
    uprev=np.zeros(Nini)
    vprev=np.zeros(Nini)
    if Mmax:
        regmax=300./Mmax
    else:
        regmax=300./np.max(M)
    #reg=regmax*(1-exp(-reg/regmax))
    #print reg

    K = np.exp(-reg*M)
    #print np.min(K)

    Kp = np.dot(np.diag(1/distribS),K)
    transp = K
    cpt = 0
    err=1
    while (err>stopThr and cpt<numItermax):
        if np.any(np.dot(K.T,u)==0) or np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev
            break
        uprev = u
        vprev = v
        v = np.divide(distribT,np.dot(K.T,u))
        u = 1./np.dot(Kp,v)
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
            err = np.linalg.norm((np.sum(transp,axis=0)-distribT))**2
        cpt = cpt +1
    #print 'err=',err,' cpt=',cpt

    return np.dot(np.diag(u),np.dot(K,np.diag(v)))


def computeTransportSinkhorn_noscaling(distribS,distribT, M, reg,numItermax = 1000,stopThr=1e-9):
    # init data
    Nini = len(distribS)
    Nfin = len(distribT)


    cpt = 0

    # we assume that no distances are null except those of the diagonal of distances
    u = np.ones(Nini)/Nini
    v = np.ones(Nfin)/Nfin
    uprev=np.zeros(Nini)
    vprev=np.zeros(Nini)

    #print reg

    K = np.exp(-reg*M)
    #print np.min(K)

    Kp = np.dot(np.diag(1/distribS),K)
    transp = K
    cpt = 0
    err=1
    while (err>stopThr and cpt<numItermax):
        if np.any(np.dot(K.T,u)==0) or np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev
            break
        uprev = u
        vprev = v
        v = np.divide(distribT,np.dot(K.T,u))
        u = 1./np.dot(Kp,v)
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
            err = np.linalg.norm((np.sum(transp,axis=0)-distribT))**2
        cpt = cpt +1
    #print 'err=',err,' cpt=',cpt

    return np.dot(np.diag(u),np.dot(K,np.diag(v)))


def computeTransportSinkhornWithDualOptimum(distribS,distribT, M, reg,Mmax=0):
    # init data
    Nini = len(distribS)
    Nfin = len(distribT)

    numItermax = 200
    cpt = 1

    # we assume that no distances are null except those of the diagonal of distances
    uprev = u = np.ones(Nini)/Nini
    vprev = v = np.ones(Nfin)/Nfin
    uprev=np.zeros(Nini)
    if Mmax:
        regmax=300./Mmax
    else:
        regmax=300./np.max(M)
    reg=regmax*(1-exp(-reg/regmax))

    K = np.exp(-reg*M)
    #print np.min(K)

    Kp = np.dot(np.diag(1./distribS),K)
    transp = K
    err=1

    while (err>1e-3 and cpt<numItermax):
        if np.any(np.dot(K.T,u)==0) or np.any(np.isnan(np.sum(u))) or np.any(np.isnan(np.sum(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev
            break
        uprev = u
        vprev = v
        v = np.divide(distribT,np.dot(K.T,u))
        u = 1./np.dot(Kp,v)
        cpt = cpt +1
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
            err = np.linalg.norm((np.sum(transp,axis=0)-distribT))

    # now compute dual optimum
    alpha = np.ones(Nini)*np.sum(np.log(u))/(Nini*reg) - np.log(u) / reg

    return transp,alpha


# unlabelled should have the -1 label
def computeTransportSinkhornLabelsLpL1(distribS,LabelsS, distribT, M, reg, eta=0.1):
    p=0.5
    epsilon = 1e-3

    # init data
    Nini = len(distribS)
    Nfin = len(distribT)

    indices_labels = []
    idx_begin = np.min(LabelsS)
    for c in range(idx_begin,np.max(LabelsS)+1):
        idxc = indices(LabelsS, lambda x: x==c)
        indices_labels.append(idxc)

    W=np.zeros(M.shape)

    for cpt in range(10):
        Mreg = M + eta*W
        transp=computeTransportSinkhorn(distribS,distribT,Mreg,reg,numItermax = 200)
        # the transport has been computed. Check if classes are really separated
        W = np.ones((Nini,Nfin))
        for t in range(Nfin):
            column = transp[:,t]
            all_maj = []
            for c in range(idx_begin,np.max(LabelsS)+1):
                col_c = column[indices_labels[c-idx_begin]]
                if c!=-1:
                    maj = p*((sum(col_c)+epsilon)**(p-1))
                    W[indices_labels[c-idx_begin],t]=maj
                    all_maj.append(maj)

            # now we majorize the unlabelled by the min of the majorizations
            # do it only for unlabbled data
            if idx_begin==-1:
                W[indices_labels[0],t]=np.min(all_maj)

    return transp

def computeTransportSinkhornLpL1(distribS, distribT, M, reg, eta=0.1):
    p=0.5
    epsilon = 1e-3

    # init data
    Nini = len(distribS)
    Nfin = len(distribT)


    W=np.zeros(M.shape)

    for cpt in range(10):
        Mreg = M + eta*W
        transp=computeTransportSinkhorn(distribS,distribT,Mreg,reg,numItermax = 200)
        # the transport has been computed. Check if classes are really separated
        W = np.ones((Nini,Nfin))
        for t in range(Nini):
            maj = p*((np.sum(transp[t,:])+epsilon)**(p-1))
            W[t,:]=maj


    return transp

# unlabelled should have the -1 label
def computeTransportQPLabelsLpL1(distribS,LabelsS, distribT, M, reg, eta=0.1,solver=None):
    p=0.5
    epsilon = 1e-2

    # init data
    Nini = len(distribS)
    Nfin = len(distribT)

    indices_labels = []
    idx_begin = np.min(LabelsS)
    for c in range(idx_begin,np.max(LabelsS)+1):
        idxc = indices(LabelsS, lambda x: x==c)
        indices_labels.append(idxc)

    W=np.zeros(M.shape)

    for cpt in range(5):
        Mreg = M + eta*W
        transp=computeTransportQP(distribS,distribT,Mreg,reg,solver=solver)
        # the transport has been computed. Check if classes are really separated
        W = np.ones((Nini,Nfin))
        for t in range(Nfin):
            column = transp[:,t]
            all_maj = []
            for c in range(idx_begin,np.max(LabelsS)+1):
                col_c = column[indices_labels[c-idx_begin]]
                if c!=-1:
                    maj = p*((sum(col_c)+epsilon)**(p-1))
                    W[indices_labels[c-idx_begin],t]=maj
                    all_maj.append(maj)

            # now we majorize the unlabelled by the min of the majorizations
            # do it only for unlabbled data
            if idx_begin==-1:
                W[indices_labels[0],t]=np.min(all_maj)

    return transp

def computeTransportLabelsLpL1(distribS,LabelsS, distribT, M, eta=0.1):
    p=0.5
    epsilon = 1e-2

    # init data
    Nini = len(distribS)
    Nfin = len(distribT)

    indices_labels = []
    idx_begin = np.min(LabelsS)
    for c in range(idx_begin,np.max(LabelsS)+1):
        idxc = indices(LabelsS, lambda x: x==c)
        indices_labels.append(idxc)

    W=np.zeros(M.shape)
    #print eta

    for cpt in range(5):
        Mreg = M + eta*W
        transp=computeTransportLP(distribS,distribT,Mreg)
        # the transport has been computed. Check if classes are really separated
        W = np.ones((Nini,Nfin))
        for t in range(Nfin):
            column = transp[:,t]
            all_maj = []
            for c in range(idx_begin,np.max(LabelsS)+1):
                col_c = column[indices_labels[c-idx_begin]]
                if c!=-1:
                    maj = p*((sum(col_c)+epsilon)**(p-1))
                    W[indices_labels[c-idx_begin],t]=maj
                    all_maj.append(maj)

            # now we majorize the unlabelled by the min of the majorizations
            # do it only for unlabbled data
            if idx_begin==-1:
                W[indices_labels[0],t]=np.min(all_maj)

    return transp



# unlabelled should have the -1 label
# now, classes are also available in the target domain
def computeTransportSinkhornLabelsLpL1_semi(distribS,LabelsS, distribT,LabelsT, M, reg, eta=0.1):
    viz = False
    p=0.5
    epsilon = 1e-2

    # init data
    Nini = len(distribS)
    Nfin = len(distribT)

    eta=p*300*(epsilon)**(p-1)*(1-exp(-eta))

    indices_labels_S = []
    for c in range(np.min(LabelsS),np.max(LabelsS)+1):
        idxc = indices(LabelsS, lambda x: x==c)
        indices_labels_S.append(idxc)

    indices_labels_T = []
    for c in range(np.min(LabelsT),np.max(LabelsT)+1):
        idxc = indices(LabelsT, lambda x: x==c)
        indices_labels_T.append(idxc)

    W=np.zeros(M.shape)

    for cpt in range(5):
        Mreg = M + eta*W
        transp=computeTransportSinkhorn(distribS,distribT,Mreg,reg)
        # the transport has been computed. Check if classes are really separated
        W = np.ones((Nini,Nfin))

        for t in range(Nfin):
            column = transp[:,t]
            for c in range(np.min(LabelsS),np.max(LabelsS)+1):
                col_c = column[indices_labels_S[c-np.min(LabelsS)]]
                if c!=-1:
                    W[indices_labels_S[c-np.min(LabelsS)],t]=p*((sum(col_c)+epsilon)**(p-1))
                else:
                    W[indices_labels_S[c-np.min(LabelsS)],t]=p*((col_c+epsilon)**(p-1))

        for s in range(Nini):
            line = transp[s,:]
            for c in range(np.min(LabelsT),np.max(LabelsT)+1):
                lin_c = line[indices_labels_T[c-np.min(LabelsT)]]
                if c!=-1:
                    W[s,indices_labels_T[c-np.min(LabelsT)]]=p*((sum(lin_c)+epsilon)**(p-1))
                else:
                    W[s,indices_labels_T[c-np.min(LabelsT)]]=p*((lin_c+epsilon)**(p-1))


        if viz:
            fig = pl.figure()
            implot = pl.imshow(W, interpolation="nearest", aspect='auto')
            implot.set_cmap('hot')
            fig.colorbar(implot)
            pl.show()
            fig = pl.figure()
            implot = pl.imshow(transp, interpolation="nearest", aspect='auto')
            implot.set_cmap('pink')
            fig.colorbar(implot)
            pl.show()

    return transp
