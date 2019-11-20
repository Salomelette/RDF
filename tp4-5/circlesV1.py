import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData
from torch.utils.tensorboard import SummaryWriter
import random
writer=SummaryWriter()

def init_params(nx, nh, ny):
    params = dict()
    params["Wh"]=torch.from_numpy(np.random.normal(scale=0.3,size=(nx,nh)))
    params["bh"]=torch.from_numpy(np.random.normal(scale=0.3,size=(nh)))
    params["Wy"]=torch.from_numpy(np.random.normal(scale=0.3,size=(nh,ny)))
    params["by"]=torch.from_numpy(np.random.normal(scale=0.3,size=(ny)))
    for key in params.keys():
        params[key].requires_grad=True
    # TODO remplir avec les paramètres Wh, Wy, bh, by
    # params["Wh"] = ...

    return params


def forward(params, X):
    outputs = dict()
    activtanh=torch.nn.Tanh()
    activsoftmax=torch.nn.Softmax(dim=-1)
    outputs["X"]=X
    outputs["htilde"]=torch.mm(X,params["Wh"])
    outputs["htilde"]+=params["bh"]
    outputs["h"]=activtanh(outputs["htilde"])
    outputs['ytilde']=torch.mm(outputs["h"],params["Wy"])+params["by"]
    outputs["yhat"]=activsoftmax(outputs['ytilde'])
    outputs["yhat"]=outputs['ytilde']
    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    # outputs["X"] = ...

    return outputs['yhat'], outputs



def loss_accuracy(Yhat, Y):
    #print("yhay" , Yhat.shape)
    #print("y" , Y.shape)
    criterion=torch.nn.CrossEntropyLoss() #torch.nn.NLLLoss() 

    _,indsY=torch.max(Y,1)
    L=criterion(Yhat,indsY)


    nb_correct=0
    _,indsYhat = torch.max(Yhat,1)
    for i in range(len(indsY)):
        if indsY[i].item()==indsYhat[i].item():
            nb_correct+=1
    
    acc=int(nb_correct)/len(Y)
    return L, acc

def backward(params, outputs, Y):
    grads = {}
    L,acc=loss_accuracy(outputs["yhat"],Y)
    print(L)
    print(acc)
    L.backward()
    
    for key in params.keys():
        grads[key]=params[key].grad
    # TODO remplir avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...
    #print(grads)
    return grads

def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params
    with torch.no_grad():
        for key in params.keys():
            params[key]-=eta * grads[key]
    return params



if __name__ == '__main__':

    # init
    data = CirclesData()
    #data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 20
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.003
    index=np.arange(len(data.Xtrain))
    random.shuffle(index)
    data.Xtrain=data.Xtrain[index]
    data.Ytrain=data.Ytrain[index]
    # Premiers tests, code à modifier
    params = init_params(nx, nh, ny)
    """Yhat, outs = forward(params, data.Xtrain.double())
    L, _ = loss_accuracy(Yhat.double(), data.Ytrain.long())
    grads = backward(params, outs, data.YtrainY)
    params = sgd(params, grads, eta)"""

    # TODO apprentissage

    nb_epoch=100
    newx=[]
    newy=[]
    for i in range(0,N-Nbatch,Nbatch):
        newx.append(data.Xtrain[i:i+Nbatch])
        newy.append(data.Ytrain[i:i+Nbatch])

    for i in range(nb_epoch):
        tab_loss_train=[]
        tab_acc=[]
        for j in range(len(newx)):
            yhat,outs = forward(params,newx[j].double())
            L,acc =loss_accuracy(yhat.double(),newy[j].long())
            tab_loss_train.append(L.item())
            tab_acc.append(acc)
            grads=backward(params,outs,newy[j])
            params =sgd(params,grads,eta)
            #print(L)
        yhat,outs=forward(params,data.Xtest.double())
        L,acc=loss_accuracy(yhat,data.Ytest)
        writer.add_scalar('Loss/train',np.mean(tab_loss_train),i)

        writer.add_scalar('Loss/test',L,i)
        writer.add_scalar('acc/train',np.mean(tab_acc),i)

        writer.add_scalar('acc/test',acc,i)
    # attendre un appui sur une touche pour garder les figures
    input("done")

    #tessst

    yhat,outs=forward(params,data.Xtest.double())
    L,acc=loss_accuracy(yhat,data.Ytest)
    print("loss test ",L)
    print("accu test ",acc)
