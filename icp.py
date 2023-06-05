import pandas as pd
import numpy as np
import plotly.graph_objects as go

#Read the data and convert to numpy array
dfx = pd.read_csv("pclX.txt",sep=" ",header=None)
X=dfx.to_numpy(dtype=float)

dfy = pd.read_csv("pclY.txt",sep=" ",header=None)
Y=dfy.to_numpy(dtype=float)

#Visualize the two point clouds
fig = go.Figure()
fig.add_trace(go.Scatter3d(mode='markers',x=X[:,0],y=X[:,1],z=X[:,2],marker=dict(color='rgb(256,0,0)',size=1)))
fig.add_trace(go.Scatter3d(mode='markers',x=Y[:,0],y=Y[:,1],z=Y[:,2],marker=dict(color='rgb(0,0,256)',size=1)))
fig.show()

def EstimateCorrespondencies(X,Y,t,R,d_max):
    c=[]
    x_corrected = np.dot(X,R)+t
    for i in range(len(x_corrected)):
        norm=np.linalg.norm(Y-x_corrected[i],axis=1)
        y_correspondence = np.argmin(norm)
        if norm[y_correspondence]<d_max:
            c.append((i,y_correspondence))
    return np.array(c)

def ComputeOptimalRigidRegistration(X,Y,c):
    #point cloud centroids
    X_centroid = np.mean(X[c[:,0]],axis=0)
    Y_centroid = np.mean(Y[c[:,1]],axis=0)

    #Calculate the deviation from centroid
    x_deviation = X[c[:,0]] - X_centroid
    y_deviation = Y[c[:,1]] - Y_centroid

    #Compute cross-covariance matrix
    H=np.dot(x_deviation.T,y_deviation)
    
    #Compute SVD
    U,S,V= np.linalg.svd(H)

    #Construct optimal rotation
    Rotation = np.dot(U,V)

    #Optimal Translation
    translation = Y_centroid - np.dot(X_centroid,Rotation)

    return Rotation, translation

def ICP_alg(X,Y,T0,R0,d_max,max_iter):
    for i in range(max_iter):
        C=EstimateCorrespondencies(X,Y,T0,R0,d_max)
        R, t = ComputeOptimalRigidRegistration(X,Y,C)
        T0=t
        R0=R
    return t,R,C


#Implementation
t=np.zeros((1,3))
R=np.array([[1,0,0],[0,1,0],[0,0,1]])
d_max = 0.25
iter = 30
t,R,C = ICP_alg(X,Y,t,R,d_max,iter)
print("This is the Rotation Matrix","\n",R,"\n")
print("This is the translation Matrix","\n",t,"\n")

Y_correspondence = Y[C[:,1]]
X_correspondence = X[C[:,0]]

corrected_X = np.dot(X,R)+t
error = Y[C[:,1]]-corrected_X[C[:,0]]
square_error  = np.square(np.linalg.norm(error,axis=1))
MSE = square_error.sum()/len(X)
RMSE = np.sqrt(MSE)
print("ERROR: ",RMSE)

#Visualise the corrected point clouds
fig = go.Figure()
fig.add_trace(go.Scatter3d(name='X',mode='markers',x=X[:,0],y=X[:,1],z=X[:,2],marker=dict(color='rgb(256, 0 ,0)',size=1)))
fig.add_trace(go.Scatter3d(name='Y',mode='markers',x=Y[:,0],y=Y[:,1],z=Y[:,2],marker=dict(color='rgb(0, 0 ,256)',size=1)))
fig.add_trace(go.Scatter3d(name='Corrected_X',mode='markers',x=corrected_X[:,0],y=corrected_X[:,1],z=corrected_X[:,2],marker=dict(color='rgb(0, 256 ,0)',size=1)))
fig.show()