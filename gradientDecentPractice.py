import numpy as np
import matplotlib.pyplot as plt
import math

#### hyper parameter
startPoint = np.array([0,0,0,0,0,0,0])
learningRate = 0.1

#### function what we want to predict
def targetFunc(x,y):
    return 2*x**2+3*y**2 
    # return 10*x-y 

def cartesianPruducts(x,y):
    xy=[]
    m=[]
    for i in y:
        for j in x:
            m.append([j,i])
        xy.append(m)
        m=[]
    return xy

#### data
x=np.arange(-10, 10, 1)
y=np.arange(-10, 10, 1)
xy=cartesianPruducts(x,y)
x,y = np.meshgrid(x, y)

z=targetFunc(x,y)

fig = plt.figure(figsize = (6, 6))

ax = plt.axes(projection = '3d')
ax.plot_wireframe(x, y, z, color = 'red')
ax.set_title('train data')
ax.view_init(50, 60)

ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)


ANN=np.random.rand(8)
initAnn = np.copy(ANN)

#### forward propagation
def forward(ANN, data):
    W0=[ [ANN[0],ANN[2]],
         [ANN[1], ANN[3]] ]
    W1=[ANN[4], ANN[5]]
    B0=ANN[6]
    B1=ANN[7]

    data=np.dot(W0, data) + B0
    data=np.dot(W1, data) + B1
    return data

#### hypothesis function
# def hypothesis(w,x,y):
#     return w[0]*x**3 + w[1]*y**3 + w[2]*x**2 + w[3]*y**2 + w[4]*x + w[5]*y + w[6] 

def hypothesis(ANN,xy):
    output=[]
    m=[]
    for i in xy:
        for j in i:
            m.append( forward(ANN, j) )
        output.append(m)
        m=[]
    return output


#### loss function
def getLoss(ANN):
    return -1 * meanSquaredError(targetFunc(x,y), hypothesis(ANN,xy) )   
    # return -1 * absolutedError(f(x), hypothesis(w) )

#### mse
def meanSquaredError(x,y):
    return np.sum( (x-y)**2 ) / len(x)

def absolutedError(x ,y):
    return np.sum( abs(x-y) ) / len(x)

def nomalize(vector):
    len = math.sqrt( np.sum(vector**2) )
    return vector/len

def getGradient(f, point, h):
    w0= point[0]
    w1= point[1]
    w2= point[2]
    w3= point[3]
    w4= point[4]
    w5= point[5]
    w6= point[6]
    w7= point[7]    
    
    dw0 = ( f( [w0+h, w1, w2, w3, w4, w5, w6, w7] ) - f( [w0, w1, w2, w3, w4, w5, w6, w7] ) )/h
    dw1 = ( f( [w0, w1+h, w2, w3, w4, w5, w6, w7] ) - f( [w0, w1, w2, w3, w4, w5, w6, w7] ) )/h
    dw2 = ( f( [w0, w1, w2+h, w3, w4, w5, w6, w7] ) - f( [w0, w1, w2, w3, w4, w5, w6, w7] ) )/h
    dw3 = ( f( [w0, w1, w2, w3+h, w4, w5, w6, w7] ) - f( [w0, w1, w2, w3, w4, w5, w6, w7] ) )/h
    dw4 = ( f( [w0, w1, w2, w3, w4+h, w5, w6, w7] ) - f( [w0, w1, w2, w3, w4, w5, w6, w7] ) )/h
    dw5 = ( f( [w0, w1, w2, w3, w4, w5+h, w6, w7] ) - f( [w0, w1, w2, w3, w4, w5, w6, w7] ) )/h
    dw6 = ( f( [w0, w1, w2, w3, w4, w5, w6+h, w7] ) - f( [w0, w1, w2, w3, w4, w5, w6, w7] ) )/h
    dw7 = ( f( [w0, w1, w2, w3, w4, w5, w6, w7+h] ) - f( [w0, w1, w2, w3, w4, w5, w6, w7] ) )/h

    gredient = np.array( [dw0, dw1, dw2, dw3, dw4, dw5, dw6, dw7] )
    # return gredient
    return nomalize(gredient) 

def gradientDescent( init, learningRate, f ):
    count=0
    while True:
        step = learningRate * getGradient(f, init, 0.00001)
        lose1 = abs(f(init))
        lose2 = abs(f( init + step ))
        if lose2 < lose1 :
            init = init + step
            count+=1
            if count==10 :
                print(init)
                print(lose2, forward(init, [10,10]))
                count=0
        else :
            return init


def plot(x,y,f,arg):
    xy=cartesianPruducts(x,y)
    x,y = np.meshgrid(x, y)
    z=np.array( f(arg, xy) )

    fig = plt.figure(figsize = (6, 6))

    ax = plt.axes(projection = '3d')
    ax.plot_wireframe(x, y, z, color = 'red')
    ax.view_init(50, 60)

    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    plt.show()

ANN = gradientDescent(ANN, learningRate, getLoss) 

plot( np.arange(-10, 10, 1), np.arange(-10, 10, 1), hypothesis, ANN)
# plot( np.arange(-10, 10, 1), np.arange(-10, 10, 1), hypothesis, initAnn)



