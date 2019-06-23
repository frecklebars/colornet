import numpy as np

#getting RGB values of the background
def getColor():
    colR = np.random.randint(0, 255) / 255
    colG = np.random.randint(0, 255) / 255
    colB = np.random.randint(0, 255) / 255
    color = [[colR], [colG], [colB]]
    return(color)

#finding if text should be black (0) or white (1)
def findTextCol(color):
    threshold = 150 / 255
    target = color[0][0] * 0.299 + color[1][0] * 0.587 + color[2][0] * 0.114
    if target > threshold:
        target = 0
    else: 
        target = 1
    return target

#declaring how many nodes(neurons) per layer there will be
nodes_in = 3 #for R G and B
nodes_h = 3
nodes_out = 1

#initialising weights and biases
wH = np.array([[np.random.randn() for x in range(nodes_in)] for x in range(nodes_h)])
bH = np.array([[np.random.randn()] for x in range(nodes_h)])

wOut = np.array([[np.random.randn() for x in range(nodes_h)] for x in range(nodes_out)])
bOut = np.array([[np.random.randn()] for x in range(nodes_out)])

#activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))

#learn rate
rate = 0.2

for i in range(1):
    color = getColor()
    target = findTextCol(color)

    #feed forward
    zH = np.dot(wH, color) + bH
    predH = sigmoid(zH)
    
    z = np.dot(wOut, predH) + bOut
    pred = sigmoid(z)

    #back propagate
    cost = (pred - target) ** 2

    dcost_pred = 2 * (pred-target)
    dpred_z = sigmoid_p(z)
    dz_wOut = predH
    dz_bOut = 1

    dcost_wOut = dcost_pred * dpred_z * dz_wOut
    dcost_bOut = dcost_pred * dpred_z * dz_bOut
    dcost_wOut.shape = (nodes_out, nodes_h)

    #update wOut and bOut
    wOut =  wOut - rate * dcost_wOut
    bOut =  bOut - rate * dcost_bOut

'''
    TO DO

>   look into updating hidden layer weights and biases
>   i have a suspicion about what was wrong in the first version, which might be
    related to the way i calculated the original weight updates
>   keep learning

'''