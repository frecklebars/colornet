import numpy as np

#getting RGB values of the background
def getColor():
    colR = np.random.randint(0, 255) / 255
    colG = np.random.randint(0, 255) / 255
    colB = np.random.randint(0, 255) / 255
    color = [[colR, colG, colB]]
    return(color)

#finding if text should be black (0) or white (1)
def findTextCol(color):
    threshold = 150 / 255
    target = color[0][0] * 0.299 + color[0][1] * 0.587 + color[0][2] * 0.114
    if target > threshold:
        target = 0
    else: 
        target = 1
    return target

#declaring how many nodes(neurons) per layer there will be
nodes_in = 3 #for R G and B
nodes_h = 4
nodes_out = 1

#initialising weights and biases
wH = np.random.rand(nodes_in, nodes_h)
bH = np.random.rand(1, nodes_h)

wOut = np.random.rand(nodes_h, nodes_out)
bOut = np.random.rand(1, nodes_out)

#activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))

#learn rate
rate = 0.2

#NN
train_loops = 50000
for i in range(train_loops):
    color = np.array(getColor())
    target = findTextCol(color)

    #feed forward
    zH = np.dot(color, wH) + bH
    predH = sigmoid(zH)
    
    z = np.dot(predH, wOut) + bOut
    pred = sigmoid(z)

    #back propagate
    #from output layer to hidden layer
    cost = (pred - target) ** 2

    dcost_pred = 2 * (pred-target)
    dpred_z = sigmoid_p(z)
    dz_wOut = predH
    dz_bOut = 1

    dcost_wOut = np.dot(dz_wOut.T, dcost_pred * dpred_z)
    dcost_bOut = dcost_pred * dpred_z * dz_bOut

    #from hidden layer to input layer
    dcost_z = dcost_pred * dpred_z

    dz_predH = wOut
    dcost_predH = np.dot(dcost_z, dz_predH.T)

    dpredH_zH = sigmoid_p(zH)
    dzH_wH = color
    dzH_bH = 1

    dcost_wH = np.dot(dzH_wH.T, dcost_predH * dpredH_zH)
    dcost_bH = dcost_predH * dpredH_zH * dzH_bH

    #update weights and biases
    wOut =  wOut - rate * dcost_wOut
    bOut =  bOut - rate * dcost_bOut
    wH = wH - rate * dcost_wH
    bH = bH - rate * dcost_bH

#TEST
tests = 100000
pred_black = 0
pred_white = 0
t_black = 0
t_white = 0
wrong = 0

for i in range(tests):
    #get color to check
    color = getColor()
    target = findTextCol(color)

    if target == 0:
        t_black += 1
    else:
        t_white += 1

    #feed forward
    zH = np.dot(color, wH) + bH
    predH = sigmoid(zH)
    z = np.dot(predH, wOut) + bOut
    pred = sigmoid(z)

    if pred < 0.50:
        pred_black += 1
    else:
        pred_white += 1

    if pred < 0.50:
        pred = 0
    else:
        pred = 1

    if pred != target:
        wrong += 1

succ = ((tests-wrong)/tests)*100
print(train_loops, "training loops and", nodes_h, "neurons in the hidden layer")
print("done", tests, "tests\n")
print("black count: ", t_black, "\npredicted black: ", pred_black)
print("\nwhite count: ", t_white, "\npredicted white: ", pred_white)

print("\nsuccess rate: %.2f" % succ, "%")