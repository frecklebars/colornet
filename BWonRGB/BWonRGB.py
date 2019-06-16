import numpy as np

log = open("log.txt", "w")

#getting RGB values of the background
def getColor():
    colR = np.random.randint(0, 255)
    colG = np.random.randint(0, 255)
    colB = np.random.randint(0, 255)
    color = [colR, colG, colB]
    return(color)

#finding if text should be black (0) or white (1)
#threshold of 186 is based on theory but can be adjusted to taste
def findTextCol(color):
    threshold = 150
    target = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
    if target > threshold:
        target = 0
    else: 
        target = 1
    return target

#params
#2 input nodes > 2 hidden nodes > 1 output node
w11 = np.random.randn()
w12 = np.random.randn()
w13 = np.random.randn()
b1 = np.random.randn()

w21 = np.random.randn()
w22 = np.random.randn()
w23 = np.random.randn()
b2 = np.random.randn()

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
rate = 0.2

def sigmoid(x):
    x = 1/(1 + np.exp(-x))
    return x
#derivative of sigmoid
def sigmoid_p(x):
    x = sigmoid(x) * (1-sigmoid(x))
    return x

#TRAIN LOOP
train_count = 50000
for i in range(train_count):
    #get dataset
    data = getColor()
    target = findTextCol(data)

    #FEED FORWARD
    z1 = w11 * data[0] + w12 * data[1] + w13 * data[2] + b1
    z2 = w21 * data[0] + w22 * data[1] + w23 * data[2] + b2
    pred1 = sigmoid(z1)
    pred2 = sigmoid(z2)

    z = pred1 * w1 + pred2 * w2 + b
    pred = sigmoid(z)

    #BACK PROPAGATION
    cost = (pred - target) ** 2

    #calculating derivatives for hidden layer
    dcost_pred = 2 * (pred - target)
    dpred_z = sigmoid_p(z)

    dz_w1 = pred1
    dz_w2 = pred2
    dz_b = 1

    dcost_w1 = dcost_pred * dpred_z * dz_w1
    dcost_w2 = dcost_pred * dpred_z * dz_w2
    dcost_b = dcost_pred * dpred_z * dz_b

    #update weights and bias for hidden layer
    w1 = w1 - rate * dcost_w1
    w2 = w2 - rate * dcost_w2
    b = b - rate * dcost_b

    #calculating derivatives for input layer
    dcost1_pred1 = 2 * (pred1 - target)
    dpred1_z1 = sigmoid_p(z1)

    dz1_w11 = data[0]
    dz1_w12 = data[1]
    dz1_w13 = data[2]
    dz1_b1 = 1

    dcost1_w11 = dcost1_pred1 * dpred1_z1 * dz1_w11
    dcost1_w12 = dcost1_pred1 * dpred1_z1 * dz1_w12
    dcost1_w13 = dcost1_pred1 * dpred1_z1 * dz1_w13
    dcost1_b1 = dcost1_pred1 * dpred1_z1 * dz1_b1

    dcost2_pred2 = 2 * (pred2 - target)
    dpred2_z2 = sigmoid_p(z2)

    dz2_w21 = data[0]
    dz2_w22 = data[1]
    dz2_w23 = data[2]
    dz2_b2 = 1

    dcost2_w21 = dcost2_pred2 * dpred2_z2 * dz2_w21
    dcost2_w22 = dcost2_pred2 * dpred2_z2 * dz2_w22
    dcost2_w23 = dcost2_pred2 * dpred2_z2 * dz2_w23
    dcost2_b2 = dcost2_pred2 * dpred2_z2 * dz2_b2

    #update weights and bias for hidden layer
    w11 = w11 - rate * dcost1_w11
    w12 = w12 - rate * dcost1_w12
    w13 = w13 - rate * dcost1_w13
    b1 = b1 - rate * dcost1_b1

    w21 = w21 - rate * dcost2_w21
    w22 = w22 - rate * dcost2_w22
    w23 = w23 - rate * dcost2_w23
    b2 = b2 - rate * dcost2_b2


#TEST
white = 0
black = 0
whiteP = 0
blackP = 0
for i in range(100):
    #find color and ideal text color
    color = getColor()
    textCol = findTextCol(color)
    if textCol == 1:
        textCol = "black"
        black += 1
    else:
        textCol = "white"
        white += 1

    #predict output
    z1 = w11 * data[0] + w12 * data[1] + w13 * data[2] + b1
    z2 = w21 * data[0] + w22 * data[1] + w23 * data[2] + b2
    pred1 = sigmoid(z1)
    pred2 = sigmoid(z2)

    z = pred1 * w1 + pred2 * w2 + b
    pred = sigmoid(z)

    #set text color based on prediction
    if pred > 0.50:
        pred = "black"
        blackP += 1
    else:
        pred = "white"
        whiteP += 1
print("t:", black, white, "p:", blackP, whiteP)
    
    
