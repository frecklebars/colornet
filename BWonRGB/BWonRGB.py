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

w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
b = np.random.randn()
rate = 0.2

def sigmoid(x):
    x = 1/(1 + np.exp(-x))
    return x
#derivative of sigmoid
def sigmoid_p(x):
    x = sigmoid(x) * (1-sigmoid(x))
    return x

def predict(w1, w2, w3, b, data):
    z = w1 * data[0] + w2 * data[1] + w3 * data[2] + b
    return z

#TRAIN LOOP
train_count = 50000
for i in range(train_count):
    #get dataset
    data = getColor()
    target = findTextCol(data)

    #FEED FORWARD
    z = predict(w1, w2, w3, b, data)
    pred = sigmoid(z)

    #BACK PROPAGATION
    cost = (pred - target) ** 2

    #calculating derivatives
    dcost_pred = 2 * (pred - target)
    dpred_z = sigmoid_p(z)
    dz_w1 = data[0]
    dz_w2 = data[1]
    dz_w3 = data[2]
    dz_b = 1

    dcost_w1 = dcost_pred * dpred_z * dz_w1
    dcost_w2 = dcost_pred * dpred_z * dz_w2
    dcost_w3 = dcost_pred * dpred_z * dz_w3
    dcost_b = dcost_pred * dpred_z * dz_b

    #update weights and bias
    w1 = w1 - rate * dcost_w1
    w2 = w2 - rate * dcost_w2
    w3 = w3 - rate * dcost_w3
    b = b - rate * dcost_b

#TEST
white = 0
black = 0
whiteP = 0
blackP = 0
for i in range(100):
    color = getColor()
    textCol = findTextCol(color)
    if textCol == 1:
        textCol = "black"
        black += 1
    else:
        textCol = "white"
        white += 1
    z = predict(w1, w2, w3, b, color)
    pred = sigmoid(z)
    if pred > 0.50:
        pred = "black"
        blackP += 1
    else:
        pred = "white"
        whiteP += 1
print("t:", black, white, "p:", blackP, whiteP)
    
    
