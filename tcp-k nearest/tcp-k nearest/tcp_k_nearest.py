import numpy as np

#getting RGB values of the background
def getColor():
    colR = np.random.randint(0, 255)
    colG = np.random.randint(0, 255)
    colB = np.random.randint(0, 255)
    color = [colR, colG, colB]
    return(color)

#finding if text should be black (0) or white (1)
def findTextCol(color):
    threshold = 150
    target = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
    if target > threshold:
        target = 0
    else: 
        target = 1
    return target

#generated text color for every RGB combination in data.txt using genData.py
#reading the data now

log = open("data.txt", "r")

data = [[[-1 for i in range(256)] for i in range(256)] for i in range(256)]
print("reading data...")
for R in range(256):
    for G in range(256):
        for B in range(256):
            data[R][G][B] = log.readline()
print("done!")

#get color
color = getColor()

