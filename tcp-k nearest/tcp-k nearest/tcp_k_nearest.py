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
            if np.random.randint(1,100) > 95:
                data[R][G][B] = log.readline()
            else:
                data[R][G][B] = 2
                log.readline()
print("done!")

#get color
color = getColor()

#deciding the numbers of neighbors
k = 13
neighbors = [-1 for i in range(k)]

def calcDistance(coords1, coords2):
    sum = 0
    for i in range(len(coords1)):
        sum = sum + (coords1[i] - coords2[i]) ** 2
    return np.sqrt(sum)

#calc distance for every point
for R in range(len(data)):
    for G in range(len(data[R])):
        for B in range(len(data[R][G])):
            print("checking R="+str(R)+", G="+str(G)+", B="+str(B)+"\r")
            if data[R][G][B] == 2:
                continue
            distance = calcDistance(color, [R, G, B])
            for i in range(len(neighbors)):
                if neighbors[i] == -1:
                    neighbors[i] = [distance, data[R][G][B]]
                    break
                elif neighbors[i][0] > distance:
                    neighbors[i] = [distance, data[R][G][B]]
                    break

print(neighbors)
