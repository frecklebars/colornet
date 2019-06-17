
log = open("data.txt", "w")

#finding if text should be black (0) or white (1)
def findTextCol(color):
    threshold = 150
    target = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
    if target > threshold:
        target = 0
    else: 
        target = 1
    return target

#generate data set
data = [[[-1 for i in range(256)] for i in range(256)] for i in range(256)]

for R in range(256):
    for G in range(256):
        for B in range(256):
            data[R][G][B] = findTextCol([R, G, B])
            log.write(str(data[R][G][B]) + "\n")
