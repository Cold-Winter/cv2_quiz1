import random
#print random.randint(1,100)
def uniformsample():
    partition = []
    templist = []
    idlist = range(1,1001)
    random.shuffle(idlist)

    for i,value in enumerate(idlist):
        templist.append(idlist[i])
        if i % 10 == 0:
            partition.append(templist)
            templist = []
    samplen = random.randint(1,100)
    return partition[samplen]
    
a = uniformsample()
print a
    
    
        
        