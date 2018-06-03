# import Hello
#
# Hello.checkPrime(x)
#
from math import sqrt,ceil,sin,pi
# from math import ceil

def checkPrime(x):
    ''' returns true if x is prime'''
    result = True

    for i in range(2,int(ceil(sqrt(x)))):
        # print(i)
        if int(x/i)*i==x:
            result = False

    return result

def powerFun(arg,power):
    result = 1
    for i in range(0,power):
        result = result*arg
    return result

def fuctorialFun(x):
    result = x
    for i in range(x-1,1,-1):
        result = result*i
    return result

def taylorSine(x,steps = 10):
    x = x%(2*pi)
    # print(x)
    # print(type(x))
    result = 0
    for n in range(steps):
        result = result + powerFun(-1,n)*powerFun(x,2*n+1)/fuctorialFun(2*n+1)
    return result

def hanoyTest(inputList,targetList):
    for i in range(len(inputList)-1):
        if inputList[i] > inputList[i + 1]:
            print("input discs are not in order")
            print(i)
    for i in range(len(targetList)-1):
        if targetList[i] > targetList[i + 1]:
            print("target discs are not in order")
            print(i)


def towerHaony(inputList,targetList,utilList = []):
    if len(inputList==1):
        # Top disc
        utilList.append(inputList[-1])  # copy top disc
        inputList.remove[-1]  # remove top disc
        #bottome Disc
        targetList.append(inputList[-1])  # copy top disc
        inputList.remove[-1]  # remove top disc
        # # Top disc
        # target.append(utilList[-1])  # copy top disc
        # utilList.remove[-1]  # remove top disc
        return (inputList,targetList)
    else:
        towerHanoy(inputList[1,:],inputList[0])

    return targetList


def iterFib(index):
    a0 = 1
    a1 = 1
    for n in range(index-1):
        a2 = a0+a1
        a0 = a1
        a1 = a2
    return a2


def recFibIn(a0,a1,index):
    if index<2:
        return a1
    else:
        return recFibIn(a1,a0+a1,index-1)

def recFib(index):
    a0 = 1
    a1 = 1
    return recFibIn(a0,a1,index)
    # return (recFib(n-1)+recFib(n-1)) if n>1 else return 1

def memoFib(index,inDict = {0 : 1 , 1 : 1}):

    if index in inDict:
        return inDict[index]
    elif index-1 in inDict:
        inDict[index] = inDict[index-1] + inDict[index-2]
        return inDict[index]
    else:
        inDict[index] = memoFib(index-2) + memoFib(index-1)
    return inDict[index]

def intGen():
    i = 1
    while True:
        yield i
        i += 1

def intGenPrime():
    I = intGen()
    while True:
        x = I.next()
        # print(x)
        if checkPrime(x):
            yield x
        # else:
        #     I.next()

# if str(index) in inDict:
#     return inDict[str(index)]
# else if str(index) in inDict[str(index-1)] & & str(index) in inDict[str(index-2)]:
#     return inDict[str(index - 1)] + inDict[str(index - 2)]
# else:
# return

##### main

I = intGenPrime()
for i in range(6):
    print(I.next())
# P = intGenPrime()
# print(P.next())
# print(P.next())
# print(P.next())
# print(P.next())
# print(P.next())

# list = ['a','b','c']
# print("__".join(list))

# print(iterFib(15))

# for x in [0.01,0.5*pi,6,100]:
#     print([x, sin(x), taylorSine(x, steps=10)])
# x = 6
# print([x,sin(x),taylorSine(x,steps = 10)])

# print(fuctorialFun(2))

# for i in range(0,5):
#     print i
#     y = powerFun(3,i)
#     print(y)


# x=11
# result = checkPrime(x)
# print(result)