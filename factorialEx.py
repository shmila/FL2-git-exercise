def factorialFun(x):
    result = x
    for i in range(x-1,1,-1):
        result = result*i
    return result