import time
import checkPrime, factorialEx ,factorialRec
# main
# a = factorialEx.factorialFun(20)
# a = checkPrime.checkPrime(7)
# print(a)
arrIt = []
arrRec = []
start = time.time()
for i in range(10):
    arrIt.append(factorialEx.factorialFun(30))

elapsedIt = time.time()-start

start = time.time()
for i in range(10):
    arrRec.append(factorialRec.factorialRec(30))

elapsedRec = time.time()-start

print(elapsedIt)
print(elapsedRec)


with open('facTimeLog.txt','w') as f:
    f.write('Trial\n Iterative time = ' + str(elapsedIt) +'\n Recursive time = ' + str(elapsedRec) +'\n ')
