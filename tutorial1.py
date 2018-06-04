from math import sqrt, pi, sin
import time
def is_prime(x):
    for y in range (2,int(sqrt(x)+1)):
        if(x%y==0):
            return False
    return True

def power(base, exp):
    if exp==0:
        if base==0:
            raise ValueError('undefined behaviour when both base and exponent are 0')
        else:
            return 1
    else:
        res = 1
        for i in range(exp):
            res *= base
        return res


def factorial(n):
    if(n==0 or n==1):
        return 1
    else:
        res = n
        for i in range(1,n):
            res*=i


def my_sine_taylor_element(alpha, n):
    return (float)(power(-1,n))/factorial(2*n+1)*power(alpha,2*n+1)


def my_sine(alpha, order):
    alpha = alpha%(2*pi)
    series = [my_sine_taylor_element(alpha, i) for i in range(order)]
    return sum(series)


def compare_sines(alpha, n):
    return abs(my_sine(alpha,n)-sin(alpha))


def hanoi(n, fromTower, middleTower, toTower):
    print("fromTower is : " + str(fromTower))
    print("middleTower is : " + str(middleTower))
    print("toTower is : " + str(toTower))
    if n<=0:
        raise ValueError('n must be a positive integer')
    elif n==1:
        toTower.append(fromTower.pop())
    else:
        hanoi(n-1,fromTower, toTower, middleTower)
        toTower.append(fromTower.pop())
        hanoi(n - 1, middleTower, fromTower, toTower)

#def recursive_fibbonacci(n):
def iterative_fibbonacci(n):
    if n<0:
        raise ValueError('n must be a positive integer')
    elif n==1 or n==2:
        return 1
    else:
        a,b = 1, 1
        for x in range(n-2):
            a,b = b, a+b
        return b


def recursive_fibbonacci(n):
    if n<0:
        raise ValueError('n must be a positive integer')
    elif n==1 or n==2:
        return 1
    else:
        return recursive_fibbonacci(n-1)+recursive_fibbonacci(n-2)

def testSin():
    alpha = 100
    print my_sine(alpha, 29)
    print sin(alpha)
    print compare_sines(alpha, 9)

def memorization_fibbonacci(n, cache={}):
    if n<0:
        raise ValueError('n must be a positive integer')
    elif n==1 or n==2:
        return 1
    else:
        if n in cache:
            return cache.get(n)
        else:
            cache[n] = memorization_fibbonacci(n-1, cache)+memorization_fibbonacci(n-2, cache)
            return cache[n]


def guessingANumber(arg):
    if arg=="PvP":
        guessingANumberPvP()
    elif arg=="PvC":
        guessingANumberPvC()
    elif arg=="CvC":
        guessingANumberCvC()
    else:
        raise ValueError(str(arg) + ' is an illegal argument')

def guessingANumberPvP():
    player1 = raw_input('player1 - please enter your name: ')
    player2 = raw_input('player2 - please enter your name: ')
    num = raw_input('Welcome, ' + player1 + '! please enter a number for ' + player2 + ' to guess: ')
    guess = raw_input('Welcome ' + player2 + '! please enter your guess: ')
    count = 1
    while guess!=num:
        count+=1
        if(int(guess)>int(num)):
            guess = raw_input("wrong! your guess was too big! try again: ")
        else:
            guess = raw_input("wrong! your guess was too small! try again: ")

    print('Congratiolations, ' + player2 + '! it took you ' + str(count) + ' guesses to win!')


def guessingANumberPvC():
    pass
    # num =


def guessingANumberCvC():
    pass


def testHanoi():
    fromTower = range(42,0,-1)
    toTower = []
    middleTower = []
    hanoi(len(fromTower), fromTower, middleTower, toTower)

    print("fromTower is : " + str(fromTower))
    print("middleTower is : " + str(middleTower))
    print("toTower is : " + str(toTower))


def testFibbonacci(arg,n):
    start=time.time()
    if(arg==0):
        print(iterative_fibbonacci(n))
    elif(arg==1):
        print(recursive_fibbonacci(n))
    elif(arg==2):
        print(memorization_fibbonacci(n))
    else:
        print(memorization_fibbonacci(n)==recursive_fibbonacci(n)==iterative_fibbonacci(n))
    print(time.time()-start)


def takewhile(pred, it):
    res = []
    for a in it:
        if(pred(a)):
            res.append(a)
    return res


def integers():
    """Infinite sequence of integers."""
    i = 1
    while True:
        yield i
        i = i + 1


def primes(n):
    for i in integers():
        if i>n:
            break
        if(is_prime(i)):
            yield i
            i = i+1

def stringGenerator(S):
    stringSet = S
    for s in stringSet:
        yield s


def words(S,n):
    generatorList = []
    res = ""
    for i in range(n):
        generatorList.append(stringGenerator(S))
    for gen in generatorList:
        for s in gen:
           res+=s
        yield res
        res = ""


def testTakeWhile():
    print(list(takewhile(lambda x:is_prime(x) and x<100, iter(range(5,100,6)))))
    # print(list(takewhile(is_prime, iter(range(5, 100, 6)))))

# def
n=12
def testFibo(n):
    # testFibbonacci(0,n)
    # testFibbonacci(1,n)
    # testFibbonacci(2,n)
    testFibbonacci(3,n)


def testPrimes():
    print(list(primes(100)))


def testWords():
    print(list(words({'o','la'},3)))


def baseFivePresentation(num):
    res = 0
    while(num>0):
        res *= 10
        res+=num%5
        num/=5
    return res

def print_aux(obj):
    if (isinstance(obj, (int, long))):
        return str(baseFivePresentation(obj))
    elif (isinstance(obj, basestring)):
        return str(obj + obj)
    elif (isinstance(obj, list)):
        res = []
        for e in reversed(obj):
            if e == obj:
                res.append("@")
            else:
                res.append(print_aux(e))
        return ("<"+"!".join(res)+">")
    else:
        return "?"

def print2(obj):
    # if(isinstance(obj,(int, long, basestring))):
    #     print(print_aux())
    # elif(isinstance(obj, list)):
    #     res = []
    #     for e in reversed(obj):
    #         res.append(print_aux(e))
    # print("<"+"!".join(res)+">")
    print print_aux(obj)

def testPrint2():
    a = [1, 2,"hi", {1}, 8]
    b=[5,4,{9},"hi"]
    b[1]=a
    # a[1] = b
    print2(a)

# testSin()
# testFibo(n)
# guessingANumberPvP()
# testHanoi()
# testPrimes()
# testTakeWhile()
# testWords()
# print(baseFivePresentation(5))
# testPrint2()

def main():
    start = time.time()
    factorial(100)
    end = time.time()

if __name__ == "__main__":
    main()