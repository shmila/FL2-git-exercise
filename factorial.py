def factorial(n):
    if n==0:
        return 1
    return n * factorial(n-1)

def factorial2(n):
    sum = 1
    for i in range(1,n+1):
        sum *= i
    return sum


num = int(input("Enter number\n"))
print(factorial(num))
print(factorial2(num))