#i hope this change only in feature/recursive
# running time of factorial recursive 1.10864639282e-06
import timeit

def factorial(n):
    if n==0:
        return 1
    return n * factorial(n-1)

def factorial2(n):
    sum = 1
    for i in range(1,n+1):
        sum *= i
    return sum

t = timeit.Timer(lambda: factorial(10))
t2 = timeit.Timer(lambda: factorial2(10))
sum = 0
num_times = 1000
for i in range(num_times):
    curr_time = t.timeit(number=1)
    sum += curr_time
avg = sum / num_times
print(avg)
# num = int(input("Enter number\n"))
# print(factorial(num))
# print(factorial2(num))
