# running time of factorial recursive ~1.1e-06
# running time of factorial iterative ~8e-07
import timeit

def factorial_recursive(n):
    if n==0:
        return 1
    return n * factorial_recursive(n-1)

def factorial_iterative(n):
    sum = 1
    for i in range(1,n+1):
        sum *= i
    return sum

t = timeit.Timer(lambda: factorial_recursive(10))
t2 = timeit.Timer(lambda: factorial_iterative(10))
sum1 = 0
sum2 = 0
num_times = 1000
for i in range(num_times):
    curr_time1 = t.timeit(number=1)
    sum1 += curr_time1
    curr_time2 = t2.timeit(number=1)
    sum2 += curr_time2
avg = sum1 / num_times
avg2 = sum2 / num_times
print('recursive time is: %.4E' % avg)
print('iterative time is: %.4E' % avg2)

