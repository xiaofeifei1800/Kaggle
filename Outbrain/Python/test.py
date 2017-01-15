import sys
from datetime import datetime

start = datetime.now()
def fib3(n):
    stack = [1,1,1]
    if n in [1,2,3]:
        return 1
    else:
        n = n - 3
        while n > 1:
            stack.append(sum(stack))
            stack.pop(0)
            n -= 1
    return sum(stack)

print fib3(20)

print(datetime.now()-start)

start = datetime.now()
def fib4(n):
   if n < 3:
       return 1
   a = b = c = 1
   for i in range(3, n):
       # We "shift" a, b, and c to the next values of the sequence.
       a, b, c = b, c, (a + b + c)
   return c

print fib4(20)

print(datetime.now()-start)