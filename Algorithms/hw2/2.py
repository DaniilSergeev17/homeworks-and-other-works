a = float(input())
n = int(input())

def fast_power(a, n):
    if n == 0:
        return 1
    elif n % 2 == 0:
        return fast_power(a*a, n//2)
    else:
        return a * fast_power(a, n-1)

print(fast_power(a, n))