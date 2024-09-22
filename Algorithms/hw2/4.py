n = int(input())
p = 10**6 + 3

def fibonacci(n):
    n1 = 1
    n2 = 2

    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        for i in range(2, n):
            n3 = n1 + n2
            n1 = n2
            n2 = n3
    return n2

print(fibonacci(n) % p)