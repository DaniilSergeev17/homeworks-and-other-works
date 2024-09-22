# т.к. a_i = (a_(i-1))**2 % 10_000, то:
# a_n = a_i ** (2 ** (n-1)) % 10_000

a1, n = map(int, input().split())
p = 10_000

def fast_pow(a, n):
    if n == 0:
        return 1
    elif n % 2 == 0:
        return fast_pow(a * a % p, n//2) % p
    else:
        return a * fast_pow(a, n-1) % p

middle_res = fast_pow(2, n-1)
print(fast_pow(a1, middle_res))