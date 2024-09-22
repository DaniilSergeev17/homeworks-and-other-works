# Малая Теорема Ферма
# (a/b + c/d) % p = (a/b % p + c/d % p) % p
# (a/b) % m = a * b**(-1) % m = a * b**(m-2) % m -> ** -> fast_pow

a, b, c, d = map(int, input().split())
p = 10**9 + 7

def fast_pow(x, n):
    if n == 0:
        return 1
    elif n % 2 == 0:
        return fast_pow(x*x % p, n//2) 
    else:
        return x * fast_pow(x, n-1) % p

def teorema_ferma(a, b, c, d):
    return ((a*(fast_pow(b, p-2))) % p + (c*(fast_pow(d, p-2))) % p) % p

print(teorema_ferma(a, b, c, d))