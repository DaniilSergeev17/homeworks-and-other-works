# a⋅x−1 === 0 (% 10**9+9) -> (по модулю 10**9+9)
# a⋅x === 1 (% 10**9+9) => x - обратное к a (x = a**-1)
# Малая Теорема Ферма (если p простое): a**(p-1) === 1 (%p) -> разделим все на a
# и получим: a**(p-2) === a**-1 (%p)
# а т.к x - обратное к a, то: x === a**(p-2) (%p)

p = 10**9 + 9
t = int(input())

def fast_pow(a, n):
    if n == 0:
        return 1
    elif n % 2 == 1: 
        return a * fast_pow(a, n - 1) % p
    else:
        return fast_pow(a * a % p, n // 2)

def teorema_ferma(a):
    return fast_pow(a, p-2)


for i in range(t):
    a = int(input())
    print(teorema_ferma(a))