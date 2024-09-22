# Малая Теорема Ферма
# (a**2 - b**2) % p = (a**2 % p - b**2 % p) % p

a, b = map(int, input().split())
p = 10**6 + 7

def sub(a, b):
    return (a**2 % p - b**2 % p) % p

print(sub(a, b))