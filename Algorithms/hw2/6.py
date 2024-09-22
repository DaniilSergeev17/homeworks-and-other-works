# Малая Теорема Ферма
# (a * b) % p = (a % p * b % p) % p

n = int(input())
p = 10**6 + 3

def factorial_with_teorema_ferma(n):
    if n >= p: # т.к если n >= p, то n! содержит в себе p => делится нацело
        return 0
    
    result = 1
    for i in range(2, n + 1):
        result = (result * i) % p 
    return result

print(factorial_with_teorema_ferma(n))