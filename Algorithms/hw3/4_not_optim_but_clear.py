# НЕоптимальное, не прошло какой-то из тестов по времени

s = input()
target_s = input()
p = 10**9 + 9
x = 263

def fast_pow(a, n):
     if n == 0:
        return 1
     elif n % 2 == 0:
        return fast_pow(a*a % p, n//2)
     else:
        return a * fast_pow(a, n-1) % p

def get_hash(string: str) -> int:
        hass = 0
        for i in range(len(string)):
            hass += (ord(string[i]) % p * fast_pow(x, i) % p) % p
        hass %= p
        return hass

target_hash = get_hash(target_s)
cnt = 0
res = []
for i in range(len(s) - len(target_s) + 1):
    if get_hash(s[i:i+len(target_s)]) == target_hash:
         cnt += 1
         res.append(i+1)
      

print(cnt)
print(' '.join(map(str, res)))