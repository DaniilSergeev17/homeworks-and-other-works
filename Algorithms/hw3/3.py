s = input()
m = int(input())

MOD = 10**9 + 9
N = 10**5 + 1
base = [1] * N
pref = []

def get_hash(s: str) -> int:
    hass = 0
    for char in s:
        hass = ((hass * base[1]) % MOD + ord(char) % MOD) % MOD
    return hass

def get_substring_hash(left: int, right: int) -> int:
    h = (pref[right] - pref[left - 1] * base[right - left + 1]) % MOD
    return h

for i in range(1, N):
    base[i] = base[i - 1] * 257 % MOD
s = '0' + s
pref.extend([0] * len(s))

# префиксные хэши для строки s
for i in range(1, len(s)):
        pref[i] = ((pref[i - 1] * base[1]) % MOD + ord(s[i]) % MOD) % MOD

res = []
for i in range(m):
    a, b, c, d = map(int, input().split())
    hash_s1, hash_s2 = get_substring_hash(a, b), get_substring_hash(c, d)
    if hash_s1 == hash_s2:
        res.append('Yes')
    else:
        res.append('No')


print('\n'.join(res))