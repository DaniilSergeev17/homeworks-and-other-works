# оптимальное, но не прошло какой-то из тестов

s = input()
target_s = input()

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
    if left == 0:
        return pref[right]  
    h = (pref[right] - (pref[left - 1] * base[right - left + 1]) % MOD + MOD) % MOD
    return h

for i in range(1, N):
    base[i] = base[i - 1] * 257 % MOD

# префиксные хэши для строки s
pref = [0] * len(s)
for i in range(len(s)):
    pref[i] = ((pref[i - 1] * base[1]) % MOD + ord(s[i]) % MOD) % MOD if i > 0 else ord(s[i]) % MOD


target_hash = get_hash(target_s)
res = []
for i in range(len(s) - len(target_s) + 1):
    hash_s1 = get_substring_hash(i, i + len(target_s) - 1)
    if hash_s1 == target_hash:
        res.append(i+1)

print(len(res))
print(' '.join(map(str, res)))
