m = int(input())
s = [''] * m
for i in range(m):
    s[i] = input()

k = int(input())
x = map(int, input().split())
unordered_chain = [s[j-1] for j in x]

MOD = 10**9 + 9
N = 10**5 + 1
base = [1] * N

def get_hash(s: str) -> int:
    h = 0
    for char in s:
        h = (h * base[1] + ord(char)) % MOD
    return h

for i in range(1, N):
    base[i] = base[i - 1] * 257 % MOD

l, r = 0, 0
res = []
for i in range(len(unordered_chain)-1):
    t = unordered_chain[i]
    target_str = unordered_chain[i+1]

    t_hash = get_hash(t)
    target_hash = get_hash(target_str[:len(t)])

    if len(target_str) > len(t) and t_hash == target_hash:
        r += 1
        res.append([l+1, r+1])
    else:
        l += 1


def merge_intervals(res):
    final_res = []
    current_interval = res[0]

    for next_interval in res[1:]:
        if current_interval[1] == next_interval[0]: 
            current_interval[1] = next_interval[1]
        else:
            final_res.append(current_interval)
            current_interval = next_interval
    final_res.append(current_interval)

    return final_res

maxi_ans = -float('inf')
final_ans = ''
for interval in merge_intervals(res):
    sub = interval[1] - interval[0]
    if sub > maxi_ans:
        maxi_ans = sub
        final_ans = ' '.join(map(str, interval))

print(final_ans)