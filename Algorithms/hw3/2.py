m = int(input()) # size of hash table
n = int(input())
p = 10**9 + 7
x = 263

class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for i in range(size)]

    def get_hash(self, key: str) -> int:
        hass = 0
        for i in range(len(key)):
            hass += (ord(key[i]) * x**i) % p
        hass %= p
        return hass % self.size
    
    def add(self, key: str) -> None:
        hashed_key = self.get_hash(key)
        self.table[hashed_key].append(key) if key not in self.table[hashed_key] else None

    def find(self, key: str) -> str:
        hashed_key = self.get_hash(key)
        for k in self.table[hashed_key]:
            if k == key:
                return 'yes'
        return 'no'
    
    def del_key(self, key: str) -> None:
        hashed_key = self.get_hash(key)
        self.table[hashed_key] = [k for k in self.table[hashed_key] if k != key]

    def check(self, idx: int) -> str:
        result = self.table[idx][::-1]
        return ' '.join(result)

res = []
phone_book = HashTable(size=m)
for i in range(n):
    s = input().split()
    if s[0] == 'add':
        op, name = s
        phone_book.add(name)
    else:
        op, num = s
        if op == 'del':
            phone_book.del_key(num)
        elif op == 'find':
            res.append(phone_book.find(num))
        elif op == 'check':
            res.append(phone_book.check(int(num)))
        
print('\n'.join(res))