//все прошло, оптимально

#include<iostream>
#include<string>
#include<vector>
#include<cstdint>

const int64_t MOD = 1e9 + 9;
const int64_t N = 1e5 + 1;
std::vector<int64_t> base(N);
std::vector<int64_t> pref;
int64_t get_hash(const std::string& s)
{
    int64_t h = 0;
    for(const auto& to: s)
    {
        h = (h * base[1] + to) % MOD;
    }
    return h;
}

int64_t get_hash(size_t left, size_t right)
{
    int64_t h = pref[right] - pref[left - 1] * base[right - left + 1];
    h = (h % MOD + MOD) % MOD;
    return h;
}

int main() {
    base[0] = 1;
    for(size_t i = 1; i < N; ++i)
    {
        base[i] = base[i - 1] * 257 % MOD;
    }
    std::string t = "";
    std::string target_str = "";
    std::cin >> t >> target_str;
    t = '\0' + t;
    pref.resize(t.size());
    for(size_t i = 1; i < t.size(); ++i)
    {
        pref[i] = (pref[i - 1] * base[1] + t[i]) % MOD;
    }

    int64_t target_hash = get_hash(target_str);
    std::vector<int> res;
    for (int i = 1; i <= t.size() - (target_str.size()-1); i++) {
        if (target_hash == get_hash(i, i+(target_str.size()-1))) {
            res.push_back(i);
        }
    }

    std::cout << res.size() << "\n";
    for (int i : res) {
        std::cout << i << " ";
    }
}
