#include <iostream>
#include <vector>
#include <set>

int main() {
    int m, n;
    std::cin >> m >> n;

    std::vector<int> eratosfen(n + 1, 1); 
    for (int i = 2; i <= n; ++i) {
        for (int j = 2 * i; j <= n; j += i) {
            eratosfen[j] += i;
        }
    }

    std::set<std::pair<int, int>> result;
    for (int i = m; i <= n; ++i) {
        int summ_del = eratosfen[i];
        if (summ_del > i && summ_del <= n) {
            if (eratosfen[summ_del] == i) {
                result.insert({i, summ_del});
            }
        }
    }

    if (result.empty()) {
        std::cout << "Absent" << "\n";
    } 
    else {
        for (const auto& res : result) {
            std::cout << res.first << " " << res.second << "\n";
        }
    }
}