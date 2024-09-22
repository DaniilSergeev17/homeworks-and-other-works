#include <iostream>
#include <vector>
#include <cmath>

int main() {
    int n = 1;
    std::cin >> n;

    int limit = n * 2; // bottleneck
    std::vector<bool> eratosfen(limit+1, 0);
    for (int i = 2; i * i <= limit; i++) {
        if (eratosfen[i] == 0) { // простое или нет (= 0 -> значит простое)
            for (int j = i * i; j <= limit; j += i) {
                eratosfen[j] = 1;
            }
        }
    }

    int cnt = 0;
    for (int i = 2; i <= n; i++) {
        if (!eratosfen[i]) {
            cnt++;
        }
    }

    std::cout << cnt;
}