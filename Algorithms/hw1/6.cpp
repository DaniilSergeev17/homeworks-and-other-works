#include <iostream>
#include <vector>
#include <cmath>
#include <set>

int main() {
    int l = 1;
    int r = 1;
    std::cin >> l >> r;

    int limit = r;
    std::vector<bool> eratosfen(limit+1, 0);
    for (int i = 2; i * i <= limit; i++) {
        if (eratosfen[i] == 0) { // простое или нет (= 0 -> значит простое)
            for (int j = i * i; j <= limit+1; j += i) {
                eratosfen[j] = 1;
            }
        }
    }

    std::vector<int> primes;
    for (int i = 2; i <= limit+1; i++) {
        if (!eratosfen[i]) {
            primes.push_back(i);
        }
    }

    std::set<int> fin_set;
    for (int num = l; num <= r; num++) {
        int temp_num = num;
        for (int prime : primes) {
            if (prime * prime > num) break;
            if (temp_num % prime == 0) {
                fin_set.insert(prime);
                while (temp_num % prime == 0) {
                    temp_num /= prime;
                }
            }
        }
        if (temp_num > 1) {
            fin_set.insert(temp_num);
        }
    }

    std::cout << fin_set.size() << "\n";
}