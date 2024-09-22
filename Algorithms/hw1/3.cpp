#include <iostream>
#include <cmath>
#include <vector>


int main() {
    int n = 2;  
    std::cin >> n;
    std::vector<int> eratosfen(n+1, 0);

    for (int i = 2; i <= n; i++) {
        if (eratosfen[i] == 0) { // количество делителей (= 0 -> значит простое)
            for (int j = i; j <= n; j += i) {
                eratosfen[j] += 1;
            }
        }
    }

    for (int i = 2; i <= n; i++) {
        if (eratosfen[i] >= 3) {
            std::cout << i << ' ';
        }
    }
}