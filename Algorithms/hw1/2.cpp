#include <iostream>
#include <cmath>

bool is_simple(int m) {
    for (int i = 2; i <= std::sqrt(m); i++) {
        if (m % i == 0) {
            return false;
        }
    }
    return true;
}

int main() {
    int n = 1;
    std::cin >> n;
    for (int i = 2; i <= std::sqrt(n); i++) {
        while (n % i == 0 && is_simple(i)) {
            std::cout << i << ' ';
            n /= i;
        }
    }

    if (n != 1) {
        std::cout << n << std::endl;
    }
}