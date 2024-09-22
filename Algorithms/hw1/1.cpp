#include <iostream>
#include <cmath>

int main() {
    bool flag = true;
    int n = 2;
    std::cin >> n;
    for (int i = 2; i <= std::sqrt(n); i++) {
        if (n % i == 0) {
            std::cout << i;
            flag = false;
            break;
        }
    }
    if (flag) {
        std::cout << n;
    }
}














