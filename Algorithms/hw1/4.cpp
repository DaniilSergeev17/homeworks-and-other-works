#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

int main() {
    int n = 1;
    std::cin >> n;
    std::vector<int> nums(n);
    for (int i = 0; i < n; i++) {
        std::cin >> nums[i];
    }

    int maxi = *std::max_element(nums.begin(), nums.end());
    int limit = maxi * 20;
    std::vector<int> eratosfen(limit+1, 0);
    for (int i = 2; i * i <= limit; i++) {
        if (eratosfen[i] == 0) { // количество делителей (= 0 -> значит простое)
            for (int j = i * i; j <= limit; j += i) {
                eratosfen[j] = 1;
            }
        }
    }

    std::vector<int> all_simple;
    all_simple.push_back(0);
    for (int i = 2; i <= limit; i++) {
        if (eratosfen[i] == 0) {
            all_simple.push_back(i);
        }
    }

    for (int i = 0; i < n; i++) {
        std::cout << all_simple[nums[i]] << ' ';
    }
}