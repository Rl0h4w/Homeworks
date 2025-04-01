#include <iostream>
#include <vector>

int main()
{
    int n{0};
    std::vector<int> m;
    std::cin >> n;
    for (int i{0}; i < n; ++i)
    {
        std::cin >> m[i];
    }
    return 0;
}