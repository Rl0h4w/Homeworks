#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int countValidCombinations(int n, int k, int min_gap = 3)
{
    vector<int> nums(n);
    for (int i = 0; i < n; ++i)
    {
        nums[i] = i + 1;
    }

    vector<bool> select(n, false);
    fill(select.end() - k, select.end(), true);

    int count = 0;
    do
    {
        vector<int> combo;
        for (int i = 0; i < n; ++i)
        {
            if (select[i])
            {
                combo.push_back(nums[i]);
            }
        }

        bool valid = true;
        for (size_t i = 1; i < combo.size(); ++i)
        {
            if (combo[i] - combo[i - 1] <= min_gap)
            {
                valid = false;
                break;
            }
        }

        if (valid)
        {
            count++;
            cout << count << endl;
        }

    } while (next_permutation(select.begin(), select.end()));

    return count;
}

int main()
{
    cout << countValidCombinations(50, 10) << endl;
    return 0;
}
