class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res = {-1, -1}; 
        for (int i = 0; i < nums.size(); ++i) {
            for (int j = 0; j < nums.size(); ++j) {
                if (i != j) { /
                    int sum = nums[i] + nums[j];
                    if (sum == target) {
                        res[0] = i;
                        res[1] = j;
                        return res; 
                    }
                }
            }
        }
        return res;
    }
};