import numpy as np

def min_non_repeating(nums):
    min_val = nums[0]
    vals = {}

    for i in range(len(nums)):
        if nums[i] in vals:
            vals[nums[i]] += 1
        else:
            vals[nums[i]] = 1
        
        min_val = max(nums[i], min_val)

    for i in vals:
        if vals[i] == 1 and i < min_val:
            min_val = i

    return min_val

# generate random lists
nums = [[2, 2, 3, 4, 5]]
# for i in range(10):
#     nums.append(list(np.random.randint(0, 100, size=50)))

for i in nums:
    print(f'Min non-repeating val: {min_non_repeating(i)}\n')

# time complexity: O(n)