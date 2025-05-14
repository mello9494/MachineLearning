def equally_balanced(nums, subset, i):
    sub_sum = sum([nums[i] for i in subset])
    half = sum(nums) / 2

    if sub_sum == half:
        return subset
    elif i >= len(nums) or sub_sum > half:
        return None
        
    temp1 = equally_balanced(nums, subset + [i], i+1)
    if temp1: return temp1
    
    temp2 = equally_balanced(nums, subset, i+1)
    if temp2: return temp2

    
nums = [[3, 5, 1, 1, 8], [5, 2, 2, 9, 4, 9, 3]]
for i in nums:
    indexes = equally_balanced(i, [], 0)
    print([i[j] for j in indexes], [i[j] for j in range(len(i)) if j not in indexes])