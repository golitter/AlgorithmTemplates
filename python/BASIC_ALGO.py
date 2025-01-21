####
# 基础算法模板
####

## 二分
def binary_search():
    nums = [1,2,3,4,5,6,7,8,9], target = 5

    def template1():
        # 最大值最小
        l,r = 0, len(nums) - 1
        while l < r:
            mid = l + r >> 1
            def check(mid:int) -> bool:
                return nums[mid] >= target
            if check(mid): r = mid
            else: l = mid + 1
        res1 = r 

        # 最小值最大
        l,r = 0, len(nums) - 1
        while l < r:
            mid = l + r + 1 >> 1
            def check(mid: int) -> bool:
                return nums[mid] <= target
            if check(mid): l = mid
            else: r = mid - 1
        res2 = l 
    
    def tempalte2():
        from bisect import bisect_left, bisect_right
        # bisect_left 相当于 lower_bound
        # bisect_right 相当于 upper_bound
        a = [1, 2, 4, 4, 8]
        x = 4
        print(bisect_left(a, x)) # 2
        print(bisect_right(a, x)) # 4

## 排列组合
def permute(nums: list[int]) -> list[list[int]]:
    res = []
    n = len(nums)
    vis = [0] * n
    p = [0] * n
    def dfs(stp:int) -> None:
        if stp == n:
            res.append(p[:])
            return
        for i in range(n):
            if vis[i] == 0:
                vis[i] = 1
                p[stp] = nums[i]
                dfs(stp+1)
                vis[i] = 0
    dfs(0)
    return res
def combine(n: int, k: int) -> list[list[int]]:
    vis = [0] * (n + 1)
    res = []
    def dfs(stp:int, st:int) -> None:
        if stp == k:
            res.append([i for i in range(1, n + 1) if vis[i]])
            return
        for i in range(st, n + 1):
            if vis[i]:
                continue
            vis[i] = 1
            dfs(stp + 1, i + 1)
            vis[i] = 0
    dfs(0, 1)
    return res

## 递归
# 记忆化搜索
from functools import lru_cache, cache