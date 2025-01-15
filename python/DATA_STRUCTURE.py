#####
## 数据结构
####
# 列表
# 模拟栈 O(1) 
stk = []
stk.append(1) # 向栈中添加元素
stk.pop() # 从栈中弹出元素
# 模拟队列 O(n)
que = []
que.append(1) # 向队列中添加元素
que.pop(0) # 从队列中弹出元素

# 双端队列 O(1)
from collections import deque
deq_arr = [ deque() for _ in range(10) ]
deq_arr[0].append(1) # 向右添加元素
deq_arr[0].appendleft(1) # 向左添加元素
deq_arr[0].popleft() # 向左弹出元素
deq_arr[0].pop() # 向右弹出元素

# 堆 (默认小顶推)
from heapq import *
heap = []
heappush(heap, 1) # 向堆中添加元素
heappop(heap) # 从堆中弹出元素
heapify(heap) # 将列表转换为堆（结构） O(n) 这个操作是原地的，不会返回新的堆对象
heapreplace(heap, 1) # 弹出最小的元素并将新元素推入堆中
heap[0] # 获取堆顶元素