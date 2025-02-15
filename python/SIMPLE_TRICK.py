### 格式化 （f-string）3.6+
"""
f"字符串内容{表达式:格式说明}"
f"{ans:.2f}" # 保留两位小数
f"{ans:.0f}" # 保留整数
f"{ans:.2e}" # 科学计数法
f"{ans:0>2}" # 右对齐，左边补0
f"{a:0>10.2f}" 右对齐，左边补0，保留两位小数  0000003.00
"""
### 浅拷贝、深拷贝
# 浅拷贝
a = [1,2,3]
b = a.copy()
b = a[:]
# 深拷贝
import copy
b = copy.deepcopy(a)

### 开数组
dp = [[0] * 10 for _ in range(10)]
dp = [[0] * 10] * 10 # 这样开数组会导致每一行都是同一个引用，修改一行会导致所有行都被修改

### 输入输出
# 读取一行数字
a = list(map(int, input().split()))
# 读取多行数字
a = [list(map(int, input().split())) for _ in range(n)]
# 输出一行数字
print(' '.join(map(str, a)))

### 字母/数字 转换
alp = [ chr(x + ord('a')) for x in range(26) ]
alp_num = list(map(ord, alp))

### 排序
a = []
a.sort() # 默认升序 从小到大 原地排序
a.sort(key=lambda x: (x[0], -x[1])) # 先按第一个元素升序，再按第二个元素降序 原地排序
a = sorted(a, key=lambda x: (x[0], -x[1])) # 先按第一个元素升序，再按第二个元素降序 返回排序后的新列表
# 自定义复杂排序规则
class selfHeap:
    __slots__ = ['a', 'b']
    def __init__(self, a, b):
        self.a = a; self.b = b
    def __lt__(self, rhs:'selfHeap') -> bool:
        """
        True: self < rhs
        False: self >= rhs
        """
        if self.a == rhs.a: return self.b < rhs.b
        return self.a < rhs.a
    def __str__(self) -> str:
        return str(self.a) + ' ' + str(self.b)
### 翻转
a.reverse() # 原地翻转
a = a[::-1] # 返回新列表
a = list(reversed(a)) # 返回新列表