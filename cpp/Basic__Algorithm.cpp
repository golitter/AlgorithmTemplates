/**
 * 不定向输入       UndirectedInput
 * 二分         binary
 * 离散化       discretization
 * 搜索         Bfs_Dfs
 * 结构体      Adt
 * STL
*/
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream> // 需要包含这个头文件
using namespace std;


namespace golitter {
namespace UndirectedInput {
#include <sstream> // 需要包含这个头文件
void solve() {
    stringstream put_str;
    string str;
    getline(cin, str); // 获取一行字符串
    int n(0), p;
    put_str<<str; // 将str重定向输入到put_str
    while(put_str>>p) n++; // 从put_str重定向读入数据
    cout<<n;
}

}}

namespace golitter {
namespace binary {

bool check(int mid) {
    ;
}
void solve() {
    int l,r;
    int ans;
    while(l <= r) {
        int mid = l + r >> 1;
        if(check(mid)) {
            // ans = mid; // 最小值最大
            l = mid + 1;
        } else {
            // ans = mid; // 最大值最小
            r = mid - 1;
        }
    }
    {
        // 最大值最小
        while(l < r) {
            int mid = (l + r) >> 1;
            if(check(mid)) r = mid;
            else l = mid + 1;
        } // output: r
    }
    {
        // 最小值最大
        while(l < r ) {
            int mid = (l + r + 1) >> 1; // [2, 2]; --> 
            if(check(mid)) l = mid;
            else r = mid - 1;
        } // output: l
    }

    // cpp的lower_bound和upper_bound
    // lower_bound: 返回第一个大于等于给定值的迭代器
    // upper_bound: 返回第一个大于给定值的迭代器
    // it != a.end() 表示找到了符合条件的元素
    // it != a.begin() 表示it这个元素不是第一个元素
    // index: distance(a.begin(), it) 返回it在a中的索引
}

}}

namespace golitter {
namespace discretization {

const int N = 333;
int a[N],last[N],id[N];
void test1() { // 重复数字一样 1 222 222     -----> 1 2 2 url: https://www.luogu.com.cn/record/115366968
    int n; cin>>n; for(int i = 1; i <= n; ++i) cin>>a[i],id[i] = a[i];
    sort(id+1, id+1+n);
    int cnt = unique(id+1, id+n+1) - id - 1;
    for(int i = 1; i <= n; ++i) {
        last[i] = lower_bound(id+1, id+cnt+1, a[i]) - id;
    }

        // STL 实现
    {
        vector<int> a({1,2,3,555,4,33,33,10}); // 测试数据 
        vector<int> id(a), last(a); // id 用于存储 a 中的唯一元素，last 用于存储每个元素在 id 中的索引 == 声明并初始化
        sort(id.begin(), id.end()); // 对 id 进行排序
        id.erase(unique(id.begin(), id.end()), id.end()); // 删除 id 中的重复元素
        for(int i = 0; i < a.size(); ++i) { // 遍历 a 中的每个元素
            // 使用 lower_bound 查找 a[i] 在 id 中的位置
            last[i] = lower_bound(id.begin(), id.end(), a[i]) - id.begin();
        }
        for(auto &t: last) cout<<t<<' ';
    }

}

void solve() {
        ;
    }

}}

namespace golitter {

// 常用二进制操作
namespace bitwise {
    string to_bitstr(int n) { string s = ""; while(n) { s += (n % 2) + '0'; n >>= 1; } reverse(s.begin(), s.end()); return s; } // 字符串形式的二进制表示
    int bitsz(int n) { return to_bitstr(n).size(); } // 转为二进制的长度
    int ones(int n) { int cnt = 0; while(n) { n &= (n - 1); cnt++; } return cnt; } // 1的个数
    int zeros(int n) { return bitsz(n) - ones(n); } // 从左往右第一个非0开始计算，0的个数
    bool ispower2(int n) { return (n & (n - 1)) == 0; } // 判断是否是2的次幂
    int lowbit(int n) { return n & -n; } // 二进制最低位1与其后面的0组成的十进制数值
}
}

namespace golitter {

namespace Adt_Struct {

struct Adt {
    int a,b;
    bool operator<(const Adt& rhs) const {
        return a > rhs.a;
    }
};

void solve() {
    // cpp 11 之前
    vector<Adt> v(3,{Adt{0,0}});
    v.push_back(Adt{1,2});
    v[3] = Adt{3,4};

    // cpp 11 之后
    vector<Adt> v(3,{0,0});
    v.push_back({1,2});
    v[3] = Adt{3,4};
} 

}
}

#include <vector>
#include <list>
#include <stack>
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
namespace golitter {
namespace STL {
void Vector() {
    /**
     * vector<int> vi || vi(n)
            初始化  vector<int> v(n) - 此时默认n个元素且值为0   v(n,v) - 此时默认n个元素且值为v
     * size()     返回元素个数
     * clear()    清空
     * front() back()   第一个，最后一个元素
     * []

     vector模拟stack
        * push_back()
        * pop_back()
        * back()
    */
}
void List() {
    /**
     * list<int> lst; 双向链表
     * 插入/删除不使其他迭代器失效，定位后插删 O(1)
     * 不支持 []，不能用 std::sort（用成员函数 lst.sort()）
     * push_front() push_back() pop_front() pop_back() front() back() size() clear()
     * insert(it, v)   在 it 前插入
     * erase(it)       ** it = lst.erase(it) 接住返回值 **
     * remove(v) remove_if(cond)   删除元素
     * splice(pos, lst, it)        O(1) 剪切节点
    */
}
void String() {
    /**
     * string str;
     * substr(pos, len);
     * size()
     * reverse(bg, ed);
    */
}
void Queue() {
    /**
     * queue<int> q;
     * size()
     * clear()
     * push()
     * front()
     * pop()
    */

    /**
     * deque<int> dq; // https://blog.csdn.net/mataojie/article/details/122310752?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168994535816800192227446%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168994535816800192227446&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-122310752-null-null.142^v90^insert_down1,239^v3^control&utm_term=deque&spm=1018.2226.3001.4187
     * size()
     * empty()
     * front() back()
     * push_front() push_back()
     * pop_front() pop_back()
     * 可以数组下标访问
     * 可以排序
    */

   // 单调队列
// 常见模型：找出滑动窗口中的最大值/最小值
// int hh = 0, tt = -1;
// for (int i = 0; i < n; i ++ )
// {
//     while (hh <= tt && check_out(q[hh])) hh ++ ;  // 判断队头是否滑出窗口
//     while (hh <= tt && check(q[tt], i)) tt -- ;
//     q[ ++ tt] = i;
// }
}
void Priority_queue() {
    /**
     * priority_queue<int> heap 大顶堆 [大的在上面] 默认大顶堆
     *      等价于 priority_queue<int,vector<int>,less<int>> heap 大顶堆
     * priority_queue<int, vector<int>, greater<int>> q; 小顶堆 [小的在上面]
     * size()    push()     pop()     top()
    */
}
// 用priority_queue 自定义堆 http://www.cbww.cn/news/37826.shtml
//      要重载 < 操作符 ，注意两个const才可以通过编译
// 方法一 重载运算符<
struct adt { // 小顶堆
    int a;
    bool operator<(const adt& rhs) const { // 优先队列的><与sort的><相反. ** 没有const会报错
        return a > rhs.a; // 这里 从大到小进行排序，队列从最右边开始，所以是小顶堆
    }
};
// 方法二 使用lambda表达式
void test_priority_queue() {
    auto cmp = [](int pre, int suf) { return pre > suf; }; // 小顶堆
    priority_queue<int,vector<int>, decltype(cmp)> pq(cmp); // decltype 类型说明符

    // 实现自定义PII堆结构
    auto pii_cmp = [](PII pre, PII suf) {return pre.vf < suf.vf; };
    priority_queue<PII, vector<PII>, decltype(pii_cmp)> heap(pii_cmp);

}
void Stack() {
    /**
     * stack<int> s;
     * size()
     * clear()
     * push()
     * pop()
     * top()
    */
   // 单调栈

// 常见模型：找出每个数左边离它最近的比它大/小的数
// auto linear_stack = [&]() {
//     int tt = 0;
//     for (int i = 1; i <= n; i ++ )
//     {
//         while (tt && check(stk[tt], i)) tt -- ;
//         stk[ ++ tt] = i;
//     }    
// }
}
void Map() {
    /**
     * 
     *  map 自带大常数，但是卡不掉map，
     * stl 里 套 stl会很慢 ***
     * size()   clear()
     *     
     * map:
     * 可以元组映射
     * map<PII,int> mpi; mpi[{1, 2}] = 3;
     * 
     * unordered_map 不可以元组映射
     * multimap:
     *  multimap<PII,PII> mmpp;
     * mmpp.insert(pair<PII,PII>({x1,y1}, {x2,y2}));
     * count()    find()
     * 
     * ** multimap不支持 [] 操作。** *
     * 
     * map 和 unordered_map 比较：
     *              unordered_map最坏O(n)，会被卡
     *              # cf 有专门卡umap的
    */
   struct custom_hash { // 防止卡umap
	static uint64_t splitmix64(uint64_t x) {
		x += 0x9e3779b97f4a7c15;
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
		x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
		return x ^ (x >> 31);
	}
	
	size_t operator()(uint64_t x) const {
		static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count(); // <chrono>
		return splitmix64(x + FIXED_RANDOM);
	}
};
    unordered_map<int,int,custom_hash> umii; 

}
void Set() {
    /**
     * set<int> s;
     * insert()   erase()
     * count()
    */
}
void Unordered_All() {
    /**
     * 
    */
}
}}

namespace golitter {
namespace Bfs_Dfs {
// void dfs(int k)
// 　{
// 　  if  (到目的地) 输出解;
// 　　　else
// 　　　　for (i=1;i<=算符种数;i++)
// 　　　　　if  (满足条件) 
// 　　　　　　　{
// 　　　　　　　　保存结果;
// 　　　                  Search(k+1)
//                              恢复：保存结果之前的状态{回溯一步}
// 　　　　　　　}
// 　}

// void bfs() {
//     // queue
// }


int T() // 全排列 组合 模板
{
    // 全排列
    // int n; cin>>n;
    // vector<int> p(n + 1), vis(n + 1);
    // auto dfs = [&](auto &&self, int stp) -> void {
    //     if(stp == n + 1) {
    //         for(int i = 1; i <= n; ++i) {
    //             printf("%5d",p[i]);
    //         }
    //         cout<<endl;
    //         return ;
    //     }
    //     for(int i = 1; i <= n; ++i) {
    //         if(vis[i]) continue;
    //         vis[i] = 1;
    //         p[stp] = i;
    //         self(self, stp + 1);
    //         vis[i] = 0;
    //     }
    // };
    // dfs(dfs, 1);

    // 组合
    int n,m; cin>>n>>m;
    vector<int> vis(n + 1);
    auto dfs = [&](auto &&self, int stp, int st) -> void {
        if(stp == m + 1) {
            for(int i = 1; i <= n; ++i) {
                if(!vis[i]) continue;
                printf("%3d",i);
            }
            cout<<endl;
            return ;
        }
        for(int i = st; i <= n; ++i) {
            if(vis[i]) continue;
            vis[i] = 1;
            self(self, stp + 1, i + 1);
            vis[i] = 0;
        }
    };
    dfs(dfs,1,1);
}
}}