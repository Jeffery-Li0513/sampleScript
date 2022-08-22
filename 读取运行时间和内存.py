import time
from functools import wraps

class Solution(object):
    # 定义一个装饰器来测量函数执行时间
    def fn_timer(function):
        @wraps(function)
        def function_timer(*args, **kwargs):
            t0 = time.time()
            result = function(*args, **kwargs)
            t1 = time.time()
            print("消耗时间为：%s" % str(t1 - t0))
            return result
        return function_timer

    @fn_timer
    def removeDuplicates(self, nums):
        """
                :type nums: List[int]
                :rtype: int
                """
        i = 0
        while True:
            if i + 1 < len(nums):
                print(len(nums))
                print(i)
                if nums[i] == nums[i + 1]:
                    nums.pop(i + 1)
                    continue
                else:
                    i += 1
            else:
                break
        print(str(len(nums)) + ', nums=' + str(nums))


if __name__ == '__main__':
    test = [1,1,2,3,4,4]
    solve = Solution()
    solve.removeDuplicates(test)