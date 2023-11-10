# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 19:27
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : __init__.py
# @Software: PyCharm

# class _Loss(Module):
#     reduction: str

#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(_Loss, self).__init__()
#         if size_average is not None or reduce is not None:
#             self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
#         else:
#             self.reduction = reduction