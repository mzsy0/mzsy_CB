"""
《邢不行-Python股票量化投资课程》
author: 邢不行
微信:xbx8662

（1）xbx-py11库也属于第三方库，同样可以在https://pypi.org/search/搜索到

（2）里面集成了数字货币/股票学习过程中所需要的所有库，直接安装xbx-py11更加方便

（3）前面介绍的安装方式依然适用，命令为：pip install xbx-py11 -i https://pypi.tuna.tsinghua.edu.cn/simple

（4）请注意，如果要安装xbx-py11，你的Python版本最好是3.11.10

（5）这个代码除了安装xbx-py11之外，也可以用来安装其他库

"""

import subprocess
import sys


def install_package():
    """
    根据用户输入安装指定的包
    """
    package = input("请输入要安装的包名称（如需带版本号用'=='指定，如：pandas==2.2.3）：")
    exe_path = sys.executable

    # 使用清华源安装需要的包
    if '==' in package:
        pip_command = f"{exe_path} -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple {package}"
    else:
        pip_command = f"{exe_path} -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade {package}"

    try:
        subprocess.check_call(pip_command.split())
    except subprocess.CalledProcessError as e:
        print(f"清华源安装{package}失败：{e}\n尝试从阿里源安装")
        try:
            # 使用阿里源安装需要的包
            if '==' in package:
                pip_command = f"{exe_path} -m pip install -i https://mirrors.aliyun.com/pypi/simple {package}"
            else:
                pip_command = f"{exe_path} -m pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade {package}"
            subprocess.check_call(pip_command.split())
        except subprocess.CalledProcessError as e:
            print(f"阿里源安装{package}失败：{e}")
        else:
            print(f"{package}已成功安装")
    else:
        print(f"{package}已成功安装")


if __name__ == "__main__":
    # 针对不是Python3.11的版本，进行提示
    if not str(sys.version).startswith('3.11'):
        # 重要的事情说三遍
        print("请注意，当前Python版本不是3.11，可能会导致安装失败，如果安装过程中有异常，请尝试切换到Python3.11再安装！\n" * 3)
    install_package()
