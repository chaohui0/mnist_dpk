import os,sys
print(os.getcwd())
# 当前目录作为执行目录
os.chdir(os.path.join(os.getcwd(), '.'))
# script搜索路径
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), '../../'))

sys.path.append("/data/workspace")
from wuyuan import DPK

dpk,ip,port = sys.argv[1:]
DPK().install(path=dpk,
              ip=ip,
              port=port)
# DPK().uninstall(ip='10.11.8.145', port=8082, dpk_id=1)
