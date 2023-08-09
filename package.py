import sys
sys.path.append("/data/workspace")

from wuyuan import dpk_package

if len(sys.argv) < 4:
   raise ValueError(f"请输入dpk源目录、输出路径、dpk名字")

dpk_src = sys.argv[1]
dpk_dst = sys.argv[2]
dpk_name = sys.argv[3]
dpk = dpk_package(
   src_path=dpk_src,
   dst_path=dpk_dst,
   dpk_name=dpk_name)
