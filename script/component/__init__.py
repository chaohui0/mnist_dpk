import sys,os,platform

if platform.system() == "Windows":
    sys.path.append("\\".join(os.path.dirname(os.path.abspath(sys.argv[0])).split("\\")[0:len(os.path.dirname(os.path.abspath(sys.argv[0])).split("\\"))-1]))
else:
    sys.path.append("/".join(os.path.dirname(os.path.abspath(sys.argv[0])).split("/")[0:len(os.path.dirname(os.path.abspath(sys.argv[0])).split("/"))-1]))


