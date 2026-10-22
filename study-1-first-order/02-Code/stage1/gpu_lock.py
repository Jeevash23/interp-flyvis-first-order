"""Exec worker while holding the existing portfolio GPU advisory lock."""
import os,sys,fcntl,subprocess,json
from pathlib import Path
P=Path(__file__).resolve().parents[2]
lock=P/os.environ.get('GPU_LOCK','gpu.lock')  # release: GPU_LOCK names the lock file
f=open(lock,'rb');fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(f.fileno(),True)
s=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
free=int(s.stdout.split(',')[0]);
if free<2048:raise SystemExit('Insufficient GPU availability')
print('GPU lock acquired; free/used MiB and temperature:',s.stdout.strip(),flush=True)
os.environ['AI06_GPU_LOCK_FD']=str(f.fileno());os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.execv(sys.argv[1],sys.argv[1:])
