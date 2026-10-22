"""Run one project worker; record resource samples and stop only its process group."""
import argparse,json,os,signal,subprocess,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seconds',type=int,default=3600);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
P=Path(__file__).resolve().parents[1];H=P/'06-Study-History'
def available():return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])
if available()<3*1024**2 or shutil.disk_usage(P).free<10*1024**3:raise SystemExit('Resource admission failed')
start=time.monotonic();maxrss=0;reason=None
with (H/f'{a.name}.log').open('w') as log,(H/f'{a.name}-resources.jsonl').open('w') as rec:
 proc=subprocess.Popen(a.command,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while proc.poll() is None:
  try:
   s=dict(x.split(':',1) for x in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if ':' in x);rss=int(s.get('VmRSS','0').split()[0]);peak=int(s.get('VmHWM','0').split()[0]);maxrss=max(maxrss,peak);ram=available();elapsed=time.monotonic()-start
   reason='rss_limit' if rss>4*1024**2 else 'ram_pressure' if ram<2*1024**2 else 'timeout' if elapsed>a.seconds else None
   rec.write(json.dumps({'elapsed_s':elapsed,'rss_kib':rss,'peak_rss_kib':peak,'available_kib':ram,'stop_reason':reason})+'\n');rec.flush()
   if reason:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
    break
  except FileNotFoundError:pass
  time.sleep(.5)
 code=proc.wait()
r={'name':a.name,'exit_code':code,'stop_reason':reason,'elapsed_s':time.monotonic()-start,'peak_rss_kib':maxrss,'command':a.command,'monitored_from_launch':True};(H/f'{a.name}-summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(0 if code==0 and reason is None else 1)
