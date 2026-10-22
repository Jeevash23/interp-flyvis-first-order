"""Bounded host-visible guardian of one explicitly identified AI-06 worker."""
import json,os,signal,sys,time
from pathlib import Path
pid=int(sys.argv[1]);p=Path('/proc')/str(pid);out=Path(sys.argv[2]);t=time.monotonic()
if b'02-Code/qualify.py' not in (p/'cmdline').read_bytes():raise SystemExit('Wrong worker')
start=(p/'stat').read_text().split()[21]
with out.open('w') as f:
 while p.exists():
  try:
   if (p/'stat').read_text().split()[21]!=start:break
   s=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
   if s['State'].strip().startswith('Z'):break
   rss=int(s.get('VmRSS','0 kB').split()[0]);peak=int(s.get('VmHWM','0 kB').split()[0]);mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());avail=int(mem['MemAvailable'].split()[0]);elapsed=time.monotonic()-t
   r={'utc_unix':time.time(),'elapsed_monitor_s':elapsed,'pid':pid,'rss_kib':rss,'peak_rss_kib':peak,'available_kib':avail}
   reason='rss_limit' if rss>4*1024**2 else 'ram_pressure' if avail<2*1024**2 else 'timeout' if elapsed>3600 else None
   if reason:r['stop_reason']=reason
   f.write(json.dumps(r)+'\n');f.flush()
   if reason:os.kill(pid,signal.SIGTERM);break
   time.sleep(2)
  except FileNotFoundError:break
print('Monitor finished')
