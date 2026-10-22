"""Durable, single-worker study pipeline. Completion still requires visual review."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from revision_common import ROOT, STUDY, RESULTS, freeze_check, health, save_json, utc_now

PYTHON=ROOT/".venv-stage1/bin/python"
LOGS=RESULTS/"logs"


def main():
    LOGS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/"pipeline.lock").open("a+") as own:
        try:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A revision pipeline is already running")
        started=utc_now()
        state={"pid":os.getpid(),"started":started,"state":"running","stage":"waiting_for_gpu_lock"}
        def status(stage,**extra):
            state.update(stage=stage,updated=utc_now(),**extra)
            save_json(RESULTS/"pipeline.json",state)
            print(json.dumps(state),flush=True)
        env=os.environ.copy()
        for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
            env[name]="2"
        env["PYTHONDONTWRITEBYTECODE"]="1"
        def run(stage,script,*args,fd=None):
            status(stage)
            with (LOGS/(stage+".log")).open("a") as output:
                output.write(f"\nPipeline invocation {utc_now()}\n");output.flush()
                subprocess.run([str(PYTHON),"-B","-u",str(STUDY/"code"/script),*args],cwd=ROOT,
                    env=env,stdout=output,stderr=subprocess.STDOUT,check=True,
                    pass_fds=() if fd is None else (fd,))
        try:
            freeze_check()
            health()
            status("waiting_for_gpu_lock")
            lock=ROOT/os.environ.get("GPU_LOCK","gpu.lock")  # release: GPU_LOCK names the lock file
            with lock.open("rb") as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX)
                os.set_inheritable(gpu.fileno(),True)
                info=subprocess.run(["nvidia-smi","--query-gpu=memory.free,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout.strip()
                if int(info.split(",")[0])<2048:
                    raise RuntimeError("Insufficient GPU memory after acquiring lock")
                state["gpu_at_admission"]=info
                env.update(AI06_GPU_LOCK_FD=str(gpu.fileno()),CUDA_VISIBLE_DEVICES="0",CUBLAS_WORKSPACE_CONFIG=":4096:8")
                run("development-resume","run_experiments.py","--phase","development","--device","cuda",fd=gpu.fileno())
                run("development-verification","verify_revision.py","--phase","development")
                run("development-analysis","analyse_revision.py","--source","development")
                run("confirmation-experiments","run_experiments.py","--phase","confirmation","--device","cuda",fd=gpu.fileno())
            env.pop("AI06_GPU_LOCK_FD",None)
            run("confirmation-verification","verify_revision.py","--phase","confirmation")
            run("confirmation-analysis","analyse_revision.py","--source","confirmation")
            run("all-model-analysis","analyse_revision.py","--source","all")
            run("supplement","supplement.py")
            run("paper-build","build_paper.py","--final")
            status("pdf-export")
            build=ROOT/"paper/build";build.mkdir(parents=True,exist_ok=True)
            for _ in range(2):
                with (build/"compile-console.log").open("w") as output:
                    subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error",
                        f"-output-directory={build}",str(ROOT/"paper/manuscript.tex")],cwd=ROOT,
                        stdout=output,stderr=subprocess.STDOUT,check=True)
            destination=ROOT/"output/pdf/AI-06-manuscript.pdf";destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(build/"manuscript.pdf",destination)
            shutil.copy2(destination,ROOT/"paper/manuscript.pdf")
            preview=ROOT/"tmp/pdfs/final";preview.mkdir(parents=True,exist_ok=True)
            subprocess.run(["pdftoppm","-scale-to","1600","-png",str(destination),str(preview/"page")],check=True)
            run("delivery-audit","check_delivery.py")
            status("ready_for_visual_review",state="ready_for_visual_review",completed=utc_now())
        except BaseException as exc:
            status(state.get("stage","unknown"),state="failed",error=repr(exc))
            traceback.print_exc()
            raise


if __name__=="__main__":
    main()
