"""Run every experiment and store its console output next to its figures.
Usage:  python run_all.py            (all experiments)
        python run_all.py 5 11 19    (only these experiment numbers)
Each experiment writes figures to ../outputs/<script-name>/ and the console log to ../outputs/<script-name>/output.txt"""
import glob, os, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__)); out_root = os.path.join(os.path.dirname(here), "outputs")
scripts = sorted(glob.glob(os.path.join(here, "exp*.py")))
if len(sys.argv) > 1:
    wanted = {int(a) for a in sys.argv[1:]}; scripts = [s for s in scripts if int(os.path.basename(s)[3:5]) in wanted]
summary = []
for s in scripts:
    name = os.path.splitext(os.path.basename(s))[0]; od = os.path.join(out_root, name); os.makedirs(od, exist_ok=True)
    t0 = time.time(); p = subprocess.run([sys.executable, s], cwd=here, capture_output=True, text=True); dt = time.time() - t0
    log = p.stdout + (("\n--- stderr ---\n" + p.stderr) if p.returncode else "")
    if p.returncode and "TensorFlow/Keras is required" in p.stdout + p.stderr:
        log += "\n[NOT EXECUTED] This experiment needs TensorFlow/Keras, which was not available when this package was built.\nRun it on your machine/Colab: python codes/%s.py\n" % name
    with open(os.path.join(od, "output.txt"), "w", encoding="utf-8") as f: f.write(log)
    status = "OK" if p.returncode == 0 else "NEEDS TF" if "TensorFlow/Keras is required" in p.stdout + p.stderr else "FAILED"
    summary.append((name, status, dt)); print(f"{name:45s} {status:9s} {dt:6.1f}s")
with open(os.path.join(out_root, "run_summary.txt"), "w") as f:
    f.write("\n".join(f"{n:45s} {st:9s} {d:6.1f}s" for n, st, d in summary) + "\n")
