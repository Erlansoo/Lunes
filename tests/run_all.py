"""Every test that needs no hardware, one after the other: python tests/run_all.py"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = ("test_geometry.py", "test_scope_auto.py", "test_simulated_bench.py")

failed = []
for name in TESTS:
    t0 = time.time()
    print(f"\n##### {name}", flush=True)
    if subprocess.run([sys.executable, str(HERE / name)]).returncode != 0:
        failed.append(name)
    print(f"##### {name}: {time.time() - t0:.0f} s", flush=True)
print(f"\n{'all passed' if not failed else 'FAILED: ' + ', '.join(failed)}")
sys.exit(1 if failed else 0)
