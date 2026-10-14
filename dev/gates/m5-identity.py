#!/usr/bin/env python3
"""Main-session M5 identity phase. No builds, transfers or timing."""
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys

if os.environ.get('M5_WINDOW') != '1':
    sys.exit('Requires M5_WINDOW=1 from the main-session queue')
root = Path(sys.argv[1]).resolve()
if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip() != (root/'build/pf-m4.sha').read_text().strip():
    sys.exit('Source SHA differs from M4 receipt')
if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
    sys.exit('Candidate source tree is dirty')
package = sys.argv[2]
out = Path(os.environ['OUT']).resolve()
out.mkdir(parents=True, exist_ok=True)
manifest = root / 'build/pf-artifacts.sha256'
if not manifest.exists() or (root / 'build/pf-m4.result').read_text().strip() != 'PASS':
    sys.exit('Missing successful M4 receipt; identity cannot start')
for line in manifest.read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    if hashlib.sha256((root / name.strip()).read_bytes()).hexdigest() != digest:
        sys.exit('M4 artifact hash mismatch: ' + name)
env = {k:v for k,v in os.environ.items() if not k.startswith('SPLASH_') and k != 'MTL_SHADER_VALIDATION'}
if package == 'PF7': env['SPLASH_PREFILL_QSA_COMPACT'] = 'validate'
with open('/tmp/m5-gpu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard = Path.home()/'Developer/splash-speed/scripts/m5-bench-split.py'
    subprocess.run([sys.executable,str(guard),'guard','0'],check=True,env=env)
    shared = Path.home()/'Developer/kernels/kernel-gate.sh'
    if shared.exists():
        gate_env = dict(env,KERNEL_GATE_PACKAGE=package.lower(),KERNEL_GATE_PHASE='identity')
        # The universal gate owns its lock; release ours while it runs.
        fcntl.flock(lock, fcntl.LOCK_UN)
        subprocess.run(['bash',str(shared),str(root)],check=True,env=gate_env)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for validation in (True,False):
        arm_env = dict(env,SPLASH_KERNEL_SELFTEST='1')
        if validation: arm_env['MTL_SHADER_VALIDATION']='1'
        for name,command in (
            ('u2',[str(root/'build/splash'),'--kernel-selftest',str(root/'build/splash.metallib')]),
            ('u3',[str(root/('build/engine-tests/prefill-qsa-compact' if package == 'PF7' else 'build/engine-tests/prefill-attention-variants')),str(root/'build/splash.metallib'),'--full']),
            ('u5',[str(root/'build/engine-tests/mtp-runtime'),str(root/'build/splash.metallib')])):
            with (out/f'{name}-sv{int(validation)}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=arm_env)
            if 'UNAVAILABLE' in (out/f'{name}-sv{int(validation)}.log').read_text():
                sys.exit('M5 candidate unavailable; no timing allowed')
    for mode in (['qk2','prob2','rows4'] if package == 'PF3' else ['validate']):
        arm_env = dict(env)
        arm_env['SPLASH_ATTN_PIPE' if package == 'PF3' else 'SPLASH_PREFILL_QSA_COMPACT'] = mode
        with (out/f'u5-{mode}.log').open('w') as log:
            subprocess.run([str(root/'build/engine-tests/mtp-runtime'),str(root/'build/splash.metallib')],env=arm_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'identity.result').write_text('PASS\n')
print('PASS identity only; U7/quality and remaining universal gates still required before timing')
