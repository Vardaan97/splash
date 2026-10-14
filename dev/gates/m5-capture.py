#!/usr/bin/env python3
"""Main-session queue only: ABBA full-head captures. No builds or timing."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if os.environ.get('M5_WINDOW')!='1': sys.exit('Requires main-session M5_WINDOW=1')
root,prompts,model,out=map(lambda s:Path(s).resolve(),sys.argv[1:5])
if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip() != (root/'build/pf-m4.sha').read_text().strip():
    sys.exit('Source SHA differs from M4 receipt')
if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
    sys.exit('Candidate source tree is dirty')
if (out/'identity.result').read_text().strip()!='PASS': sys.exit('M5 identity first')
# Verify the exact M4-built artifacts before the quality arms too.
for line in (root/'build/pf-artifacts.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    if hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()!=digest: sys.exit('Artifact SHA differs')
provenance=json.loads((prompts/'provenance.json').read_text())
manifest={'vocabulary':int(os.environ['VOCABULARY']),'contexts':[],
          'provenance':provenance,'candidate_sha':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()}
env={k:v for k,v in os.environ.items() if not k.startswith('SPLASH_') and k!='MTL_SHADER_VALIDATION'}
with open('/tmp/m5-gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for prompt in provenance['prompts']:
        path=prompts/prompt['prompt']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=prompt['prompt_sha256']: sys.exit('Prompt SHA differs')
        # Copy the small token file for a self-contained comparison manifest.
        saved=out/path.name
        saved.write_bytes(path.read_bytes())
        record={'tokens':prompt['tokens'],'prompt':saved.name,'prompt_sha256':prompt['prompt_sha256']}
        for arm in ('off0','on0','on1','off1'):
            subprocess.run([sys.executable,str(Path.home()/'Developer/splash-speed/scripts/m5-bench-split.py'),'guard','0'],check=True,env=env)
            arm_env=dict(env,SPLASH_PREFILL_QSA_CAPTURE=str(out/f'{prompt["tokens"]}-{arm}.f32'))
            if arm.startswith('on'): arm_env['SPLASH_PREFILL_QSA_COMPACT']='validate'
            command=[str(root/'build/engine-tests/model-runtime-oracle'),str(root/'build/splash.metallib'),str(model),
                     '--long-prompt-file',str(path),'--long-score-stride','64','--ple-ssd-cache','2147483648','--ane-ffn-share','0']
            log=out/f'{prompt["tokens"]}-{arm}.log'
            with log.open('w') as stream: subprocess.run(command,env=arm_env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            if f'PASS long-context-capture rows=256 context={prompt["tokens"]}' not in log.read_text(): sys.exit('Incomplete capture')
            if arm.startswith('on') and ('PF7 tolerance pass' not in log.read_text() or 'PF7 runtime compact dispatched' not in log.read_text()): sys.exit('Candidate fell back; quality result invalid')
            record[arm]=f'{prompt["tokens"]}-{arm}.f32'
        manifest['contexts'].append(record)
(out/'captures.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Captures complete; merge real oracle/greedy/needle/acceptance receipts into captures.json, then run pf7-quality.py. No promotion yet.')
