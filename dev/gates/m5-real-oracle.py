#!/usr/bin/env python3
"""Main-session-only U7 ABBA; PF3 exact, PF7 numerical class N."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path);p.add_argument('model',type=Path);p.add_argument('out',type=Path)
p.add_argument('--package',choices=('PF3','PF7'),required=True)
p.add_argument('--mode',default='prob2',choices=('prob2','qk2','rows4'))
a=p.parse_args()
if os.environ.get('M5_WINDOW')!='1': sys.exit('Requires main-session M5_WINDOW=1')
a.root=a.root.resolve();a.model=a.model.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
if subprocess.check_output(['git','-C',str(a.root),'rev-parse','HEAD'],text=True).strip() != (a.root/'build/pf-m4.sha').read_text().strip():
    sys.exit('Source SHA differs from M4 receipt')
if subprocess.check_output(['git','-C',str(a.root),'status','--porcelain'],text=True).strip():
    sys.exit('Candidate source tree is dirty')
if (a.out/'identity.result').read_text().strip()!='PASS': sys.exit('Identity phase first')
for line in (a.root/'build/pf-artifacts.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    if hashlib.sha256((a.root/name.strip()).read_bytes()).hexdigest()!=digest: sys.exit('Artifact SHA differs')
env={k:v for k,v in os.environ.items() if not k.startswith('SPLASH_') and k!='MTL_SHADER_VALIDATION'}
reports=[]
with open('/tmp/m5-gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for kv,state,cache in (('int8','fp32',0),('int8','fp32',2147483648),('bf16','bf16',0)):
        tokens={};cosines=[]
        for arm in ('off0','on0','on1','off1'):
            subprocess.run([sys.executable,str(Path.home()/'Developer/splash-speed/scripts/m5-bench-split.py'),'guard','0'],check=True,env=env)
            arm_env=dict(env)
            if arm.startswith('on'):
                arm_env['SPLASH_ATTN_PIPE' if a.package=='PF3' else 'SPLASH_PREFILL_QSA_COMPACT']=a.mode if a.package=='PF3' else 'validate'
            command=[str(a.root/'build/engine-tests/model-runtime-oracle'),str(a.root/'build/splash.metallib'),str(a.model),
                     '--kv-format',kv,'--state-format',state,'--ane-ffn-share','0']
            if cache:command+=['--ple-ssd-cache',str(cache)]
            log=a.out/f'oracle-{a.mode if a.package=="PF3" else "compact"}-{kv}-{state}-{cache}-{arm}.log'
            with log.open('w') as stream: subprocess.run(command,env=arm_env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            text=log.read_text()
            if 'model_runtime_oracle_test: PASS' not in text: sys.exit('Missing oracle PASS')
            if arm.startswith('on'):
                marker=(a.mode+' identical') if a.package=='PF3' else 'PF7 tolerance pass'
                if marker not in text: sys.exit('Candidate self-test did not pass; no oracle promotion')
            tokens[arm]=[s for s in text.splitlines() if s.startswith('TOKENS')]
            if not tokens[arm]: sys.exit('Missing oracle token vectors')
            cosines.extend(float(x) for x in re.findall(r'(?:conv|recurrent|logits)_cos=([0-9.eE+-]+)',text))
        if tokens['off0']!=tokens['off1'] or tokens['on0']!=tokens['on1']: sys.exit('Oracle repeat changed tokens')
        exact=all(v==tokens['off0'] for v in tokens.values())
        if a.package=='PF3' and not exact: sys.exit('PF3 oracle token identity failed')
        if not cosines: sys.exit('Missing oracle partition cosine report')
        reports.append({'kv':kv,'state':state,'ple_bytes':cache,'off_pass':True,'on_pass':True,
                        'min_state_cosine':min(cosines),'tokens_exact':exact,'tolerance_class':'B' if a.package=='PF3' else 'N-partition'})
(a.out/f'{a.package.lower()}-oracle.json').write_text(json.dumps(reports,indent=2)+'\n')
print('PASS U7; token drift is diagnostic for PF7, whose full long-context quality gate remains mandatory')
