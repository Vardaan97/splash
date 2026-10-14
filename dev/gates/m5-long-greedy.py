#!/usr/bin/env python3
"""Main-session ABBA on pinned long prompts/needles; no server or timing.

Input JSON: tokenizer (tokenizer.json path), cases [{kind: greedy|needle,
context,prompt,prompt_sha256,depth?,seed?,expected?}]. Produces quality manifest
`greedy` and `needles` fields. Baseline and candidate repeat controls are exact.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tokenizers import Tokenizer
if os.environ.get('M5_WINDOW')!='1': sys.exit('Requires main-session M5_WINDOW=1')
root,model,suite,out=map(lambda s:Path(s).resolve(),sys.argv[1:5])
if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip() != (root/'build/pf-m4.sha').read_text().strip():
    sys.exit('Source SHA differs from M4 receipt')
if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
    sys.exit('Candidate source tree is dirty')
if (out/'identity.result').read_text().strip()!='PASS': sys.exit('Identity first')
for line in (root/'build/pf-artifacts.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    if hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()!=digest:sys.exit('Artifact SHA differs')
data=json.loads(suite.read_text());tokenizer=Tokenizer.from_file(str(suite.parent/data['tokenizer']))
result={'greedy':[],'needles':[],'suite_sha256':hashlib.sha256(suite.read_bytes()).hexdigest()}
env={k:v for k,v in os.environ.items() if not k.startswith('SPLASH_') and k!='MTL_SHADER_VALIDATION'}
with open('/tmp/m5-gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for i,case in enumerate(data['cases']):
        prompt=suite.parent/case['prompt']
        if hashlib.sha256(prompt.read_bytes()).hexdigest()!=case['prompt_sha256']:sys.exit('Prompt SHA differs')
        if len(prompt.read_text().split())!=case['context']:sys.exit('Prompt context differs')
        arms={}
        for arm in ('off0','on0','on1','off1'):
            subprocess.run([sys.executable,str(Path.home()/'Developer/splash-speed/scripts/m5-bench-split.py'),'guard','0'],check=True,env=env)
            arm_env=dict(env)
            if arm.startswith('on'):arm_env['SPLASH_PREFILL_QSA_COMPACT']='validate'
            command=[str(root/'build/engine-tests/model-runtime-oracle'),str(root/'build/splash.metallib'),str(model),
                     '--long-prompt-file',str(prompt),'--long-greedy','128','--ple-ssd-cache','2147483648','--ane-ffn-share','0']
            log=out/f'long-greedy-{i}-{arm}.log'
            with log.open('w') as stream:subprocess.run(command,env=arm_env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            lines=[line for line in log.read_text().splitlines() if line.startswith('LONG_GREEDY ')]
            if len(lines)!=1:sys.exit('Missing greedy token vector')
            if arm.startswith('on') and ('PF7 tolerance pass' not in log.read_text() or 'PF7 runtime compact dispatched' not in log.read_text()):sys.exit('Candidate fell back')
            arms[arm]=list(map(int,lines[0].split()[1:]))
        if arms['off0']!=arms['off1'] or arms['on0']!=arms['on1']:sys.exit('Greedy repeat changed')
        if case['kind']=='greedy':
            result['greedy'].append({'context':case['context'],'off':arms['off0'],'on':arms['on0'],
                                     'prompt_sha256':case['prompt_sha256']})
        elif case['kind']=='needle':
            result['needles'].append({'context':case['context'],'depth':case['depth'],'seed':case['seed'],
                                      'expected':case['expected'],
                                      'off':tokenizer.decode(arms['off0'],skip_special_tokens=True).strip(),
                                      'on':tokenizer.decode(arms['on0'],skip_special_tokens=True).strip()})
        else:sys.exit('Unknown case kind')
(out/'long-greedy.json').write_text(json.dumps(result,indent=2)+'\n')
print('Collected greedy/needle arms; pf7-quality.py decides the thresholds')
