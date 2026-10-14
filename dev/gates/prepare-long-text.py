#!/usr/bin/env python3
"""M4, under build lock: pin genuine neutral text, with no repetition/downloads."""
import hashlib
import json
from pathlib import Path
import sys
from tokenizers import Tokenizer

text, tokenizer, output = map(Path,sys.argv[1:])
ids = Tokenizer.from_file(str(tokenizer)).encode(text.read_text(),add_special_tokens=False).ids
if len(ids)<204800:
    sys.exit('Need at least 204800 tokens of genuine source text; repetition/padding forbidden')
output.mkdir(parents=True,exist_ok=True)
provenance={'text':str(text),'text_sha256':hashlib.sha256(text.read_bytes()).hexdigest(),
            'tokenizer_sha256':hashlib.sha256(tokenizer.read_bytes()).hexdigest(),'prompts':[]}
for length in (32768,131072,204800):
    path=output/f'{length}.tokens'
    path.write_text(' '.join(map(str,ids[:length]))+'\n')
    provenance['prompts'].append({'tokens':length,'prompt':path.name,'prompt_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
