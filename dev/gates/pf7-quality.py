#!/usr/bin/env python3
"""Compare full-vocabulary Splash captures; fail closed on missing coverage.

Capture records: little-endian uint32 vocabulary, vocabulary float32 logits.
Manifest: contexts [{tokens, prompt_sha256, off0, on0, on1, off1}], plus
oracle [{kv,state,off_pass,on_pass,min_state_cosine}], greedy [{context,off,on}],
needles [{context,depth,seed,expected,off,on}], acceptance {off,on}.
Paths are relative to the manifest. Never accepts a shortlist as a full head.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def rows(path, vocabulary):
    with path.open('rb') as stream:
        while header := stream.read(4):
            if len(header) != 4 or struct.unpack('<I', header)[0] != vocabulary:
                raise ValueError('missing/truncated/full-vocabulary header: ' + str(path))
            payload = stream.read(vocabulary * 4)
            if len(payload) != vocabulary * 4:
                raise ValueError('truncated logits: ' + str(path))
            values = array.array('f')
            values.frombytes(payload)
            if sys.byteorder != 'little':
                values.byteswap()
            if not all(map(math.isfinite, values)):
                raise ValueError('non-finite logit: ' + str(path))
            yield values


def compare(a, b):
    ma, mb = max(a), max(b)
    za = ma + math.log(math.fsum(math.exp(x - ma) for x in a))
    zb = mb + math.log(math.fsum(math.exp(x - mb) for x in b))
    kld = max(0., math.fsum(math.exp(x - za) * (x - za - y + zb) for x, y in zip(a, b)))
    norm = math.sqrt(math.fsum(x*x for x in a) * math.fsum(y*y for y in b))
    cosine = math.fsum(x*y for x, y in zip(a, b)) / norm if norm else float(a == b)
    return kld, cosine, max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)


def evaluate(manifest, root):
    vocab = manifest['vocabulary']
    if vocab < 100000:
        raise ValueError('full Flash-Next vocabulary required')
    report = {'pass': True, 'contexts': []}
    if sorted(x['tokens'] for x in manifest['contexts']) != [32768, 131072, 204800]:
        raise ValueError('32K/128K/200K coverage required')
    for context in manifest['contexts']:
        prompt = root / context['prompt']
        if hashlib.sha256(prompt.read_bytes()).hexdigest() != context['prompt_sha256']:
            raise ValueError('prompt SHA-256 differs')
        if len(prompt.read_text().split()) != context['tokens']:
            raise ValueError('prompt token count differs')
        streams = [rows(root / context[key], vocab) for key in ('off0', 'on0', 'on1', 'off1')]
        klds, cosines, top1, repeat = [], [], [], []
        import itertools
        for values in itertools.zip_longest(*streams):
            if any(v is None for v in values):
                raise ValueError('different capture lengths')
            a, b, c, d = values
            if a != d or b != c:
                raise ValueError("off/off or on/on repeat is not deterministic")
            for on in (b, c):
                kld, cosine, agrees = compare(a, on)
                klds.append(kld); cosines.append(cosine); top1.append(agrees)
            repeat.extend((compare(a, d)[0], compare(b, c)[0]))
        if len(klds) < 512:
            raise ValueError('at least 256 positions across the last 16K text tokens required')
        klds.sort()
        result = {'tokens': context['tokens'], 'positions': len(klds)//2,
                  'mean_kld': math.fsum(klds)/len(klds), 'p95_kld': klds[math.ceil(.95*len(klds))-1],
                  'max_kld': max(klds), 'min_logit_cosine': min(cosines),
                  'teacher_forced_top1': sum(top1)/len(top1), 'repeat_kld': max(repeat)}
        result['pass'] = (result['mean_kld'] <= 2e-4 and result['p95_kld'] <= 5e-4 and
                          result['max_kld'] <= 2e-3 and min(cosines) >= .999 and
                          result['teacher_forced_top1'] >= .99 and max(repeat) <= 1e-8)
        report['contexts'].append(result)
        report['pass'] &= result['pass']
    oracle = manifest['oracle']
    expected = {('int8', 'fp32'), ('bf16', 'bf16')}
    if {(x['kv'], x['state']) for x in oracle} != expected:
        raise ValueError('both oracle tolerance configurations required')
    report['oracle_pass'] = all(x['off_pass'] is True and x['on_pass'] is True and
        x['min_state_cosine'] >= (.999 if x['state'] == 'fp32' else .99) for x in oracle)
    report['greedy'] = []
    for length in (32768, 131072, 204800):
        prompts = [x for x in manifest['greedy'] if x['context'] == length]
        if len(prompts) < 14 or any(len(x['off']) < 128 or len(x['on']) < 128 for x in prompts):
            raise ValueError('14 long prompts x 128 greedy tokens per length required')
        equal = sum(x['off'] == x['on'] for x in prompts)
        agreement = sum(sum(a == b for a, b in zip(x['off'], x['on'])) for x in prompts) / sum(max(len(x['off']), len(x['on'])) for x in prompts)
        report['greedy'].append({'tokens': length, 'exact': equal, 'total': len(prompts), 'agreement': agreement,
                                'pass': equal/len(prompts) >= 13/14 and agreement >= .99})
    needles = manifest['needles']
    required = {(c,d,s) for c in (65536,131072,204800) for d in (.1,.5,.9) for s in (0,1)}
    if {(x['context'],x['depth'],x['seed']) for x in needles} != required:
        raise ValueError('18 needle checks (64K/128K/200K x 10/50/90% x 2 seeds) required')
    report['needle_pass'] = all(x['off'] == x['expected'] and x['on'] == x['expected'] for x in needles)
    acceptance = manifest['acceptance']
    rates = {}
    for arm in ('off', 'on'):
        proposed = acceptance[arm + '_proposed']
        accepted = acceptance[arm + '_accepted']
        if proposed < 4096 or not 0 <= accepted <= proposed:
            raise ValueError('acceptance needs at least 4096 teacher-forced proposals per arm')
        rates[arm] = accepted / proposed
    report['acceptance_rates'] = rates
    report['acceptance_pass'] = abs(rates['on'] - rates['off']) <= .01
    report['pass'] &= report['oracle_pass'] and report['needle_pass'] and report['acceptance_pass'] and all(x['pass'] for x in report['greedy'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(json.loads(args.manifest.read_text()), args.manifest.parent)
        report['manifest_sha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        print(json.dumps(report, indent=2))
        return 0 if report['pass'] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('PF7 quality FAIL: ' + str(error), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
