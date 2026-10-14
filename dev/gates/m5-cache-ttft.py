#!/usr/bin/env python3
"""M5 cache replay. Protocol helpers adapted from pr-def/D/measure_m5.py.
Fixture is original synthetic public-domain (CC0) repository inspection text;
no private files, network corpora, chat history, or generated continuation is ingested.
"""
import argparse
import copy
import csv
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers=headers())
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)

def headers():
    result = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if key := os.environ.get("SPLASH_API_KEY"):
        result.update({"x-api-key": key, "Authorization": f"Bearer {key}"})
    return result

def run_stream(base, body):
    body = {**body, "stream": True}
    req = urllib.request.Request(
        base + "/v1/messages", data=json.dumps(body).encode(), headers=headers()
    )
    started = time.monotonic()
    first = None
    usage = {}
    blocks = {}
    stopped = False
    stop_reason = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            kind = event.get("type")
            if kind == "error":
                raise RuntimeError(event)
            if kind == "message_start":
                usage.update(event["message"].get("usage", {}))
            elif kind == "content_block_start":
                block = event["content_block"]
                blocks[event["index"]] = {
                    "type": block["type"],
                    "name": block.get("name"),
                    "text": block.get("text", block.get("thinking", "")),
                }
                if block.get("input"):
                    blocks[event["index"]]["text"] = json.dumps(
                        block["input"], sort_keys=True
                    )
                if text := block.get("text", block.get("thinking", "")):
                    first = first or time.monotonic()
            elif kind == "content_block_delta":
                delta = event["delta"]
                text = delta.get(
                    "text", delta.get("thinking", delta.get("partial_json", ""))
                )
                if text:
                    first = first or time.monotonic()
                    # Keep semantic output streams, ignoring SSE chunk boundaries.
                    blocks[event["index"]]["text"] += text
            elif kind == "message_delta":
                usage.update(event.get("usage", {}))
                stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
            elif kind == "message_stop":
                stopped = True
    if not stopped or stop_reason is None:
        raise RuntimeError("Incomplete generation stream")
    if first is None:
        raise RuntimeError("No generated content arrived")
    streams = [blocks[index] for index in sorted(blocks)]
    encoded = json.dumps(streams, sort_keys=True, ensure_ascii=False).encode()
    return {
        "ttft_s": first - started,
        "wall_s": time.monotonic() - started,
        "usage": usage,
        "output": streams,
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "stop_reason": stop_reason,
    }

def body_for(model, messages):
    return {
        "model": model,
        "system": "You are a coding assistant reading tool results from one session.",
        "messages": messages,
        "tools": [
            {
                "name": "read_file",
                "description": "Read a repository file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
    }

def same_generation(left, right):
    return (left["output"], left["stop_reason"]) == (
        right["output"],
        right["stop_reason"],
    )

def tool_turn(index, lines):
    call = f"read_{index}"
    data = "".join(
        f"file={index} line={row} status=checked value={(row * 17 + index) % 997}\n"
        for row in range(lines)
    )
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Inspect the next file."},
                {
                    "type": "tool_use",
                    "id": call,
                    "name": "read_file",
                    "input": {"path": f"src/file_{index}.txt"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call, "content": data}],
        },
    ]

def count(base, body):
    return request(base, "/v1/messages/count_tokens", body)["input_tokens"]

def prompt_tokens(base, body):
    from server.api_shapes import anthropic_to_chat_prompt

    chat = anthropic_to_chat_prompt(body, thinking_resolver=lambda value: value)
    text = request(base, "/apply-template", chat)["prompt"]
    return request(base, "/tokenize", {"content": text})["tokens"]


def launcher_module():
    path = Path(os.environ.get('FN2_LAUNCHER', Path(__file__).resolve().parents[1] / 'scripts/splash-fn2'))
    loader = importlib.machinery.SourceFileLoader('fn2', str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def cache_bytes(directory):
    return sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) if directory.exists() else 0


class Server:
    """Own only this Popen process; never look up/kill listeners by port."""
    def __init__(self, config, cache, disk, grid, output):
        self.config, self.cache, self.disk, self.grid, self.output = config, cache, disk, grid, output

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        command = self.config['command'][:]
        for flag in ('--persistent-cache', '--cache-dir', '--max-cache-disk', '--port'):
            if flag in command:
                i = command.index(flag)
                del command[i:i + (1 if flag == '--persistent-cache' else 2)]
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        if port in (4096, 8003, 8004):
            raise RuntimeError('Reserved port selected; rerun')
        command += ['--port', str(port), '--max-cache-disk', self.config['size'] if self.disk else '0']
        if self.disk:
            command += ['--persistent-cache', '--cache-dir', str(self.cache)]
        self.base = f'http://127.0.0.1:{port}'
        env = dict(os.environ, PYTHONPATH=self.config['tree'], HF_HUB_OFFLINE='1',
                   SPLASH_GDN_CHECKPOINT_TOKENS=str(self.grid))
        (self.output / 'command.json').write_text(json.dumps(command, indent=2))
        self.log = (self.output / 'server.log').open('wb')
        self.process = subprocess.Popen(command, cwd=self.config['tree'], env=env,
                                        stdin=subprocess.DEVNULL, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Server exited; see ' + str(self.output))
                try:
                    # Short readiness timeout; generation requests get 1800 seconds.
                    with urllib.request.urlopen(self.base + '/v1/models', timeout=2) as r:
                        json.load(r)
                    return self
                except (OSError, ValueError):
                    time.sleep(2)
            raise RuntimeError('Readiness timeout')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                # No next arm while a previous engine might still own GPU memory.
                raise RuntimeError(f'Graceful shutdown failed, inspect owned PID {self.process.pid}')
        self.log.close()


def fixture(base, model):
    """Six growing prompts, actual template token count within 100 of each target."""
    messages = [{'role': 'user', 'content': 'Inspect these public example files and track checked values.'}]
    question = {'role': 'user', 'content': 'Report the final file number and checked status in one sentence.'}
    bodies = []
    for index, target in enumerate(range(20000, 120001, 20000)):
        low, high = 1, 20000
        best = None
        while low <= high:
            lines = (low + high) // 2
            candidate = messages + tool_turn(index, lines)
            body = body_for(model, candidate + [question])
            actual = count(base, body)
            if best is None or abs(actual-target) < best[0]:
                best = (abs(actual-target), candidate, body)
            if actual < target:
                low = lines + 1
            else:
                high = lines - 1
        if best[0] > 100:
            raise RuntimeError(f'Cannot calibrate fixture to {target} tokens')
        messages = best[1]
        bodies.append(best[2])
    edited = copy.deepcopy(bodies[-1])
    # Edit fourth tool result: shared prefix about 60K; same-length payload.
    edited['messages'][8]['content'][0]['content'] = edited['messages'][8]['content'][0]['content'].replace('status=checked', 'status=revised', 1)
    bodies.append(edited)
    return bodies


def measure(server, body):
    before = request(server.base, '/status')
    result = run_stream(server.base, body)
    after = request(server.base, '/status')
    result.update(actual_input_tokens=count(server.base, body), status_before=before, status_after=after)
    result['cache_read_tokens'] = result['usage'].get('cache_read_input_tokens', 0)
    return result


def experiment(config, output, rounds, settle, include_grid=True):
    rows, bodies = [], None
    cases = [('memory', False, 0), ('disk', True, 0)]
    if include_grid:
        cases.append(('grid', True, 16384))
    for repeat in range(rounds):
        for name, disk, grid in (cases if repeat % 2 == 0 else cases[::-1]):
            directory = output / f'r{repeat+1}-{name}'
            directory.mkdir()
            cache = directory / 'cache'
            if config.get('size') and shutil.disk_usage(output).free < (int(config['size'][:-1]) + 32) * 1024**3:
                raise RuntimeError('Insufficient internal SSD space for cache quota plus 32 GiB reserve')
            with Server(config, cache, disk, grid, directory / 'growing') as server:
                if bodies is None:
                    bodies = fixture(server.base, 'splash-flashnext')
                    (output / 'public-requests.json').write_text(json.dumps(bodies))
                # Exact LCP permits rejecting an invalid restore beyond the edit.
                left, right = prompt_tokens(server.base, bodies[5]), prompt_tokens(server.base, bodies[6])
                shared = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
                for turn, body in enumerate(bodies, 1):
                    result = measure(server, body)
                    result.update(repeat=repeat+1, case=name, phase='grow' if turn <= 6 else 'deep-edit', turn=turn)
                    if turn == 7:
                        result['shared_tokens'] = shared
                        if result['cache_read_tokens'] > shared:
                            raise RuntimeError('Restore beyond edited prefix')
                    rows.append(result)
                    (output / 'receipts.json').write_text(json.dumps(rows, indent=2))
                    print(f"r{repeat+1} {name} turn {turn}: {result['actual_input_tokens']} tokens, TTFT {result['ttft_s']:.3f}s, hit {result['cache_read_tokens']}", flush=True)
                time.sleep(settle)
            # Same cache directory after orderly exit; measure first request, no warm-up.
            persisted = cache_bytes(cache)
            with Server(config, cache, disk, grid, directory / 'restart') as server:
                result = measure(server, bodies[-1])
                result.update(repeat=repeat+1, case=name, phase='restart', turn=7, disk_bytes=persisted)
                rows.append(result)
            # Fresh process, disk disabled: same request and grid, no prefix priming.
            with Server(config, directory / 'unused', False, grid, directory / 'cold') as server:
                result = measure(server, bodies[-1])
                result.update(repeat=repeat+1, case=name, phase='cold', turn=7)
                rows.append(result)
            (output / 'receipts.json').write_text(json.dumps(rows, indent=2))
            # Only this freshly created arm's cache; retain receipts/logs, avoid accumulation.
            if cache.exists():
                shutil.rmtree(cache)
    return rows


def report(rows, output):
    reference = {r['turn']: r for r in rows if r['case'] == 'memory' and r['repeat'] == 1 and r['phase'] in ('grow', 'deep-edit')}
    identity = all(same_generation(r, reference[r['turn']]) for r in rows)
    restarts = [r for r in rows if r['phase'] == 'restart']
    hits = all(r['cache_read_tokens'] > 0 and r['disk_bytes'] > 0 for r in restarts if r['case'] != 'memory')
    cold = all(r['cache_read_tokens'] == 0 for r in rows if r['phase'] == 'cold' or (r['phase'] == 'restart' and r['case'] == 'memory'))
    def percentile(values, p):
        values = sorted(values)
        return values[round((len(values)-1)*p)]
    summary = []
    for case, phase, turn in sorted({(r['case'], r['phase'], r['turn']) for r in rows}):
        group = [r for r in rows if (r['case'],r['phase'],r['turn']) == (case,phase,turn)]
        values = [r['ttft_s'] for r in group]
        summary.append(dict(case=case, phase=phase, turn=turn, n=len(values), ttft_p50_s=statistics.median(values), ttft_p95_s=percentile(values,.95)))
    def median(case, phase):
        return statistics.median(r['ttft_s'] for r in rows if r['case'] == case and r['phase'] == phase)
    disk_gain = 1 - median('disk', 'restart') / median('disk', 'cold')
    grid_present = any(r['case'] == 'grid' for r in rows)
    grid_gain = 1 - median('grid', 'deep-edit') / median('disk', 'deep-edit') if grid_present else None
    grid_engaged = grid_present and statistics.median(r['cache_read_tokens'] for r in rows if r['case'] == 'grid' and r['phase'] == 'deep-edit') > statistics.median(r['cache_read_tokens'] for r in rows if r['case'] == 'disk' and r['phase'] == 'deep-edit')
    result = dict(disk_restart_ttft_reduction_fraction=disk_gain, grid_edit_ttft_reduction_fraction=grid_gain, grid_saved_prefill=grid_engaged, disk_benefit_observed=identity and hits and cold and disk_gain >= .1, grid_benefit_observed=identity and grid_present and grid_engaged and grid_gain >= .1, outputs_identical=identity, persistent_restart_hits=hits, cold_cache_empty=cold,
                  summary=summary, note='TTFT includes HTTP and first semantic output; startup excluded. OS file cache is not flushed. Small-n percentiles are descriptive; inspect per-turn receipts. D historical grid retention after restart is not assumed.')
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    with (output / 'ttft.csv').open('w') as f:
        fields = ['repeat','case','phase','turn','actual_input_tokens','ttft_s','wall_s','cache_read_tokens','output_sha256','disk_bytes']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k != 'summary'}, indent=2))
    return identity and hits and cold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true', help='Explicit M5-only live measurement; starts isolated servers')
    mode.add_argument('--mock-test', action='store_true', help='CPU-only mocked protocol/lifecycle tests, no servers')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-grid', action='store_true', help='Isolate CACHE: memory/disk ABBA, D off')
    parser.add_argument('--rounds', type=int, default=2, help='Alternating ABC/CBA rounds; >=2, use 10+ for useful p95')
    parser.add_argument('--settle', type=float, default=10, help='Idle seconds before orderly restart; hits still verified')
    args = parser.parse_args()
    if args.mock_test:
        mock_test(); return
    if os.environ.get('SPLASH_M5_WINDOW') != '1':
        parser.error('Set SPLASH_M5_WINDOW=1 in the reserved M5 window')
    if 'M5' not in subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'], text=True):
        parser.error('Live measurements require M5')
    if args.rounds < 2 or args.settle < 0:
        parser.error('Require rounds >=2 and settle >=0')
    config = launcher_module().configuration()
    tree = Path(config['tree'])
    for path in (tree / '.venv/bin/python', tree / 'build/splash'):
        if not os.access(path, os.X_OK):
            parser.error(f'Prebuilt executable required: {path}')
    if 'SPLASH_GDN_CHECKPOINT_TOKENS' not in (tree / 'runtime/main.mm').read_text():
        parser.error('All three arms need the SAME prebuilt D+#410+A+B tree; default A+B lacks D')
    if not (Path(config['root']) / 'tokenizer').is_dir() or config['size'] == '0':
        parser.error('Prepared FN2_MODEL_ROOT and nonzero FN2_CACHE_SIZE required')
    sys.path.insert(0, config['tree'])
    output = (args.output or Path(config['cache']).parent / ('fn2-measure-' + time.strftime('%Y%m%d-%H%M%S'))).expanduser().resolve()
    parent = output
    while not parent.exists(): parent = parent.parent
    if parent.stat().st_dev != Path.home().stat().st_dev:
        parser.error('Measurement caches must be on internal SSD')
    output.mkdir(parents=True, exist_ok=False)
    config['commit'] = subprocess.check_output(['git','-C', config['tree'],'rev-parse','HEAD'], text=True).strip()
    config['dirty'] = subprocess.check_output(['git','-C', config['tree'],'status','--porcelain'], text=True)
    with (tree / 'build/splash').open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
        config['binary_sha256'] = h.hexdigest()
    (output / 'configuration.json').write_text(json.dumps(config, indent=2))
    def interrupted(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    rows = experiment(config, output, args.rounds, args.settle, include_grid=not args.no_grid)
    passed = report(rows, output)
    print('Receipts:', output)
    raise SystemExit(0 if passed else 1)


def mock_test():
    import io
    import tempfile
    import unittest
    from unittest.mock import patch

    module = sys.modules[__name__]
    fn2 = launcher_module()

    class Tests(unittest.TestCase):
        def test_launcher_flags_and_protected_ports(self):
            with patch.dict(os.environ, {'FN2_PORT':'8004','FN2_CACHE_SIZE':'8G','FN2_GRID':'16384'}, clear=True), patch.object(fn2.shutil, 'disk_usage', return_value=type('D', (), {'free':64*1024**3})()):
                c = fn2.configuration()
                self.assertIn('--persistent-cache', c['command'])
                self.assertEqual(c['grid'], 16384)
                self.assertEqual(c['command'][c['command'].index('--ple-ssd-cache')+1], '2G')
                for port in ('4096', '8003'):
                    with patch.dict(os.environ, {'FN2_PORT':port}):
                        with self.assertRaises(ValueError): fn2.configuration()
                with patch.dict(os.environ, {'FN2_CACHE_SIZE':'0','FN2_PLE_STREAM':'0'}):
                    c = fn2.configuration()
                    self.assertNotIn('--persistent-cache', c['command'])
                    self.assertNotIn('--cache-dir', c['command'])
                    self.assertNotIn('--ple-ssd-cache', c['command'])

        def test_stop_refuses_reused_pid(self):
            with patch.object(fn2, 'identity', return_value='unrelated'), patch.object(fn2.os, 'killpg') as kill:
                with self.assertRaises(RuntimeError):
                    fn2.stop({'pid':1234,'identity':'old'}, Path('/unused'))
                kill.assert_not_called()

        def test_fixture_growth(self):
            def tokens(base, body):
                return len(json.dumps(body)) // 4
            with patch.object(module, 'count', side_effect=tokens):
                bodies = fixture('mock', 'model')
            self.assertEqual(len(bodies), 7)
            for body, target in zip(bodies, range(20000,120001,20000)):
                self.assertLessEqual(abs(tokens('',body)-target),100)
            self.assertEqual(bodies[0]['messages'][:-1], bodies[1]['messages'][:3])
            self.assertNotEqual(bodies[5], bodies[6])

        def test_sse_ttft_and_truncation(self):
            events = [
                {'type':'message_start','message':{'usage':{'input_tokens':20}}},
                {'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}},
                {'type':'content_block_delta','index':0,'delta':{'text':'Hello'}},
                {'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'cache_read_input_tokens':10}},
                {'type':'message_stop'}]
            data = b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)
            with patch.object(urllib.request, 'urlopen', return_value=io.BytesIO(data)), patch.object(time,'monotonic',side_effect=[100,102,103]):
                r=run_stream('http://mock', {})
            self.assertEqual(r['ttft_s'],2)
            self.assertEqual(r['usage']['cache_read_input_tokens'],10)
            with patch.object(urllib.request,'urlopen',return_value=io.BytesIO(b'data: {"type":"message_start","message":{}}\n')):
                with self.assertRaises(RuntimeError): run_stream('http://mock', {})

        def test_full_matrix_no_servers(self):
            active=[]
            class FakeServer:
                def __init__(self,config,cache,disk,grid,output):
                    self.base='mock'; self.disk=disk; self.output=output; self.cache=cache
                    self.phase=output.name
                def __enter__(self):
                    self.output.mkdir(parents=True)
                    active.append(self)
                    return self
                def __exit__(self,*_):
                    if self.disk:
                        self.cache.mkdir(exist_ok=True)
                        (self.cache/'mock-cache').write_bytes(b'public')
            bodies=[{'turn':i} for i in range(1,8)]
            def measured(server,body):
                hit=0 if server.phase=='cold' or (server.phase=='restart' and not server.disk) else 10
                return dict(ttft_s=1.,wall_s=2.,actual_input_tokens=min(body['turn'],6)*20000,cache_read_tokens=hit,
                            usage={},output=[{'text':'same'}],stop_reason='end_turn',output_sha256='mock')
            with tempfile.TemporaryDirectory() as tmp, patch.object(module,'Server',FakeServer), patch.object(module,'fixture',return_value=bodies), patch.object(module,'prompt_tokens',return_value=list(range(100))), patch.object(module,'measure',side_effect=measured), patch.object(time,'sleep'), patch.object(subprocess,'Popen',side_effect=AssertionError('No processes')), patch.object(socket,'socket',side_effect=AssertionError('No sockets')):
                out=Path(tmp)
                rows=experiment({},out,2,0)
                self.assertEqual(len(rows),54)
                self.assertEqual(len(active),18)
                self.assertTrue(report(rows,out))
                next(r for r in rows if r['case']=='disk' and r['phase']=='restart')['cache_read_tokens']=0
                self.assertFalse(report(rows,out))
                next(r for r in rows if r['case']=='grid')['output']=[{'text':'different'}]
                self.assertFalse(report(rows,out))

    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    main()
