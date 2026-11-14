# The Linux launcher (zero-launcher.sh runs it): checks the system, lists the models of the catalog (downloaded or
# not), downloads one with scripts/hf-download.py, starts bin/serverz and opens the web UI. The Windows launcher is
# scripts/zero-launcher.ps1 (a window); this one is a terminal menu with the same steps, standard library only.
#   ./zero-launcher.sh                    the menu
#   ./zero-launcher.sh --check            what the system has and lacks (GPU runtime, device access, CPU, RAM)
#   ./zero-launcher.sh --list             the catalog, with what is downloaded
#   ./zero-launcher.sh --download 4b      download a catalog model (no server)
#   ./zero-launcher.sh --model 4b         download if needed, then start the server (also a path to a .gguf)
#   ./zero-launcher.sh --model 4b --install-service   a systemd user service for it instead of starting it here
# Options: --port N (8090), --host H (127.0.0.1), --gpus N, --no-browser, --yes (no questions), and anything after
# `--` goes to serverz as it is (e.g. -- --slots 2; not --port / --host, which the launcher must know).
# The catalog below follows scripts/model-catalog.ps1 (tests/check_catalog_sync.py checks that they agree). Flash
# Next's AutoRound import (flash-ar4) and the speech models are setup.ps1's on Windows and not offered here.
import argparse, os, shutil, signal, socket, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVERZ = os.path.join(ROOT, 'bin', 'serverz')
CATALOG = [
    dict(key='4b', name='Qwen3.5 4B (fastest; MTP, reads images)', repo='unsloth/Qwen3.5-4B-MTP-GGUF', gb=3.7, ram=16, mtp=True,
         files=['Qwen3.5-4B-UD-Q4_K_XL.gguf', 'mmproj-F16.gguf']),
    dict(key='9b', name='Qwen3.5 9B (MTP, reads images)', repo='unsloth/Qwen3.5-9B-MTP-GGUF', gb=7.1, ram=16, mtp=True,
         files=['Qwen3.5-9B-UD-Q4_K_XL.gguf', 'mmproj-F16.gguf']),
    dict(key='35b', name='Qwen3.6 35B-A3B (fast mixture of experts; MTP, reads images)', repo='unsloth/Qwen3.6-35B-A3B-MTP-GGUF',
         gb=19.1, ram=32, mtp=True, files=['Qwen3.6-35B-A3B-UD-IQ4_XS.gguf', 'mmproj-F16.gguf']),
    dict(key='27b', name='Qwen3.8 27B (best answers, slowest; MTP, reads images)', repo='ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF',
         gb=13.1, ram=32, mtp=True, files=['Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf', 'mmproj-Qwen3.8-27B-BF16.gguf']),
    dict(key='flash', name='Qwen3.8 Flash Next (512 experts on the SSD; reads images)', repo='ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
         gb=84.5, disk=141, ram=32, mtp=False,
         files=['IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf', 'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf',
                'mmproj-Qwen3.8-Flash-Next-f16.gguf'],
         extra=dict(repo='bartowski/Qwen3.8-Flash-Next-GGUF', file='mmproj-Qwen3.8-Flash-Next-f16.gguf', into='IQ3_S')),
    dict(key='g12', name='Gemma 4 12B (reads images; drafter)', repo='google/gemma-4-12B-it-qat-q4_0-gguf', gb=7.4, ram=32, mtp=True,
         files=['gemma-4-12b-it-qat-q4_0.gguf', 'mmproj-gemma-4-12b-it-qat-q4_0.gguf', 'mtp-gemma-4-12B-it.gguf'],
         drafter=dict(repo='unsloth/gemma-4-12B-it-qat-GGUF', file='mtp-gemma-4-12B-it.gguf')),
    dict(key='g31', name="Gemma 4 31B (Google's largest dense Gemma; slow; drafter)", repo='google/gemma-4-31B-it-qat-q4_0-gguf', gb=19.1,
         ram=32, mtp=True, files=['gemma-4-31B_q4_0-it.gguf', 'gemma-4-31B-it-mmproj.gguf', 'mtp-gemma-4-31B-it.gguf'],
         drafter=dict(repo='unsloth/gemma-4-31B-it-qat-GGUF', file='mtp-gemma-4-31B-it.gguf')),
    dict(key='e4b', name='Gemma 4 E4B (images, audio; drafter)', repo='google/gemma-4-E4B-it-qat-q4_0-gguf', gb=6.2, ram=16, mtp=True,
         files=['gemma-4-E4B_q4_0-it.gguf', 'gemma-4-E4B-it-mmproj.gguf', 'mtp-gemma-4-E4B-it.gguf'],
         drafter=dict(repo='unsloth/gemma-4-E4B-it-qat-GGUF', file='mtp-gemma-4-E4B-it.gguf')),
]
# (the environment the server and the downloader read for the cache: a systemd unit gets them too)
CACHE_ENV = ('HF_HUB_CACHE', 'HF_HOME', 'XDG_CACHE_HOME', 'HF_TOKEN_PATH')


def hub_dir():  # the Hugging Face cache, as the engine (src/platform.h hf_home_hub) and scripts/hf-download.py find it
    if os.environ.get('HF_HUB_CACHE'): return os.environ['HF_HUB_CACHE']
    if os.environ.get('HF_HOME'): return os.path.join(os.environ['HF_HOME'], 'hub')
    x = os.environ.get('XDG_CACHE_HOME', '')
    if os.name != 'nt' and x.startswith('/'): return os.path.join(x, 'huggingface', 'hub')
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')


def repo_root(repo):
    return os.path.join(hub_dir(), 'models--' + repo.replace('/', '--'))


def snapshots(repo):  # the repository's snapshot folders: the one refs/main points at first, then the others
    root = repo_root(repo)
    out = []
    try:
        with open(os.path.join(root, 'refs', 'main')) as f:
            main = os.path.join(root, 'snapshots', f.read().strip())
        if os.path.isdir(main): out.append(main)
    except OSError:
        pass
    sd = os.path.join(root, 'snapshots')
    if os.path.isdir(sd):
        out += [os.path.join(sd, c) for c in sorted(os.listdir(sd)) if os.path.join(sd, c) not in out and os.path.isdir(os.path.join(sd, c))]
    return out


def has_file(d, f):
    if os.path.isfile(os.path.join(d, *f.split('/'))): return True
    # (a file copied in from another repository may sit in a subfolder: Flash Next's projector in IQ3_S/; found by its
    #  name anywhere in the snapshot, as model-catalog.ps1's Test-ArcModel does)
    return any(os.path.basename(f) in fs for _, _, fs in os.walk(d))


def model_path(m):  # the first file of a downloaded catalog model (a snapshot that holds all its files), or None
    for d in snapshots(m['repo']):
        if all(has_file(d, f) for f in m['files']):
            return os.path.join(d, *m['files'][0].split('/'))
    return None


def present_gb(m):  # what is already on disk of a model (complete files, partial downloads), any snapshot
    n = 0
    for f in m['files']:
        best = 0
        for d in snapshots(m['repo']):
            p = os.path.join(d, *f.split('/'))
            for q in (p, p + '.incomplete'):
                try:
                    best = max(best, os.path.getsize(q))
                except OSError:
                    pass
        n += best
    return n / 1e9


def ram_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'): return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


def check():  # [(ok, what)]: a missing runtime library or device access is a fix to make, not a crash later
    out = []
    out.append((sys.platform.startswith('linux'), 'Linux (this launcher is for Linux; on Windows use zero-launcher.cmd)'))
    out.append((os.access(SERVERZ, os.X_OK), f'bin/serverz built ({SERVERZ}; ./build.sh builds it)'))
    try:
        with open('/proc/cpuinfo') as f:
            flags = next((l for l in f if l.startswith('flags')), '')
        out.append((all(x in flags.split() for x in ('avx2', 'fma', 'f16c')), 'CPU with AVX2, FMA and F16C'))
    except OSError:
        out.append((False, 'CPU flags (/proc/cpuinfo unreadable)'))
    import ctypes
    # (libgomp: GCC's OpenMP runtime the programs link to; minimal installs and containers may lack it)
    for lib, pkg in (('libze_loader.so.1', 'libze1'), ('libOpenCL.so.1', 'ocl-icd-libopencl1'), ('libgomp.so.1', 'libgomp1')):
        try:
            ctypes.CDLL(lib)
            out.append((True, f'{lib}'))
        except OSError:
            out.append((False, f'{lib} (sudo apt install {pkg})'))
    icds = [f for f in (os.listdir('/etc/OpenCL/vendors') if os.path.isdir('/etc/OpenCL/vendors') else []) if 'intel' in f.lower()]
    out.append((bool(icds), "Intel's OpenCL driver (/etc/OpenCL/vendors/intel*.icd; sudo apt install intel-opencl-icd libze-intel-gpu1,"
                            " or Intel's packages for Battlemage cards: docs/linux-port.md)"))
    nodes = [os.path.join('/dev/dri', n) for n in (os.listdir('/dev/dri') if os.path.isdir('/dev/dri') else []) if n.startswith('renderD')]
    if not nodes and os.path.exists('/dev/dxg'):  # (WSL2: the Windows driver's GPU through /dev/dxg, no render node)
        nodes = ['/dev/dxg']
    out.append((bool(nodes), 'a GPU render node (/dev/dri/renderD*, or /dev/dxg in WSL2)'))
    if nodes:
        out.append((any(os.access(n, os.R_OK | os.W_OK) for n in nodes),
                    'access to the render node (sudo usermod -aG render $USER, then log in again)'))
    r = ram_gb()
    out.append((r >= 15, f'RAM {r:.0f} GB (16 GB for the 4B / 9B / E4B, 32 GB for the others)'))
    return out


def print_check():
    ok = True
    for good, what in check():
        print(('  ok    ' if good else '  MISS  ') + what)
        ok = ok and good
    return ok


def find(key):
    for m in CATALOG:
        if m['key'] == key: return m
    return None


def copy_into(src, dst):  # (atomic: a copy stopped halfway never looks like the finished file)
    tmp = dst + '.incomplete'
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def download(m, yes):
    os.makedirs(hub_dir(), exist_ok=True)
    need = max(0.0, m.get('disk', m['gb']) - present_gb(m))
    free = shutil.disk_usage(hub_dir()).free / 1e9
    if free < need + 2:
        print(f'{m["name"]}: needs ~{need:.0f} GB more, {free:.0f} GB free in {hub_dir()} (HF_HUB_CACHE moves the cache)')
        return False
    if not yes and input(f'Download {m["name"]} (~{m["gb"]:g} GB) into {hub_dir()}? [y/N] ').strip().lower() != 'y':
        return False
    dl = [sys.executable, '-I', os.path.join(ROOT, 'scripts', 'hf-download.py')]
    own = [f for f in m['files'] if f not in (m.get('drafter', {}).get('file'), m.get('extra', {}).get('file'))]
    if subprocess.call(dl + [m['repo']] + own) != 0:
        print('the download did not finish: run the same command again to continue it')
        return False
    snap = next(iter(snapshots(m['repo'])), None)
    if not snap:
        print(f'no snapshot of {m["repo"]} in {hub_dir()} after the download')
        return False
    # (a file from another repository copied next to the model, where the engine looks for it: Gemma's drafter,
    #  Flash Next's image projector)
    for x, into in ((m.get('drafter'), ''), (m.get('extra'), m.get('extra', {}).get('into', ''))):
        if not x: continue
        if subprocess.call(dl + [x['repo'], x['file']]) != 0:
            print('the download did not finish: run the same command again to continue it')
            return False
        src = next((os.path.join(d, *x['file'].split('/')) for d in snapshots(x['repo'])
                    if os.path.isfile(os.path.join(d, *x['file'].split('/')))), None)
        if not src:
            print(f'{x["repo"]}/{x["file"]}: not found in the cache after its download')
            return False
        dst_dir = os.path.join(snap, *into.split('/')) if into else snap
        os.makedirs(dst_dir, exist_ok=True)
        copy_into(src, os.path.join(dst_dir, os.path.basename(x['file'])))
    if model_path(m) is None:
        print(f'{m["name"]}: some of its files are still missing (a new commit of the repository during the download?); '
              'run it again')
        return False
    return True


def server_cmd(path, mtp, a):
    cmd = [SERVERZ, '-m', path, '--port', str(a.port), '--ctx', 'auto', '--think-budget', '8192']
    if mtp: cmd.append('--mtp')
    if a.host: cmd += ['--host', a.host]
    if a.gpus and a.gpus > 1: cmd += ['--gpus', str(a.gpus)]
    return cmd + a.extra


def exposed(host):
    return host not in ('', '127.0.0.1', 'localhost', '::1')


def install_service(cmd, a):
    if any(('\n' in c or '\r' in c) for c in cmd):
        print('--install-service: a path or option holds a line break; a unit file cannot carry it')
        return 1
    base = os.environ.get('XDG_CONFIG_HOME', '')
    base = base if base.startswith('/') else os.path.join(os.path.expanduser('~'), '.config')
    d = os.path.join(base, 'systemd', 'user')
    os.makedirs(d, exist_ok=True)
    unit = os.path.join(d, 'inferzero.service')
    # (systemd's own quoting, not the shell's: each word in double quotes with \ and " escaped, and its % specifiers
    #  and $ variables doubled so a path holding them stays as it is)
    def sq(c):
        return '"' + c.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'
    env = ''.join(f'Environment={sq(k + "=" + os.environ[k])}\n' for k in CACHE_ENV if os.environ.get(k))
    existed = os.path.exists(unit)
    # (KillMode=control-group, the default: the server's children (terminal tool, MCP servers) end with it;
    #  WorkingDirectory: the package, where the server finds its kernels and web UI)
    with open(unit, 'w', newline='') as f:
        f.write('[Unit]\nDescription=inferzero server\nAfter=network.target\n\n[Service]\n'
                f'WorkingDirectory={sq(ROOT)}\n{env}ExecStart={" ".join(sq(c) for c in cmd)}\n'
                'Restart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n')
    print(f'{"replaced" if existed else "wrote"} {unit}')
    if exposed(a.host):
        print(f'note: --host {a.host} makes the server reachable from your network; it has no password of its own')
    print('start it now and at every login:\n  systemctl --user daemon-reload\n  systemctl --user enable inferzero\n'
          '  systemctl --user restart inferzero\nits log: journalctl --user -u inferzero -f\n'
          '(to keep it running while logged out: sudo loginctl enable-linger $USER)')
    return 0


def port_busy(host, port):  # something already answers there (another server: it would answer our /health)
    h = host or '127.0.0.1'
    if h == '0.0.0.0': h = '127.0.0.1'
    if h == '::': h = '::1'
    try:
        with socket.create_connection((h, port), timeout=1):
            return True
    except OSError:
        return False


def run_server(cmd, a):
    host = a.host or '127.0.0.1'
    # (serverz's IPv6 socket is IPv6-only: '::' is reached through ::1, '0.0.0.0' through 127.0.0.1)
    shown = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(host, host)
    url = f'http://{"[" + shown + "]" if ":" in shown else shown}:{a.port}/'
    if port_busy(host, a.port):
        print(f'port {a.port} is in use (another server?): stop it, or choose another with --port')
        return 1
    if exposed(host):
        print(f'note: --host {host} makes the server reachable from your network; it has no password of its own')
    print('starting:', ' '.join(cmd))
    p = subprocess.Popen(cmd, cwd=ROOT)
    # (passed on: Ctrl+C reaches the server from the terminal too, which is harmless (it ends on the first); a
    #  SIGINT or SIGTERM sent to the launcher alone (kill, an IDE's stop, systemd) must reach it as well)
    signal.signal(signal.SIGINT, lambda s, _f: p.send_signal(s))
    signal.signal(signal.SIGTERM, lambda s, _f: p.send_signal(s))
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # (never through http_proxy)
    opened = False
    while p.poll() is None:
        if not opened:
            try:
                with direct.open(url + 'health', timeout=2) as r:
                    if r.status == 200:
                        opened = True
                        print(f'\nready: {url}  (Ctrl+C stops the server)')
                        if not a.no_browser and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')) and shutil.which('xdg-open'):
                            # (a session of its own: Ctrl+C or a closed terminal must not take the browser with it)
                            subprocess.Popen(['xdg-open', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                             start_new_session=True)
            except Exception:
                pass
        time.sleep(1)
    rc = p.returncode
    return 128 - rc if rc < 0 else rc  # (killed by signal N: 128 + N, as a shell reports it)


def menu(a):
    ram = ram_gb()
    print('inferzero: models (* downloaded)\n')
    for i, m in enumerate(CATALOG, 1):
        mark = '*' if model_path(m) else ' '
        fit = '' if ram == 0 or ram + 1 >= m['ram'] else f'  (wants {m["ram"]} GB of RAM)'
        print(f' {i:2d} {mark} {m["name"]}, ~{m["gb"]:g} GB{fit}')
    print('\n  c  check the system    q  quit')
    s = input('\nmodel number: ').strip().lower()
    if s == 'c':
        return 0 if print_check() else 1
    if s in ('q', ''):
        return 0
    if not s.isdigit() or not 1 <= int(s) <= len(CATALOG):
        print(f'no model "{s}"')
        return 2
    return start(CATALOG[int(s) - 1], a)


def start(m, a):
    path = model_path(m)
    if not path and not download(m, a.yes):
        return 1
    path = model_path(m)
    if not path:
        return 1
    cmd = server_cmd(path, m['mtp'], a)
    return install_service(cmd, a) if a.install_service else run_server(cmd, a)


def main():
    # (line by line into a pipe, a file or the journal too: "ready" must show when it happens, not at the exit)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description='inferzero on Linux: pick a model, download it, start the server')
    ap.add_argument('--check', action='store_true', help='what the system has and lacks')
    ap.add_argument('--list', action='store_true', help='the catalog, with what is downloaded')
    ap.add_argument('--download', metavar='KEY', help='download a catalog model')
    ap.add_argument('--model', metavar='KEY|FILE', help='start a catalog model (downloaded first if needed) or a .gguf')
    ap.add_argument('--mtp', action='store_true', help='with --model FILE: speculative decoding with its MTP block / drafter')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--host', default='')
    ap.add_argument('--gpus', type=int, default=0)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--yes', action='store_true', help='download without asking')
    ap.add_argument('--install-service', action='store_true', help='a systemd user service for --model instead of starting it')
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        k = argv.index('--')
        argv, extra = argv[:k], argv[k + 1:]
    a = ap.parse_args(argv)
    # (the launcher polls the port it was given: those two go through its own options)
    for x in extra:
        if x.split('=', 1)[0] in ('--port', '--host'):
            ap.error(f'{x.split("=", 1)[0]} after --: give it to the launcher itself (before --)')
    a.extra = extra
    if not 1 <= a.port <= 65535: ap.error('--port: 1..65535')
    if a.check:
        return 0 if print_check() else 1
    if a.list:
        for m in CATALOG:
            p = model_path(m)
            print(f'{m["key"]:6s} {"*" if p else " "} {m["name"]}, ~{m["gb"]:g} GB, {m["ram"]} GB RAM' + (f'\n         {p}' if p else ''))
        return 0
    if a.download:
        m = find(a.download)
        if not m: ap.error(f'--download: no model "{a.download}" (--list shows the keys)')
        return 0 if (model_path(m) or download(m, a.yes)) else 1
    bad = [w for g, w in check() if not g]
    if bad:
        print('missing:\n  ' + '\n  '.join(bad) + '\n(./zero-launcher.sh --check lists everything)')
        if not os.access(SERVERZ, os.X_OK): return 1
    if a.model:
        m = find(a.model)
        if m: return start(m, a)
        if not os.path.isfile(a.model): ap.error(f'--model: no catalog model or file "{a.model}" (--list shows the keys)')
        cmd = server_cmd(os.path.abspath(a.model), a.mtp, a)
        return install_service(cmd, a) if a.install_service else run_server(cmd, a)
    if a.install_service: ap.error('--install-service needs --model')
    return menu(a)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(130)
