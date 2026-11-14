# The Linux launcher's model list (scripts/zero_launcher.py CATALOG) against the Windows one (scripts/model-catalog.ps1):
# every key the launcher offers must have the same repository, files, drafter and extra file there. Run by CI.
#   python3 tests/check_catalog_sync.py
import importlib.util, os, re, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('zl', os.path.join(root, 'scripts', 'zero_launcher.py'))
zl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zl)
with open(os.path.join(root, 'scripts', 'model-catalog.ps1'), encoding='utf-8') as f:
    ps = f.read()
# (each entry starts at `[pscustomobject]@{ key = "..."` and runs to the next one)
starts = [m.start() for m in re.finditer(r'\[pscustomobject\]@\{\s*key\s*=', ps)] + [len(ps)]
entries = {}
for a, b in zip(starts, starts[1:]):
    e = ps[a:b]
    key = re.search(r'key\s*=\s*"([^"]+)"', e).group(1)
    repo = re.search(r'\brepo\s*=\s*"([^"]+)"', e).group(1)
    files = re.findall(r'"([^"]+)"', (re.search(r'files\s*=\s*@\(([^)]*)\)', e) or re.search('()', '')).group(1))
    dr = re.search(r'drafter\s*=\s*@\{\s*repo\s*=\s*"([^"]+)";\s*file\s*=\s*"([^"]+)"', e)
    ex = re.search(r'extra\s*=\s*@\{\s*repo\s*=\s*"([^"]+)";\s*file\s*=\s*"([^"]+)";\s*into\s*=\s*"([^"]+)"', e)
    entries[key] = dict(repo=repo, files=files, drafter=dr.groups() if dr else None, extra=ex.groups() if ex else None)
bad = 0
for m in zl.CATALOG:
    w = entries.get(m['key'])
    if not w:
        print(f'{m["key"]}: not in model-catalog.ps1'); bad += 1; continue
    dr = (m['drafter']['repo'], m['drafter']['file']) if m.get('drafter') else None
    ex = (m['extra']['repo'], m['extra']['file'], m['extra']['into']) if m.get('extra') else None
    for what, mine, theirs in (('repo', m['repo'], w['repo']), ('files', m['files'], w['files']),
                               ('drafter', dr, w['drafter']), ('extra', ex, w['extra'])):
        if mine != theirs:
            print(f'{m["key"]}: {what} differs: launcher {mine!r}, model-catalog.ps1 {theirs!r}'); bad += 1
print(f'catalog sync: {len(zl.CATALOG)} models checked, {bad} difference(s)')
sys.exit(1 if bad else 0)
