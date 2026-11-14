#!/usr/bin/env python3
# run_test.py [out dir] : the real sizer_plan() of dapp_sizer.hpp (cuphydriver stubbed, sizer_yterm_test.cpp) against the Python replay of
# the sizer rule (work.py) for the 24 UL slots of the 5-cell generator pattern (work5c.json).  Builds two binaries with the host g++:
# the header of this tree and the header before the YOLO term (git show BASE).  CPU only, no L1 needed.
#   BASE=<rev> python3 run_test.py        (default BASE = ca7a71e, the last commit before the term)
import json, os, subprocess, sys, tempfile
import work as W
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
INC = os.path.join(ROOT, 'cuPHY-CP', 'dapp_hook', 'include'); HDR = 'cuPHY-CP/dapp_hook/include/dapp_hook/dapp_sizer.hpp'
OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='sizer_yterm_'); os.makedirs(os.path.join(OUT, 'old', 'dapp_hook'), exist_ok=True)
BASE = os.environ.get('BASE', 'ca7a71e')
open(os.path.join(OUT, 'old', 'dapp_hook', 'dapp_sizer.hpp'), 'w').write(subprocess.run(['git', '-C', ROOT, 'show', BASE + ':' + HDR], capture_output=True, text=True, check=True).stdout)
CXX = ['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', os.path.join(HERE, 'sizer_yterm_test.cpp')]
subprocess.run(CXX + ['-I' + INC, '-I' + os.path.join(INC, 'dapp_hook'), '-o', os.path.join(OUT, 'sizer_yterm_test_new')], check=True)
subprocess.run(CXX + ['-DYTERM_OLD', '-I' + os.path.join(OUT, 'old'), '-I' + os.path.join(INC, 'dapp_hook'), '-I' + INC, '-o', os.path.join(OUT, 'sizer_yterm_test_old')], check=True)
wk = {int(k): v for k, v in json.load(open(os.path.join(HERE, 'work5c.json'))).items()}
with open(os.path.join(OUT, 'work5c.txt'), 'w') as f:
    for k in sorted(wk): f.write('%d %r %r %r %r %r\n' % (k, wk[k]['chest'], wk[k]['eq'], wk[k]['llr'], wk[k]['ldpc'], wk[k]['uci']))
UL = (7, 8, 9, 17, 18, 19); LV = {0: 'L1', 1: 'L2', 3: 'L3', 2: 'L4'}
def expect(ypos):
    out = {}
    for s80, w in wk.items():
        st = W.M['start'][s80 % 20]; y = ypos.get(s80 % 20, 0.0)
        base = next((s for s in W.SMS if st + W.work_us(w, s) <= W.LIMIT), None)
        if base is None: out[s80] = (W.SMS[-1], 0); continue          # infeasible without the term: the term is not applied
        sm = next((s for s in W.SMS if st + W.work_us(w, s) + y <= W.LIMIT), W.SMS[-1])
        out[s80] = (sm, 1)
    return out
def run(binary, env_us, mark, hold=None):
    env = dict(os.environ); env['DAPP_SIZER_MODEL'] = W.MODEL
    for k in ('DAPP_SIZER_YOLO_US', 'DAPP_SIZER_YOLO_HOLD_MS'): env.pop(k, None)
    if env_us is not None: env['DAPP_SIZER_YOLO_US'] = env_us
    if hold is not None: env['DAPP_SIZER_YOLO_HOLD_MS'] = str(hold)
    o = subprocess.run([os.path.join(OUT, binary), os.path.join(OUT, 'work5c.txt'), mark], env=env, capture_output=True, text=True, check=True).stdout
    rows = {}; notes = []
    for l in o.splitlines():
        if l.startswith('#'): notes.append(l); continue
        f = l.split(); rows[int(f[0])] = (int(f[2]), int(f[4]), int(f[1]), float(f[3]), int(f[5]))
    return rows, notes
def vec(rows):
    return ' | '.join('%s %s' % (LV[fr], ' '.join('%3d' % rows[fr * 20 + s][0] for s in UL)) for fr in (0, 1, 3, 2))
fails = 0
def case(name, binary, env_us, mark, ypos, hold=None):
    global fails
    rows, notes = run(binary, env_us, mark, hold); exp = expect(ypos)
    bad = [s for s in exp if rows[s][0] != exp[s][0] or rows[s][1] != exp[s][1] or rows[s][2] != rows[s][4]]
    # predicted time reported = model time at the chosen profile, without the margin
    tbad = [s for s in exp if abs(rows[s][3] - (W.M['start'][s % 20] + W.work_us(wk[s], rows[s][0]))) > 0.01]
    ok = not bad and not tbad; fails += 0 if ok else 1
    print('%-4s %-46s %s' % ('PASS' if ok else 'FAIL', name, vec(rows)))
    for n in notes: print('       ' + n)
    if bad or tbad: print('       mismatch slots', bad, tbad)
    return rows
Y120 = {7: 120, 8: 120, 17: 120, 18: 120}; S120 = '7:120,8:120,17:120,18:120'
r_old = case('original header', 'sizer_yterm_test_old', None, 'none', {})
r_off = case('patched, term unset, tenant active', 'sizer_yterm_test_new', None, 'fresh', {})
r_e   = case('patched, term empty string, tenant active', 'sizer_yterm_test_new', '', 'fresh', {})
r_z   = case('patched, term all zero, tenant active', 'sizer_yterm_test_new', '7:0,8:0', 'fresh', {})
same = all(r_old[s][:4] == r_off[s][:4] == r_e[s][:4] == r_z[s][:4] for s in r_old); fails += 0 if same else 1
print('%-4s off == original header, field by field (profile, SMs, predicted time, ok)' % ('PASS' if same else 'FAIL'))
case('term 120 @7,8,17,18, tenant never seen', 'sizer_yterm_test_new', S120, 'none', {})
case('term 120 @7,8,17,18, tenant seen > hold ago', 'sizer_yterm_test_new', S120, 'stale', {})
case('term 120 @7,8,17,18, tenant active', 'sizer_yterm_test_new', S120, 'fresh', Y120)
case('term 120, hold 0 ms, tenant seen 5 ms ago', 'sizer_yterm_test_new', S120, 'stale', {}, hold=0)
case('term 150 @7,8,17,18, tenant active', 'sizer_yterm_test_new', '7:150,8:150,17:150,18:150', 'fresh', {7: 150, 8: 150, 17: 150, 18: 150})
case('term 120 first / 185 second, tenant active', 'sizer_yterm_test_new', '7:120,8:185,17:120,18:185', 'fresh', {7: 120, 8: 185, 17: 120, 18: 185})
case('parser: spaces, junk, out-of-range, negative', 'sizer_yterm_test_new', ' 7:120, 8:120 ,x:50,25:100,9:-3,:44,18:120,17:120,7', 'fresh', Y120)
case('term 280 @8 (no profile fits at L3/L4 -> largest)', 'sizer_yterm_test_new', '8:280', 'fresh', {8: 280})
print('FAILED %d' % fails if fails else 'ALL PASS'); sys.exit(1 if fails else 0)
