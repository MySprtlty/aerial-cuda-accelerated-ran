#!/usr/bin/env python3
# work.py : per-slot PUSCH work units of the 5-cell generator pattern (from the launch pattern + TV h5 files, formulas of sizer_add_ul),
# replayed through the sizer rule (sizer_pick, ten generator profiles), with candidate YOLO terms.  PREDICTION tool, CPU only.
import os, re, sys, json, math
D = os.environ.get('TGEN_TV_DIR', '/data/dgist_tgen/testVectors/')     # generator test vectors (only needed to rebuild work5c.json)
PAT = D + 'multi-cell/launch_pattern_F08_5C_tmp_fluct.yaml'
MODEL = os.environ.get('DAPP_SIZER_MODEL', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'estimator', 'sm_model.txt'))
SMS = [20, 28, 36, 44, 52, 60, 68, 76, 92, 108]
K = ('chest', 'eq', 'llr', 'ldpc', 'uci')
LIFT = sorted({2,4,8,16,32,64,128,256,3,6,12,24,48,96,192,384,5,10,20,40,80,160,320,7,14,28,56,112,224,9,18,36,72,144,288,11,22,44,88,176,352,13,26,52,104,208,15,30,60,120,240})
def seg(tb_bytes, rate):
    A = int(tb_bytes) * 8
    if A <= 0: return 0, 0
    R = rate / 10240.0
    bg = 2 if (A <= 292 or (A <= 3824 and R <= 0.67) or R <= 0.25) else 1
    Kcb = 8448 if bg == 1 else 3840
    B = A + (16 if (bg == 2 and A <= 3824) else 24)
    if B <= Kcb: C = 1; Bp = B
    else: C = -(-B // (Kcb - 24)); Bp = B + C * 24
    Kp = -(-Bp // C)
    Kb = 22 if bg == 1 else (10 if B > 640 else 9 if B > 560 else 8 if B > 192 else 6)
    Zc = next((z for z in LIFT if Kb * z >= Kp), LIFT[-1])
    return C, Zc
_tv = {}
def tv_work(tv):
    if tv in _tv: return _tv[tv]
    w = dict.fromkeys(K, 0.0); n = 0
    import h5py
    with h5py.File(D + tv, 'r') as h:
        for k in h.keys():
            if not re.match(r'PDU\d+$', k): continue
            v = h[k][()]; v = v[0] if v.shape else v
            if int(v['type']) != 8: continue
            n += 1
            dm = bin(int(v['DmrsSymbPos'])).count('1'); ns = int(v['NrOfSymbols']); rb = int(v['rbSize']); nl = int(v['nrOfLayers'])
            ds = max(0, ns - dm); bm = int(v['pduBitmap'])
            chest = rb * 12 * dm * nl; eq = rb * 12 * ds * nl; llr = eq * int(v['qamModOrder'])
            ldpc = 0
            if (bm & 1) and int(v['TBSize']) > 0:
                C, Zc = seg(int(v['TBSize']), int(v['targetCodeRate'])); ncb = int(v['numCb']) or C
                ldpc = ncb * Zc if Zc > 0 else ncb
            uci = (int(v['harqAckBitLength']) + int(v['csiPart1BitLength'])) if (bm & 2) else 0
            for a, b in zip(K, (chest, eq, llr, ldpc, uci)): w[a] += b
    _tv[tv] = (w, n); return _tv[tv]
def load_model():
    m = {'start': {}, 'w_par': {}, 'w_ins': {}}
    for l in open(MODEL):
        f = l.split()
        if len(f) != 2 or not f[0].startswith('PUSCH.'): continue
        k = f[0][6:]; v = f[1]
        if k.startswith('start_us.'): m['start'][int(k[9:])] = float(v)
        elif k.startswith('w_par.'): m['w_par'][k[6:]] = float(v)
        elif k.startswith('w_ins.'): m['w_ins'][k[6:]] = float(v)
        elif k != 'kind': m[k] = float(v)
    return m
M = load_model(); LIMIT = M['complete_by_us'] - M['margin_us']
def work_us(w, s): return sum(w[k] * (M['w_ins'][k] + M['w_par'][k] * M['s_ref'] / s) for k in K)
def pattern():
    s = open(PAT).read(); parts = s[s.index('SCHED:'):].split('- config:')[1:]
    out = []
    for p in parts:
        slot = int(float(re.search(r'slot: *([\d.]+)', p).group(1)))
        tvs = re.findall(r'(TVnr_ULMIX_\d+_gNB_FAPI_s0\.h5)', p)
        out.append((slot, tvs))
    return out
def pick(w, slot, extra=0.0, term=None):
    """term: None | ('mul', a) | ('add', us) | ('inv', c_us_at_sref) | ('floor', sm)"""
    st = M['start'][slot % 20]
    def t_of(s):
        wk = work_us(w, s)
        if term and term[0] == 'mul': wk *= (1 + term[1])
        elif term and term[0] == 'add': wk += term[1]
        elif term and term[0] == 'inv': wk += term[1] * M['s_ref'] / s
        return st + extra + wk
    for s in SMS:
        if term and term[0] == 'floor' and s < term[1]: continue
        t = t_of(s)
        if t <= LIMIT: return s, t, True
    return SMS[-1], t_of(SMS[-1]), False
if __name__ == '__main__':
    pat = pattern(); UL = (7, 8, 9, 17, 18, 19)
    print('model: limit %.1f, start %s, w_par %s' % (LIMIT, M['start'], M['w_par']))
    rows = {}
    for slot, tvs in pat:
        if slot % 20 not in UL or not tvs: continue
        w = dict.fromkeys(K, 0.0); n = 0
        for tv in tvs:
            ww, nn = tv_work(tv); n += nn
            for k in K: w[k] += ww[k]
        rows[slot] = (w, n, tvs)
    json.dump({str(s): rows[s][0] for s in rows}, open(sys.argv[1] if len(sys.argv) > 1 else 'work5c.json', 'w'))
    print('frame slot | PUSCH PDUs | chest eq llr ldpc uci | work_us at 20 / 28 / 36 / 44 SM | start | budget | plan (t_us)')
    for f in range(4):
        vec = []
        for u in UL:
            s = f * 20 + u
            if s not in rows: continue
            w, n, tvs = rows[s]; sm, t, ok = pick(w, s)
            vec.append(sm)
            print('  f%d s%2d | %2d | %s | %s | %.1f | %.1f | %d (%.0f%s)' % (f, u, n, ' '.join('%.0f' % w[k] for k in K), ' / '.join('%.0f' % work_us(w, x) for x in (20, 28, 36, 44)), M['start'][u], LIMIT - M['start'][u], sm, t, '' if ok else ' INFEASIBLE'))
        print('  frame %d plan vector %s' % (f, tuple(vec)))
