# candidate YOLO terms through the sizer rule (CALCULATION in model coordinates; 5 generator cells only)
import json, sys
import work as W
wk = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
UL = (7, 8, 9, 17, 18, 19)
LV = {0: 'L1(low)', 1: 'L2(mid)', 3: 'L3(auto)', 2: 'L4(high)'}
def pick_pos(w, slot, ypos):
    st = W.M['start'][slot % 20]; y = ypos.get(slot % 20, 0.0)
    for s in W.SMS:
        t = st + W.work_us(w, s) + y
        if t <= W.LIMIT: return s, W.LIMIT - (st + W.work_us(w, s))
    return -W.SMS[-1], W.LIMIT - (st + W.work_us(w, W.SMS[-1]))
def vec(fr, ypos):
    out = []
    for s in UL:
        w = wk[fr * 20 + s]; out.append(pick_pos(w, s, ypos))
    return out
def show(name, ypos):
    print(f'== {name}')
    for fr in (0, 1, 3, 2):
        v = vec(fr, ypos)
        print(f'  {LV[fr]:9s} plan ' + ' '.join(f'{p:4d}' for p, _ in v) + '   max %3d' % max(abs(p) for p, _ in v) + '   model margin ' + ' '.join(f'{m:4.0f}' for _, m in v))
show('today (no term)', {})
for y in (60, 90, 105, 120, 150, 185):
    show(f'first+second U slots +{y} us (7,8,17,18)', {7: y, 8: y, 17: y, 18: y})
for y in (120, 185):
    show(f'first slots +120, second slots +{y}', {7: 120, 17: 120, 8: y, 18: y})
show('all six U slots +120', {s: 120 for s in UL})
