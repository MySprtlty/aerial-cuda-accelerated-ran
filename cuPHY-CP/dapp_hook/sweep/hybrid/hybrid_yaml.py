#!/usr/bin/env python3
"""hybrid_yaml.py --ncells N --env benetel_cell.env --lab-cuphy FILE --lab-l2a FILE --ru FILE --testmac FILE
                  [--out-cuphy FILE] [--out-l2a FILE] --nvipc-out FILE [--base-dir DIR] [--wait-all 0|1]

Generates the hybrid L1 configuration FROM THE BASELINE FILES (sweep/hybrid/baseline/ = the researchers' proven COTS-UE setup):
  out-cuphy  = baseline cuphycontroller_config_benetal.yaml (every global key, the Benetel cell block = L1 cell 0)
               + cell_group_num N+1, + the lab nics entry (0000:01:00.0), + lab cell blocks 0..N-1 as L1 cells 1..N,
               + the keys listed in L1_KEYS_FROM_LAB / L1_TOP_KEYS_FROM_LAB copied from the lab file (capacity, instrumentation, cores),
               + L1_GLOBAL_OVERRIDES. Default: written over the lab file (the L1 is started as 'cuphycontroller_scf F08_CG1').
  out-l2a    = baseline l2_adapter_config_P5G_GH.yaml + thread cores (L2A_TIMER_CORE/L2A_MSG_CORE) + nvipc_config_file + L2A_KEYS_FROM_LAB lines
  nvipc-out  = two transports: 0 = OAI_PREFIX (baseline 'nvipc', phy_cells [0]), 1 = TESTMAC_PREFIX (phy_cells [1..N]); pool sizes from the baseline
  ru         = lab ru-emulator config: fix_beta_dl mirrored from the baseline (0) and payload_validation RU_PAYLOAD_VALIDATION
  testmac    = lab test_mac_config.yaml: shm prefix -> TESTMAC_PREFIX
Text-level edits only (comments and flow style survive; the L1 yaml parser is not libyaml-strict). Every edit asserts its match count; exit 1 on a miss.
The lab files are read before anything is written. Prints one line per decision."""
import argparse
import os
import re
import sys


def read_env(path):
    env = {}
    for ln in open(path):
        ln = ln.split("#", 1)[0].strip()
        if "=" in ln:
            k, v = ln.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def die(msg):
    sys.exit("hybrid_yaml: " + msg)


def get_val(text, key, indent):
    """value (comment stripped) of '^<indent>key: value' or None"""
    m = re.search(r"^%s%s[ \t]*:[ \t]*([^\n]*)$" % (indent, re.escape(key)), text, flags=re.M)   # 'key :' (space before the colon) occurs in the baseline
    if not m:
        return None
    v = m.group(1)
    if v.lstrip().startswith("#"):
        return None
    return v.split("#", 1)[0].strip() if not v.strip().startswith(("[", "{")) else re.sub(r"\s+#.*$", "", v).strip()


def set_val(text, key, value, indent, append_after=None):
    """set '^<indent>key: value'; append after the line matching append_after (regex) when the key is absent"""
    pat = r"^(%s%s[ \t]*:)[ \t]*[^\n]*$" % (indent, re.escape(key))
    new, n = re.subn(pat, lambda m: "%s %s" % (m.group(1), value), text, count=1, flags=re.M)
    if n == 1:
        return new, "set"
    if append_after is None:
        die("key %s not found and no insertion point" % key)
    m = re.search(append_after, text, flags=re.M)
    if not m:
        die("insertion point %r not found for key %s" % (append_after, key))
    return text[:m.end()] + "\n%s%s: %s" % (indent, key, value) + text[m.end():], "added"


def cell_blocks(text):
    """(text up to and including the '  cells:' line, [(name_index, block_text)]) for the '    - name: O-RU k' items"""
    cm = re.search(r"^  cells:[ \t]*\n", text, flags=re.M)
    if not cm:
        die("'  cells:' not found")
    body = text[cm.end():]
    end = re.compile(r"^(?![ \t]*#)(\S|  \S)", flags=re.M).search(body)      # next key with indent <= 2 (comment lines ignored) = end of the list
    body = body[:end.start()] if end else body
    starts = [m for m in re.finditer(r"^    - name: ", body, flags=re.M)]
    out = []
    for i, m in enumerate(starts):
        b = body[m.start():starts[i + 1].start() if i + 1 < len(starts) else len(body)]
        nm = re.match(r"^    - name: O-RU (\d+)", b)
        out.append((int(nm.group(1)) if nm else -1, b))
    return text[:cm.end()], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncells", type=int, required=True, help="emulator cells (L1 cells 1..N); the Benetel is L1 cell 0")
    ap.add_argument("--env", required=True)
    ap.add_argument("--lab-cuphy", required=True); ap.add_argument("--lab-l2a", required=True)
    ap.add_argument("--ru", required=True); ap.add_argument("--testmac", required=True)
    ap.add_argument("--out-cuphy", default=None); ap.add_argument("--out-l2a", default=None)
    ap.add_argument("--nvipc-out", required=True)
    ap.add_argument("--base-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--wait-all", default=None)
    a = ap.parse_args()
    E = read_env(a.env); N = a.ncells
    if N < 1 or N > 19:
        die("--ncells must be 1..19 (lab cell blocks 0..18 exist; the L2A has 20 instances)")
    out_cuphy = a.out_cuphy or a.lab_cuphy; out_l2a = a.out_l2a or a.lab_l2a
    base_cuphy = os.path.join(a.base_dir, E["BASE_CUPHY"]); base_l2a = os.path.join(a.base_dir, E["BASE_L2A"])
    B = open(base_cuphy).read(); L = open(a.lab_cuphy).read()
    BL = open(base_l2a).read(); LL = open(a.lab_l2a).read()
    RU = open(a.ru).read(); TM = open(a.testmac).read()
    log = ["base L1 %s (%d bytes), base L2A %s" % (base_cuphy, len(B), base_l2a)]
    l2a_name = os.path.basename(out_l2a)

    # ---------------- L1 yaml: baseline header + Benetel block + lab blocks
    header, bcells = cell_blocks(B)
    if len(bcells) != 1 or bcells[0][0] != 0:
        die("baseline L1 yaml must have exactly one cell block 'O-RU 0' (found %s)" % [c[0] for c in bcells])
    benetel_block = bcells[0][1]
    if ("'%s'" % E["BENETEL_NIC"]) not in benetel_block:
        die("baseline cell block does not use BENETEL_NIC %s" % E["BENETEL_NIC"])
    _, lcells = cell_blocks(L)
    lab_by_idx = {k: b for k, b in lcells}
    for k in range(N):
        if k not in lab_by_idx:
            die("lab cell block 'O-RU %d' missing (test_config.sh must keep the full cells list)" % k)
    header, how = set_val(header, "l2adapter_filename", l2a_name, "")
    header, how = set_val(header, "cell_group_num", str(N + 1), "  ")
    log.append("cell_group_num %d (1 Benetel + %d emulator cells)" % (N + 1, N))
    for key in E.get("L1_TOP_KEYS_FROM_LAB", "").split():
        v = get_val(L, key, "")
        if v is None:
            log.append("top-level %s: not in lab file - baseline value kept" % key); continue
        header, how = set_val(header, key, v, "", append_after=r"^aerial_sdk_version:.*$")
        log.append("top-level %s <- lab: %s (%s)" % (key, v, how))
    for key in E.get("L1_KEYS_FROM_LAB", "").split():
        v = get_val(L, key, "  ")
        if v is None:
            log.append("%s: not in lab file - baseline value kept" % key); continue
        bv = get_val(B, key, "  ")
        header, how = set_val(header, key, v, "  ", append_after=r"^cuphydriver_config:[ \t]*$")
        log.append("%s <- lab: %s (baseline %s, %s)" % (key, v, bv if bv is not None else "absent", how))
    for kv in E.get("L1_GLOBAL_OVERRIDES", "").split():
        key, v = kv.split("=", 1)
        bv = get_val(B, key, "  ")
        header, how = set_val(header, key, v, "  ", append_after=r"^cuphydriver_config:[ \t]*$")
        log.append("%s = %s (override; baseline %s, %s)" % (key, v, bv if bv is not None else "absent", how))
    # nics: baseline entry (block style, aerial00) + the lab entries for the emulator (flow style '- {nic: ...}'), inserted before cus_port_failover
    lab_nics = [m.group(0) for m in re.finditer(r"^    - \{nic: '([0-9a-fA-F:.]+)'[^\n]*\}[^\n]*$", L, flags=re.M)
                if m.group(1) != E["BENETEL_NIC"]]
    if not lab_nics:
        die("no lab nics entry (flow style) found for the emulator NIC")
    lab_nics = [re.sub(r"\s+#.*$", "", x) + "   # RU emulator (lab)" for x in lab_nics]
    # BENETEL_CELL_INDEX: 'last' (default) = the verified full6 layout: emulator NIC = nics[0] and emulator cells = L1 cells 0..N-1, the Benetel
    # = nics[1] / L1 cell N. 'first' = Benetel as nics[0] / cell 0: the cuphydriver has NIC-0 / cell-0 assumptions (Debug task nullptr 10-29 ue1,
    # 100 % PUSCH CRC failure on the emulator cells 10-29 ue2) - kept only for experiments.
    benetel_last = E.get("BENETEL_CELL_INDEX", "last") != "first"
    if benetel_last:
        m = re.search(r"^  nics:[ \t]*\n", header, flags=re.M)
        if not m:
            die("'  nics:' not found in the baseline header")
        header = header[:m.end()] + "\n".join(lab_nics) + "\n" + header[m.end():]
        log.append("nics: %s (lab, index 0) + baseline %s (index 1)" % (", ".join(re.search(r"nic: '([^']+)'", x).group(1) for x in lab_nics), E["BENETEL_NIC"]))
    else:
        m = re.search(r"^  cus_port_failover:", header, flags=re.M)
        if not m:
            die("'  cus_port_failover:' not found in the baseline header (nics insertion point)")
        header = header[:m.start()] + "\n".join(lab_nics) + "\n" + header[m.start():]
        log.append("nics: baseline %s (index 0) + %s" % (E["BENETEL_NIC"], ", ".join(re.search(r"nic: '([^']+)'", x).group(1) for x in lab_nics)))
    bb = benetel_block if benetel_block.endswith("\n") else benetel_block + "\n"
    blocks = []
    for k in range(N):
        b = lab_by_idx[k]
        if "'%s'" % E["BENETEL_NIC"] in b:
            die("lab block %d uses the Benetel NIC - lab file not in the lab state?" % k)
        if not benetel_last:    # lab blocks become L1 cells 1..N: names O-RU 1..N, cell_id 2..N+1 (unique M-plane ids; the emulator matches by MAC/VLAN/eAxC)
            b, n1 = re.subn(r"^    - name: O-RU %d(?!\d)[^\n]*$" % k, "    - name: O-RU %d" % (k + 1), b, count=1, flags=re.M)
            b, n2 = re.subn(r"^      cell_id:[ \t]*\d+[^\n]*$", "      cell_id: %d" % (k + 2), b, count=1, flags=re.M)
            if n1 != 1 or n2 != 1:
                die("lab block %d: name/cell_id rewrite failed" % k)
        blocks.append(b if b.endswith("\n") else b + "\n")
    if benetel_last:            # Benetel block verbatim except name/cell_id (L1 cell N, M-plane id N+1 after the lab ids 1..N)
        bb, n1 = re.subn(r"^    - name: O-RU 0(?!\d)[^\n]*$", "    - name: O-RU %d (Benetel)" % N, bb, count=1, flags=re.M)
        bb, n2 = re.subn(r"^      cell_id:[ \t]*\d+[^\n]*$", "      cell_id: %d" % (N + 1), bb, count=1, flags=re.M)
        if n1 != 1 or n2 != 1:
            die("baseline block: name/cell_id rewrite failed")
        blocks.append(bb)
        log.append("cells: %d lab blocks (L1 cells 0..%d, unchanged) + baseline block as 'O-RU %d (Benetel)' = L1 cell %d" % (N, N - 1, N, N))
    else:
        blocks.insert(0, bb)
        log.append("cells: baseline block = L1 cell 0 + %d lab blocks (O-RU 0..%d -> L1 cells 1..%d)" % (N, N - 1, N))
    out = header + "".join(blocks)
    oai_cell = N if benetel_last else 0

    # ---------------- L2 adapter: baseline P5G_GH + cores + nvipc file + lab hook line
    t = BL if BL.endswith("\n") else BL + "\n"
    if t.rstrip().endswith("..."):          # drop the YAML end marker so that appended lines stay inside the document
        t = t.rstrip()[:-3].rstrip() + "\n"
    tc, mc = E.get("L2A_TIMER_CORE"), E.get("L2A_MSG_CORE")
    for sect, core in (("timer_thread_config", tc), ("message_thread_config", mc)):
        if not core:
            continue
        m = re.search(r"^%s:\n((?:  .*\n)+)" % sect, t, flags=re.M)
        if not m:
            die("L2A section %s not found" % sect)
        body, n = re.subn(r"^(  cpu_affinity:)[ \t]*\d+", lambda mm: "%s %s" % (mm.group(1), core), m.group(1), count=1, flags=re.M)
        if n != 1:
            die("L2A %s cpu_affinity not found" % sect)
        t = t[:m.start(1)] + body + t[m.end(1):]
        log.append("L2A %s cpu_affinity %s (baseline 9)" % (sect, core))
    t, n = re.subn(r"^nvipc_config_file:[ \t]*[^\n]*$", "nvipc_config_file: %s" % os.path.basename(a.nvipc_out), t, count=1, flags=re.M)
    if n != 1:
        die("L2A nvipc_config_file not found")
    log.append("L2A nvipc_config_file -> %s" % os.path.basename(a.nvipc_out))
    for key in E.get("L2A_KEYS_FROM_LAB", "").split():
        m = re.search(r"^%s:[^\n]*$" % re.escape(key), LL, flags=re.M)
        if not m:
            log.append("L2A %s: not in lab file - skipped" % key); continue
        line = m.group(0)
        if re.search(r"^%s:" % re.escape(key), t, flags=re.M):
            t = re.sub(r"^%s:[^\n]*$" % re.escape(key), lambda mm: line, t, count=1, flags=re.M); log.append("L2A %s <- lab (replaced)" % key)
        else:
            t, n = re.subn(r"^(allowed_tick_error:[^\n]*\n)", lambda mm: mm.group(1) + line + "\n", t, count=1, flags=re.M)
            if n != 1:
                die("L2A allowed_tick_error line (insertion point) not found")
            log.append("L2A %s <- lab (added)" % key)
    # staticSlotCellMask (hybrid L2A addition, cuphyl2adapter nv_phy_module.cpp): the static*SlotNum / staticSsb* values apply only to the cells in this
    # mask = the emulator cells (test vectors for slot 0); the Benetel cell keeps real slot numbers / the dynamic SFN. Written always (harmless without the keys).
    mask = sum(1 << c for c in (range(0, N) if benetel_last else range(1, N + 1)))
    if re.search(r"^staticSlotCellMask:", t, flags=re.M):
        t = re.sub(r"^staticSlotCellMask:[^\n]*$", "staticSlotCellMask: %d" % mask, t, count=1, flags=re.M)
    else:
        t, n = re.subn(r"^(allowed_tick_error:[^\n]*\n)", lambda mm: mm.group(1) + "staticSlotCellMask: %d\n" % mask, t, count=1, flags=re.M)
        if n != 1:
            die("L2A allowed_tick_error line (insertion point) not found")
    log.append("L2A staticSlotCellMask %d (0x%X = emulator cells; static slot numbers do not apply to the Benetel cell)" % (mask, mask))
    out_l2a_text = t
    # pool sizes for the transport file: the baseline inline transport's shm_config/app_config children
    tb = re.search(r"^transport:\n(.*?)(?=^\S|\Z)", t, flags=re.M | re.S)
    if not tb:
        die("L2A transport block not found")
    body = tb.group(1)
    shm = re.search(r"^  shm_config:\n((?:    .*\n)+)", body, flags=re.M).group(1)
    app = re.search(r"^  app_config:\n((?:    .*\n)+)", body, flags=re.M).group(1)
    shm = "".join(ln for ln in shm.splitlines(True) if not re.match(r"\s*(primary|prefix):", ln))
    app = "".join(re.sub(r"\s+#.*$", "", ln.rstrip("\n")) + "\n" for ln in app.splitlines(True))

    # ---------------- RU emulator: fix_beta_dl mirrored from the baseline, payload check off with fix_beta_dl 0
    fb = get_val(B, "fix_beta_dl", "  ")
    if fb is None:
        die("baseline has no fix_beta_dl")
    RU, n = re.subn(r"^(\s*fix_beta_dl:)[ \t]*\d[^\n]*$", lambda m: "%s %s" % (m.group(1), fb), RU, count=1, flags=re.M)
    if n != 1:
        die("ru fix_beta_dl not found")
    log.append("ru-emulator fix_beta_dl %s (= baseline)" % fb)
    pv = E.get("RU_PAYLOAD_VALIDATION", "")
    if fb == "0" and pv != "":
        RU, n = re.subn(r"^(\s*payload_validation:)[ \t]*\d[^\n]*$", lambda m: "%s %s" % (m.group(1), pv), RU, count=1, flags=re.M)
        if n != 1:
            die("ru payload_validation not found")
        log.append("ru-emulator payload_validation %s (fix_beta_dl 0: the TV's bfp9 bytes are compressed at beta 65536, memcmp fails every DL slot)" % pv)

    # ---------------- testMAC: second transport prefix
    tp = E.get("TESTMAC_PREFIX", "nvipc1")
    TM, n = re.subn(r"^(\s*shm_config:\s*\{prefix:)\s*\w+(\})", lambda m: "%s %s%s" % (m.group(1), tp, m.group(2)), TM, count=1, flags=re.M)
    if n != 1:
        die("testMAC shm_config prefix line not found")
    log.append("testMAC shm prefix %s" % tp)

    # ---------------- nvIPC transports
    wait_all = a.wait_all if a.wait_all is not None else E.get("WAIT_ALL_CELL_CONFIG", "1")
    oai_prefix, pcap_core = E.get("OAI_PREFIX", "nvipc"), E.get("NVIPC_PCAP_CORE", "14")
    def transport(tid, cells, prefix, pcap):
        ind = lambda s, n: "".join(" " * n + ln if ln.strip() else ln for ln in s.splitlines(True))
        ap_ = re.sub(r"^(\s*pcap_enable:)[ \t]*\d", lambda m: "%s %s" % (m.group(1), pcap), app, count=1, flags=re.M)
        ap_ = re.sub(r"^(\s*pcap_(?:shm_caching|file_saving)_cpu_core:)[ \t]*\d+", lambda m: "%s %s" % (m.group(1), pcap_core), ap_, flags=re.M)
        return ("  - transport_id: %d\n    phy_cells: [%s]\n    type: shm\n    shm_config:\n      prefix: %s\n%s    app_config:\n%s"
                % (tid, ", ".join(map(str, cells)), prefix, ind(shm, 2), ind(ap_, 2)))
    t_oai = transport(1 if benetel_last else 0, [oai_cell], oai_prefix, E.get("OAI_NVIPC_PCAP", "0"))
    t_mac = transport(0 if benetel_last else 1, list(range(0, N)) if benetel_last else list(range(1, N + 1)), tp, "0")
    nv = ("# generated by hybrid_yaml.py: OAI gNB on the baseline prefix '%s' (gnb.conf unchanged, cell_id 0 on the wire) -> L1 cell %d (Benetel);\n"
          "# testMAC on '%s' (%d emulator cells, wire ids 0..%d) -> L1 cells %s. Rules: transport_id == index, sum(len(phy_cells)) ==\n"
          "# cell_group_num, no duplicates (cuphyl2adapter/lib/nvPHY/nv_phy_mac_transport.cpp:416-486). pcap: baseline pcap_enable kept for the OAI transport.\n"
          "---\naerial_sdk_version: 25-3-cubb\nwait_all_cell_config: %s\ntransport:\n%s%s"
          % (oai_prefix, oai_cell, tp, N, N - 1, "0..%d" % (N - 1) if benetel_last else "1..%d" % N, wait_all,
             t_mac if benetel_last else t_oai, t_oai if benetel_last else t_mac))
    log.append("nvipc file %s: OAI %s -> cell [%d] pcap %s, testMAC %s -> cells %s, wait_all_cell_config %s" % (
        a.nvipc_out, oai_prefix, oai_cell, E.get("OAI_NVIPC_PCAP", "0"), tp, "[0..%d]" % (N - 1) if benetel_last else "[1..%d]" % N, wait_all))
    log.append("OAI_CELL=%d" % oai_cell)

    # ---------------- write (all inputs were read above)
    open(out_cuphy, "w").write(out); open(out_l2a, "w").write(out_l2a_text)
    open(a.ru, "w").write(RU); open(a.testmac, "w").write(TM); open(a.nvipc_out, "w").write(nv)
    log.append("written: %s, %s, %s, %s, %s" % (out_cuphy, out_l2a, a.ru, a.testmac, a.nvipc_out))
    print("\n".join("hybrid_yaml: " + s for s in log))


if __name__ == "__main__":
    main()
