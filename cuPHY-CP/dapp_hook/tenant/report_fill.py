#!/usr/bin/env python3
"""report_fill.py <report.html>: insert the KPI table (kpi_html.py) and the estimate-accuracy summary into the report page at the
<!--KPI_TABLE--> marker (idempotent: the block between <div id="kpi-table"> and </div> is replaced)."""
import os, re, subprocess, sys
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T = os.path.join(R, "cuPHY-CP", "dapp_hook", "tenant")
page = sys.argv[1]
table = subprocess.run([sys.executable, os.path.join(T, "kpi_html.py")], capture_output=True, text=True).stdout
acc = subprocess.run([sys.executable, os.path.join(T, "est_accuracy.py")], capture_output=True, text=True).stdout
acc_html = "<h3>추정치 정확도 (RA에 보낸 est_ms 대 실제 서버 시간)</h3><pre style=\"font-family:var(--fm);font-size:12px;white-space:pre-wrap\">" + acc.replace("&", "&amp;").replace("<", "&lt;") + "</pre>" if acc.strip() else ""
s = open(page).read()
new = '<div id="kpi-table">\n' + table + "\n" + acc_html + "\n</div>"
s2 = re.sub(r'<div id="kpi-table">.*?</div>\n</section>', new + "\n</section>", s, count=1, flags=re.S)
if s2 == s: sys.exit("marker block not found")
open(page, "w").write(s2); print("report filled: %d table rows" % table.count("<tr>"))
