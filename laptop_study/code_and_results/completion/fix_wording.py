p = "completion/g_report.py"
s = open(p, encoding="utf-8").read()
pairs = [
 ('note = "" if fl else (f"RETRY after the frozen-seed run collapsed (post hoc): seed {r[\'seed\']}"',
  'note = ("guard disabled after a false abort (post hoc; identical recipe and seed)" if (fl and not r.get("guard", True)) else "") if fl else (f"RETRY after the frozen-seed run was aborted by the guard (post hoc): seed {r[\'seed\']}"'),
 ('COLLAPSED to a constant change map after epoch {ep} (dead-network guard); no result |',
  'ABORTED by the dead-network guard after epoch {ep} (a false alarm: see the guard-disabled row); no result of its own |'),
 ('RETRY (post hoc){extra_}: COLLAPSED again after epoch {ep2}; no result |',
  'RETRY (post hoc){extra_}: also ABORTED by the guard after epoch {ep2}; not pursued further |'),
]
for a, b in pairs:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
