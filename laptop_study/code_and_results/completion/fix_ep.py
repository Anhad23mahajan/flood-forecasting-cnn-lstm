p = "completion/g_report.py"
s = open(p, encoding="utf-8").read()
a = s.index('    out.append("## Exploratory: training length (local CNN')
b = s.index('    # ------------------------------------------------------------- reverse-direction transfers')
new = '''    out.append("## Exploratory: training length (hybrid; 30 epochs = frozen recipe versus 90 epochs; mean over 3 folds)\n")
    out.append("| experiment | model | never-seen, 30 epochs | never-seen, 90 epochs | held-out tiles (mean of fold skills), 30 | held-out tiles, 90 |")
    out.append("|---|---|---|---|---|---|")
    for exp in EXPS:
        for arch_e, lab_e in (("local", "local CNN"), ("clstm", "small ConvLSTM")):
            a = fold_mean(raw_runs(exp, arch_e, n_train=300, epochs=30), frozen_seed=True)
            b = fold_mean(raw_runs(exp, arch_e, n_train=300, epochs=90))
            a2 = fold_mean(raw_runs(exp, arch_e, n_train=300, epochs=30), key="heldout", frozen_seed=True)
            b2 = fold_mean(raw_runs(exp, arch_e, n_train=300, epochs=90), key="heldout")
            if b:
                out.append(f"| {exp} | {lab_e} | {fmt(a[0] if a else None)} | {fmt(b[0])} | {fmt(a2[0] if a2 else None)} | {fmt(b2[0] if b2 else None)} |")
                js.setdefault("epochs90", {})[f"{exp}|{arch_e}"] = dict(ep30=a[0] if a else None, ep90=b[0])
    out.append("")

'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
