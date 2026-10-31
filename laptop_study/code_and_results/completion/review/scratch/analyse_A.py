import numpy as np, json
CL = r"C:\Dev\flood-forecast-demo\completion\results"
out = {}
for key, claim in [("Australia_30", "rain_lag_Australia_30.json"), ("Mozambique_480", "rain_lag_Mozambique_480.json")]:
    z = np.load(f"A_{key}.npz"); D, R = z["D"], z["R"]
    c = json.load(open(f"{CL}\{claim}"))
    N = len(D) - 1
    dV = np.diff(D)          # gain over interval j (mm), j=0..N-1
    print("==", key, "N intervals", N, "n rain files", len(R), "claimed n_intervals", c["n_intervals"])
    lags = c["lags"]
    rows = []
    for k in lags:
        # pairs (dV[j], R[j+k])  valid j
        j = np.arange(N); jj = j + k; m = (jj >= 0) & (jj < len(R))
        r = np.corrcoef(dV[j[m]], R[jj[m]])[0, 1]
        # variant using only the N rain files 0..N-1 (drop last rain file)
        m2 = (jj >= 0) & (jj < N)
        r2 = np.corrcoef(dV[j[m2]], R[jj[m2]])[0, 1]
        rows.append((k, r, r2, c["corr_dV"][lags.index(k)], int(m.sum())))
    print(" lag | recomputed(all valid pairs) | recomputed(rain idx<N) | claimed | n pairs | diff(valid pairs)")
    for k, r, r2, cl, n in rows:
        print(f" {k:+d} | {r:.6f} | {r2:.6f} | {cl:.6f} | {n} | {abs(r-cl):.2e}")
    # regression lag 0
    x = R[:N]; y = dV
    A = np.vstack([x, np.ones(N)]).T
    slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]
    print(" slope %.6f (claimed %.6f) intercept %.6f (claimed %.6f)" % (slope, c["k0_slope_mmdepth_per_mmrain"], icpt, c["k0_intercept_mm"]))
    print(" total gain (sum dV) %.5f claimed %.5f ; sum rain(0..N-1) %.5f claimed k0_total_rain %.5f ; sum rain (all files) %.5f claimed rain_total %.5f"
          % (dV.sum(), c["k0_total_gain_mm"], R[:N].sum(), c["k0_total_rain_mm"], R.sum(), c["rain_total_mm"]))
    print(" ratio gain/rain %.5f ; expected 0.5 h * intensity => 0.5" % (dV.sum() / R[:N].sum()))
    # active: intervals with rain>0
    act = (R[:N] > 0).sum(); print(" n intervals with rain>0 (file j):", act, "claimed n_active", c["n_active"])
    # a few derived checks
    print(" argmax corr lag:", lags[int(np.argmax([r[1] for r in rows]))], "claimed best", c["best_lag_dV"])
    # weighted: correlation restricted to active
    m = R[:N] > 0
    print(" corr lag0 active only: %.6f" % np.corrcoef(dV[m], R[:N][m])[0, 1])
    # residual of dV - 0.5*R
    res = dV - 0.5 * R[:N]
    print(" max |dV - 0.5 R|: %.4f mm ; rms %.5f ; mean dV %.5f mean R %.5f" % (np.abs(res).max(), np.sqrt((res**2).mean()), dV.mean(), R[:N].mean()))
    # rain lag: is gain really from file j (not j-1)?  compare k=+1, -1 explicitly
    out[key] = {"lag0": rows[lags.index(0)][1], "lag+1": rows[lags.index(1)][1], "lag-1": rows[lags.index(-1)][1], "slope": slope, "icpt": icpt}
print(json.dumps(out, indent=1))
