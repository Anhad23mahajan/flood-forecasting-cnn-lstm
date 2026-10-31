summary_rows = []

for dataset, m in measurements.items():
    wettest_step = int(np.argmax(m["wet_area"]))

    summary_rows.append({
        "Dataset": dataset,
        "Deepest water (m)": np.nanmax(m["deepest_point"]),
        "Most area under water (%)": 100 * m["wet_area"][wettest_step],
        "When (hours after start)": m["hours"][wettest_step],
        "Average depth then (cm)": 100 * m["average_depth"][wettest_step],
    })

display(pd.DataFrame(summary_rows))
