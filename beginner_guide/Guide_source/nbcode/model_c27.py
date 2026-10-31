# Results table for Model A
result_rows = []

for pair in PAIRS:
    for dataset, part in [(pair["train"], "unseen time (last 15%)"), (pair["test"], "never-seen region")]:
        evaluation = evaluations[dataset]
        cells = evaluation["cells"].sum()
        copy_error = evaluation["copy_squared_error"].sum()
        model_error = evaluation["model_squared_error"].sum()
        rule_error = evaluation["rule_squared_error"].sum()

        result_rows.append({
            "Region": REGION_NAMES[dataset],
            "Tested on": part,
            "Forecasts": len(evaluation),
            "Copy last map: error (cm)": round(100 * np.sqrt(copy_error / cells), 3),
            "CNN-LSTM: error (cm)": round(100 * np.sqrt(model_error / cells), 3),
            "CNN-LSTM + water rule: error (cm)": round(100 * np.sqrt(rule_error / cells), 3),
            "CNN-LSTM: skill": round(1 - model_error / copy_error, 3),
            "CNN-LSTM + water rule: skill": round(1 - rule_error / copy_error, 3),
            "Copy last map: CSI": round(evaluation["copy_csi"].mean(), 4),
            "CNN-LSTM + water rule: CSI": round(evaluation["rule_csi"].mean(), 4),
        })

result_table = pd.DataFrame(result_rows)
display(result_table)
save_table(result_table, "03_forecast_results")
