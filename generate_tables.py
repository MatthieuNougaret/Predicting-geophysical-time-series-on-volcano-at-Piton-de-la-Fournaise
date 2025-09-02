import pandas as pd

def generate_latex_table(csv_path, tex_path, caption, label):
    """
    Generate a formatted LaTeX table from a CSV of metrics, with nice column names.
    """

    # Load results
    df = pd.read_csv(csv_path)

    # Add model names as first column
    model_names = ["Last", "Linear", "Dense", "Conv", "LSTM", "Transformer"]
    if len(df) == len(model_names):
        df.insert(0, "Model", model_names)
    else:
        raise ValueError("Number of rows in CSV does not match number of model names.")

    # --- Rename columns for LaTeX prettiness ---
    rename_map = {
        "Model": "Model",
        "Train_R2_median": r"Train $R^2$",
        "Train_pRMSE_median": r"Train pRMSE",
        "Train_MASE_median": r"Train MASE",
        "Valid_R2_median": r"Valid. $R^2$",
        "Valid_pRMSE_median": r"Valid. pRMSE",
        "Valid_MASE_median": r"Valid. MASE",
        "Test_R2_median": r"Test $R^2$",
        "Test_pRMSE_median": r"Test pRMSE",
        "Test_MASE_median": r"Test MASE",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --- Convert to LaTeX table ---
    latex_table = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        column_format="l" + "c"*(len(df.columns)-1),  # l for model, c for metrics
        escape=False
    )

    # Save to file
    with open(tex_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {tex_path}")



generate_latex_table(
    csv_path="./test_seismicity/performance_compiled_paper.csv",
    tex_path="./tables/seismicity_performance_table.tex",
    caption="Performance of machine learning models for the seismicity case. Reported peak-weighted Root Mean Squared Error (pRMSE), Mean Absolute Scaled Error (MASE) and Coefficient of determination (R$^2$ score) are average and standard deviation values calculated using predictions from 10 different runs per model. Best scores per metric are marked in bold.",
    label="tab:ml_perf_seis"
)

generate_latex_table(
    csv_path="./test_gnss/performance_compiled_paper.csv",
    tex_path="./tables/gnss_performance_table.tex",
    caption="Performance of machine learning models for the GNSS case. Reported peak-weighted Root Mean Squared Error (pRMSE), Mean Absolute Scaled Error (MASE) and Coefficient of determination (R$^2$ score) are average and standard deviation values calculated using predictions from 10 different runs per model. Best scores per metric are marked in bold.",
    label="tab:ml_perf_gnss"
)

generate_latex_table(
    csv_path="./test_both/performance_compiled_paper.csv",
    tex_path="./tables/both_performance_table.tex",
    caption="Performance of machine learning models for the joint datasets. Reported peak-weighted Root Mean Squared Error (pRMSE), Mean Absolute Scaled Error (MASE) and Coefficient of determination (R$^2$ score) are average and standard deviation values calculated using predictions from 10 different runs per model. Best scores per metric are marked in bold.",
    label="tab:ml_perf_both"
)

