import os
import re
import math
import argparse
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px


# ---------------------------
# Argument parsing for CSV dir
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Arithmetic Intensity vs. Performance Dashboard"
    )
    parser.add_argument(
        "--csv-dir",
        default=os.path.join("..", "benchmarks"),
        help="Directory to search for CSV files (default: ../benchmarks)",
    )
    return parser.parse_args()


args = parse_args()
CSV_DIR = args.csv_dir

# ---------------------------
# Initialize the Dash app
# ---------------------------
app = Dash(__name__)


# ---------------------------
# Discover CSV files in CSV_DIR
# ---------------------------
def list_csv_files():
    if not os.path.isdir(CSV_DIR):
        return []
    files = [
        f
        for f in os.listdir(CSV_DIR)
        if os.path.isfile(os.path.join(CSV_DIR, f)) and f.lower().endswith(".csv")
    ]
    return sorted(files)


# ---------------------------
# --- UI Controls ---
# ---------------------------
# File selection
file_dropdown = dcc.Dropdown(
    id="file-selector",
    options=[{"label": f, "value": f} for f in list_csv_files()],
    multi=True,
    placeholder="Select one or two CSV files",
    style={"width": "100%"},
)

# Number of Compute Units
cu_input = dcc.Input(
    id="num-cu",
    type="number",
    min=1,
    step=1,
    value=304,  # default; set to match your GPU
    debounce=True,
    style={"width": "120px"},
)

# Toggle button for color mode
color_toggle_btn = html.Button(
    "Color: file", id="color-toggle", n_clicks=0, style={"marginLeft": "12px"}
)

# Periodic refresh
auto_interval = dcc.Interval(
    id="interval", interval=60 * 1000, n_intervals=0  # 60 seconds
)

# Graph + metrics
graph = dcc.Graph(id="ai-vs-performance")
metrics_div = html.Div(id="metrics-table")

# ---------------------------
# App layout
# ---------------------------
app.layout = html.Div(
    [
        html.H2(f"Dashboard ({CSV_DIR})"),
        html.Div(
            [
                html.Div(file_dropdown, style={"flex": "1"}),
                html.Div(
                    [
                        html.Label("# Compute Units (CU)"),
                        cu_input,
                        color_toggle_btn,
                    ],
                    style={
                        "display": "flex",
                        "gap": "8px",
                        "alignItems": "center",
                        "marginLeft": "16px",
                        "whiteSpace": "nowrap",
                    },
                ),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "10px"},
        ),
        auto_interval,
        graph,
        html.H3("Aggregate Metrics Comparison"),
        metrics_div,
    ],
    style={"margin": "20px"},
)


# ---------------------------
# Callback to refresh dropdown options
# ---------------------------
@app.callback(Output("file-selector", "options"), Input("interval", "n_intervals"))
def refresh_file_options(_):
    return [{"label": f, "value": f} for f in list_csv_files()]


# ---------------------------
# --- Helpers ---
# ---------------------------
def extract_tile_dims(macro_tile_series: pd.Series) -> pd.DataFrame:
    """
    Parse tile_m and tile_n from macro_tile strings.
    Accepts '128x128', '128X64', 'tile_128x64', '128_64', etc.
    Returns DataFrame with float columns tile_m, tile_n (NaN if parse fails).
    """
    extracted = macro_tile_series.astype(str).str.extract(
        r"(\d+)[^\d]+(\d+)", expand=True
    )
    extracted.columns = ["tile_m", "tile_n"]
    extracted = extracted.apply(pd.to_numeric, errors="coerce")
    return extracted


# ---------------------------
# Compute metrics and figure
# ---------------------------
def compute_metrics(selected_files, cu, color_mode="file"):
    dfs = []
    for fname in selected_files or []:
        fullpath = os.path.join(CSV_DIR, fname)
        if not os.path.exists(fullpath):
            continue

        df = pd.read_csv(fullpath)

        # Basic requirements
        required_cols = {
            "mnk",
            "bytes",
            "tritonblas_gflops",
            "macro_tile",
            "m",
            "n",
            "k",
        }
        if not required_cols.issubset(df.columns):
            # Skip files that don't have the expected schema
            continue

        # Arithmetic Intensity
        df["ai"] = (df["mnk"] * 2) / df["bytes"]

        # Parse tile dimensions from macro_tile
        tiles = extract_tile_dims(df["macro_tile"])
        df["tile_m"] = tiles["tile_m"]
        df["tile_n"] = tiles["tile_n"]

        # --- Last-wave occupancy (computed) ---
        # ((ceil(M/tile_m) * ceil(N/tile_n)) % CU) / CU
        with np.errstate(divide="ignore", invalid="ignore"):
            ceil_m = np.ceil(df["m"] / df["tile_m"])
            ceil_n = np.ceil(df["n"] / df["tile_n"])
            waves = ceil_m * ceil_n

            cu_val = float(cu) if (cu is not None and cu > 0) else np.nan
            last_wave = np.mod(waves, cu_val) / cu_val

            # Ensure NaN when tile dims missing / infinite
            last_wave = np.where(
                np.isfinite(ceil_m) & np.isfinite(ceil_n) & np.isfinite(cu_val),
                last_wave,
                np.nan,
            )

        df["last_wave_occupancy"] = last_wave
        df["file"] = fname

        dfs.append(
            df[
                [
                    "ai",
                    "tritonblas_gflops",
                    "macro_tile",
                    "m",
                    "n",
                    "k",
                    "tile_m",
                    "tile_n",
                    "last_wave_occupancy",
                    "file",
                ]
            ]
        )

    if not dfs:
        return None, None, False  # no data

    all_df = pd.concat(dfs, ignore_index=True)

    # Decide color field
    use_occupancy = color_mode == "occupancy"
    color_field = "last_wave_occupancy" if use_occupancy else "file"

    # Build figure
    fig = px.scatter(
        all_df,
        x="ai",
        y="tritonblas_gflops",
        color=color_field,
        hover_data={
            "macro_tile": True,
            "m": True,
            "n": True,
            "k": True,
            "tile_m": True,
            "tile_n": True,
            "last_wave_occupancy": ":.2%",
            "ai": ":.2f",
            "tritonblas_gflops": ":.2f",
            "file": False if use_occupancy else True,
        },
        title="Arithmetic Intensity vs. Performance Comparison",
        labels={
            "ai": "Arithmetic Intensity (FLOPs/Byte)",
            "tritonblas_gflops": "Performance (GFLOPS)",
            "last_wave_occupancy": "Last-Wave Occupancy",
        },
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(transition_duration=500)

    # If coloring by occupancy but everything is NaN, add a gentle note
    if use_occupancy and not np.isfinite(all_df["last_wave_occupancy"]).any():
        fig.add_annotation(
            text="⚠️ Computed last-wave occupancy is NaN for all rows "
            "(check macro_tile parsing and CU value).",
            xref="paper",
            yref="paper",
            x=0,
            y=1.12,
            showarrow=False,
            font=dict(size=12),
        )

    # Aggregate metrics (include occupancy avg)
    metrics = (
        all_df.groupby("file")
        .agg(
            avg_gflops=("tritonblas_gflops", "mean"),
            max_gflops=("tritonblas_gflops", "max"),
            avg_ai=("ai", "mean"),
            max_ai=("ai", "max"),
            avg_last_wave_occ=("last_wave_occupancy", "mean"),
        )
        .reset_index()
    )

    # Build table
    header = [
        html.Th(col)
        for col in [
            "File",
            "Avg GFLOPS",
            "Max GFLOPS",
            "Avg AI",
            "Max AI",
            "Avg Last-Wave Occ",
        ]
    ]
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(row["file"]),
                    html.Td(f"{row['avg_gflops']:.2f}"),
                    html.Td(f"{row['max_gflops']:.2f}"),
                    html.Td(f"{row['avg_ai']:.2f}"),
                    html.Td(f"{row['max_ai']:.2f}"),
                    html.Td(
                        "—"
                        if pd.isna(row["avg_last_wave_occ"])
                        else f"{row['avg_last_wave_occ']:.2%}"
                    ),
                ]
            )
        )
    table = html.Table(
        [html.Thead(html.Tr(header)), html.Tbody(rows)],
        style={
            "width": "100%",
            "border": "1px solid black",
            "borderCollapse": "collapse",
        },
    )

    return fig, table, True


# ---------------------------
# Main + callbacks
# ---------------------------
def update_dashboard(selected_files, _n_intervals, cu, color_mode="file"):
    fig, table, _ = compute_metrics(selected_files, cu, color_mode=color_mode)
    if fig is None:
        empty = px.scatter(title="No file(s) selected or found")
        return empty, []
    return fig, table


@app.callback(
    Output("ai-vs-performance", "figure"),
    Output("metrics-table", "children"),
    Output("color-toggle", "children"),  # update button label
    Input("file-selector", "value"),
    Input("interval", "n_intervals"),
    Input("num-cu", "value"),
    Input("color-toggle", "n_clicks"),
)
def on_update(selected_files, n_intervals, cu, n_clicks):
    n = n_clicks or 0
    color_mode = "occupancy" if (n % 2 == 1) else "file"
    fig, table, _ = compute_metrics(selected_files, cu, color_mode=color_mode)

    if fig is None:
        empty = px.scatter(title="No file(s) selected or found")
        btn_label = (
            "Color: last_wave_occupancy" if color_mode == "occupancy" else "Color: file"
        )
        return empty, [], btn_label

    btn_label = (
        "Color: last_wave_occupancy" if color_mode == "occupancy" else "Color: file"
    )
    return fig, table, btn_label


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
