import os
import argparse
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Argument parsing for CSV directory
def parse_args():
    parser = argparse.ArgumentParser(
        description='Arithmetic Intensity vs. Performance Dashboard')
    parser.add_argument(
        '--csv-dir',
        default=os.path.join('..', 'benchmarks'),
        help='Directory to search for CSV files (default: ../benchmarks)'
    )
    return parser.parse_args()

args = parse_args()
CSV_DIR = args.csv_dir

# Initialize the Dash app
app = Dash(__name__)

# Discover CSV files in CSV_DIR
def list_csv_files():
    if not os.path.isdir(CSV_DIR):
        return []
    files = [f for f in os.listdir(CSV_DIR)
             if os.path.isfile(os.path.join(CSV_DIR, f))
             and f.lower().endswith('.csv')]
    return sorted(files)

# Dropdown for file selection
file_dropdown = dcc.Dropdown(
    id='file-selector',
    options=[{'label': f, 'value': f} for f in list_csv_files()],
    multi=True,
    placeholder='Select one or two CSV files',
    style={'width': '100%'}
)

# Interval for periodic updates
auto_interval = dcc.Interval(
    id='interval',
    interval=60 * 1000,  # 60 seconds
    n_intervals=0
)

# Graph and metrics elements
graph = dcc.Graph(id='ai-vs-performance')
metrics_div = html.Div(id='metrics-table')

# App layout
app.layout = html.Div([
    html.H2(f'Dashboard ({CSV_DIR})'),
    file_dropdown,
    auto_interval,
    graph,
    html.H3('Aggregate Metrics Comparison'),
    metrics_div
], style={'margin': '20px'})

# Callback to refresh dropdown options
@app.callback(
    Output('file-selector', 'options'),
    Input('interval', 'n_intervals')
)
def refresh_file_options(n_intervals):
    return [{'label': f, 'value': f} for f in list_csv_files()]

# Compute metrics and figure
def compute_metrics(selected_files):
    dfs = []
    for fname in selected_files or []:
        fullpath = os.path.join(CSV_DIR, fname)
        if os.path.exists(fullpath):
            df = pd.read_csv(fullpath)
            df['ai'] = (df['mnk'] * 2) / df['bytes']
            df['file'] = fname
            dfs.append(df[['ai', 'tritonblas_gflops', 'macro_tile', 'm', 'n', 'k', 'file']])
    if not dfs:
        return None, None
    all_df = pd.concat(dfs, ignore_index=True)

    # Plot
    fig = px.scatter(
        all_df,
        x='ai',
        y='tritonblas_gflops',
        color='file',
        hover_data=['macro_tile', 'm', 'n', 'k'],
        title='Arithmetic Intensity vs. Performance Comparison',
        labels={'ai': 'Arithmetic Intensity (FLOPs/Byte)',
                'tritonblas_gflops': 'Performance (GFLOPS)'}
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(transition_duration=500)

    # Aggregate metrics
    metrics = all_df.groupby('file').agg(
        avg_gflops=('tritonblas_gflops', 'mean'),
        max_gflops=('tritonblas_gflops', 'max'),
        avg_ai=('ai', 'mean'),
        max_ai=('ai', 'max')
    ).reset_index()

    # Build table
    header = [html.Th(col) for col in ['File', 'Avg GFLOPS', 'Max GFLOPS', 'Avg AI', 'Max AI']]
    rows = []
    for _, row in metrics.iterrows():
        rows.append(html.Tr([
            html.Td(row['file']),
            html.Td(f"{row['avg_gflops']:.2f}"),
            html.Td(f"{row['max_gflops']:.2f}"),
            html.Td(f"{row['avg_ai']:.2f}"),
            html.Td(f"{row['max_ai']:.2f}")
        ]))
    table = html.Table([
        html.Thead(html.Tr(header)),
        html.Tbody(rows)
    ], style={'width': '100%', 'border': '1px solid black', 'borderCollapse': 'collapse'})

    return fig, table

# Main callback
def update_dashboard(selected_files, n_intervals):
    fig, table = compute_metrics(selected_files)
    if fig is None:
        empty = px.scatter(title="No file(s) selected or found")
        return empty, []
    return fig, table

@app.callback(
    Output('ai-vs-performance', 'figure'),
    Output('metrics-table', 'children'),
    Input('file-selector', 'value'),
    Input('interval', 'n_intervals')
)
def on_update(selected_files, n_intervals):
    return update_dashboard(selected_files, n_intervals)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
