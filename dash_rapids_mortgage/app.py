# -*- coding: utf-8 -*-
import time
import os
import json
import gzip
import shutil
import requests
import numpy as np
import pyarrow as pa

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
from plotly.colors import sequential

from dask import delayed
from distributed import Client

import cudf
import cupy

from dash_rapids_mortgage.utils import (
    scheduler_url, float_columns, compute_bounds, get_dataset
)

print("CUDA_VISIBLE_DEVICES=" + str(os.environ.get("CUDA_VISIBLE_DEVICES", None)))

# Disable cupy memory pool so that cupy immediately releases GPU memory
cupy.cuda.set_allocator(None)


# Global initialization
client = None
bounds = None

def init_client():
    """
    This function must be called before any of the functions that require a client.
    """
    global client, bounds
    # Init client
    print(f"Connecting to cluster at {scheduler_url} ... ", end='')
    client = Client(scheduler_url)
    print("done")

    # Perform global operations that are expensive and require the client
    c_df_d = get_dataset(client, 'c_df_d')
    bounds = delayed(compute_bounds)(c_df_d, float_columns).compute()


# Colors
bgcolor = "#191a1a"  # mapbox dark map land color
text_color = "#cfd8dc"  # Material blue-grey 100
mapbox_land_color = "#343332"

# Figure template
row_heights = [150, 440, 200]
template = {
    'layout': {
        'paper_bgcolor': bgcolor,
        'plot_bgcolor': bgcolor,
        'font': {'color': text_color},
        "margin": {"r": 0, "t": 30, "l": 0, "b": 20},
        'bargap': 0.05,
        'xaxis': {'showgrid': False, 'automargin': True},
        'yaxis': {'showgrid': True, 'automargin': True,
                  'gridwidth': 0.5, 'gridcolor': mapbox_land_color},
    }
}


# Load mapbox token from environment variable or file
token = os.getenv('MAPBOX_TOKEN')
if not token:
    token = open(".mapbox_token").read()

# geojson URL
zip3_url = 'https://raw.githubusercontent.com/rapidsai/cuxfilter/GTC-2018-mortgage-visualization/javascript/demos/GTC%20demo/src/data/zip3-ms-rhs-lessprops.json'

# Download geojson so we can read all of the zip3 codes we have
response = requests.get(zip3_url)
zip3_json = json.loads(response.content.decode())
valid_zip3s = {int(f['properties']['ZIP3']) for f in zip3_json["features"]}


# Names of float columns
column_labels = {
    'delinquency_12_prediction': 'Risk Score',
    'borrower_credit_score': 'Borrower Credit Score',
    'current_actual_upb': 'Unpaid Balance',
    'dti': 'Debt to Income Ratio',
}


# Build Dash app and initial layout
def blank_fig(height):
    """
    Build blank figure with the requested height
    Args:
        height: height of blank figure in pixels
    Returns:
        Figure dict
    """
    return {
        'data': [],
        'layout': {
            'height': height,
            'template': template,
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
        }
    }


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.Div([
        html.H1(children=[
            'Mortgage Risk Model',
            html.A(
                html.Img(
                    src="assets/dash-logo.png",
                    style={'float': 'right', 'height': '50px', 'margin-right': '2%'}
                ), href="https://dash.plot.ly/"),
        ], style={'text-align': 'left'}),
    ]),
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.H4([
                    "Selected Mortgages",
                ], className="container_title"),
                dcc.Loading(
                    dcc.Graph(
                        id='indicator-graph',
                        figure=blank_fig(row_heights[0]),
                        config={'displayModeBar': False},
                    ),
                    style={'height': row_heights[0]},
                ),
                html.Div(children=[
                    html.Button(
                        "Clear All Selections", id='clear-all', className='reset-button'
                    ),
                ]),
            ], className='six columns pretty_container', id="indicator-div"),
            html.Div(children=[
                html.H4([
                    "Configuration",
                ], className="container_title"),
                html.Table([
                    html.Col(style={'width': '100px'}),
                    html.Col(),
                    html.Col(),
                    html.Tr([
                        html.Td(
                            html.Div("GPU"), className="config-label"
                        ),
                        html.Td(daq.DarkThemeProvider(daq.BooleanSwitch(
                            on=True,
                            color='#00cc96',
                            id='gpu-toggle',
                        ))),
                    ]),
                    html.Tr([
                        html.Td(html.Div("Color by"), className="config-label"),
                        html.Td(dcc.Dropdown(
                            id='aggregate-dropdown',
                            options=[
                                {'label': agg, 'value': agg}
                                for agg in ['count', 'mean', 'min', 'max']
                            ],
                            value='count',
                            searchable=False,
                            clearable=False,
                        )),
                        html.Td(dcc.Dropdown(
                            id='aggregate-col-dropdown',
                            value='delinquency_12_prediction',
                            searchable=False,
                            clearable=False,
                        )),
                    ]),
                    html.Tr([
                        html.Td(html.Div("Colormap"), className="config-label"),
                        html.Td(dcc.Dropdown(
                            id='colorscale-dropdown',
                            options=[
                                {'label': cs, 'value': cs}
                                for cs in ['Viridis', 'Cividis', 'Inferno', 'Magma', 'Plasma']
                            ],
                            value='Viridis',
                            searchable=False,
                            clearable=False,
                        )),
                        html.Td(dcc.Dropdown(
                            id='colorscale-transform-dropdown',
                            options=[{'label': t, 'value': t}
                                     for t in ['linear', 'sqrt', 'cbrt', 'log']],
                            value='linear',
                            searchable=False,
                            clearable=False,
                        )),
                    ]),
                    html.Tr([
                        html.Td(html.Div("Bin Count"), className="config-label"),
                        html.Td(dcc.Slider(
                            id='nbins-slider',
                            min=10,
                            max=40,
                            step=5,
                            value=20,
                            marks={m: str(m) for m in range(10, 41, 5)},
                            included=False,
                        ), colSpan=2),
                    ])
                ], style={'width': '100%', 'height': f'{row_heights[0] + 40}px'}),
            ], className='six columns pretty_container', id="config-div"),
        ]),
        html.Div(children=[
            html.H4([
                "Zip Codes",
            ], className="container_title"),
            dcc.Graph(
                id='map-graph',
                figure=blank_fig(row_heights[1]),
            ),
            html.Button("Clear Selection", id='reset-map', className='reset-button'),
        ], className='twelve columns pretty_container',
            style={
                'width': '98%',
                'margin-right': '0',
            },
            id="map-div"
        ),
        html.Div(children=[
            html.Div(
                children=[
                    html.H4([
                        "Risk SCORE",
                    ], className="container_title"),
                    dcc.Graph(
                        id='delinquency-histogram',
                        config={'displayModeBar': False},
                        figure=blank_fig(row_heights[2]),
                        animate=True
                    ),
                    html.Button(
                        "Clear Selection", id='clear-delinquency', className='reset-button'
                    ),
                ],
                className='six columns pretty_container', id="delinquency-div"
            ),
            html.Div(
                children=[
                    html.H4([
                        "Borrower Credit Score",
                    ], className="container_title"),
                    dcc.Graph(
                        id='credit-histogram',
                        config={'displayModeBar': False},
                        figure=blank_fig(row_heights[2]),
                        animate=True
                    ),
                    html.Button(
                        "Clear Selection", id='clear-credit', className='reset-button'
                    ),
                ],
                className='six columns pretty_container', id="credit-div"
            ),
        ]),
        html.Div(children=[
            html.Div(
                children=[
                    html.H4([
                        "Unpaid Balance",
                    ], className="container_title"),
                    dcc.Graph(
                        id='upb-histogram',
                        figure=blank_fig(row_heights[2]),
                        config={'displayModeBar': False},
                        animate=True
                    ),
                    html.Button(
                        "Clear Selection", id='clear-upb', className='reset-button'
                    ),
                ],
                className='six columns pretty_container', id="upb-div"
            ),
            html.Div(
                children=[
                    html.H4([
                        "Debt to Income Ratio",
                    ], className="container_title"),
                    dcc.Graph(
                        id='dti-histogram',
                        figure=blank_fig(row_heights[2]),
                        config={'displayModeBar': False},
                        animate=True
                    ),
                    html.Button(
                        "Clear Selection", id='clear-dti', className='reset-button'
                    ),
                ],
                className='six columns pretty_container', id="dti-div"
            ),
        ]),
    ]),
    html.Div(
        [
            html.H4('Acknowledgements', style={"margin-top": "0"}),
            dcc.Markdown('''\
 - Dashboard written in Python using the [Dash](https://dash.plot.ly/) web framework.
 - GPU accelerated provided by the [cudf](https://github.com/rapidsai/cudf) and
 [cupy](https://cupy.chainer.org/) libraries.
 - Base map layer is the ["dark" map style](https://www.mapbox.com/maps/light-dark/)
 provided by [mapbox](https://www.mapbox.com/).
'''),
        ],
        style={
            'width': '98%',
            'margin-right': '0',
            'padding': '10px',
        },
        className='twelve columns pretty_container',
    ),
])

# Register callbacks
@app.callback(
    [Output('aggregate-col-dropdown', 'options'),
     Output('aggregate-col-dropdown', 'disabled')],
    [Input('aggregate-dropdown', 'value')]
)
def update_agg_col_dropdown(agg):
    if agg == 'count':
        options = [{'label': 'NA',
                    'value': 'NA'}]
        disabled = True
    else:
        options = [{'label': v, 'value': k} for k, v in column_labels.items()]
        disabled = False
    return options, disabled


# Clear/reset button callbacks
@app.callback(
    Output('map-graph', 'selectedData'),
    [Input('reset-map', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_map(*args):
    return None


@app.callback(
    Output('dti-histogram', 'selectedData'),
    [Input('clear-dti', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_dti_hist_selections(*args):
    return None


@app.callback(
    Output('credit-histogram', 'selectedData'),
    [Input('clear-credit', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_credit_hist_selections(*args):
    return None


@app.callback(
    Output('upb-histogram', 'selectedData'),
    [Input('clear-upb', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_upb_hist_selection(*args):
    return None


@app.callback(
    Output('delinquency-histogram', 'selectedData'),
    [Input('clear-delinquency', 'n_clicks'), Input('clear-all', 'n_clicks')]
)
def clear_delinquency_hist_selection(*args):
    return None


@app.callback(
    [Output('indicator-graph', 'figure'), Output('map-graph', 'figure'),
     Output('dti-histogram', 'figure'), Output('credit-histogram', 'figure'),
     Output('upb-histogram', 'figure'), Output('delinquency-histogram', 'figure')],
    [Input('map-graph', 'selectedData'),
     Input('dti-histogram', 'selectedData'), Input('credit-histogram', 'selectedData'),
     Input('upb-histogram', 'selectedData'), Input('delinquency-histogram', 'selectedData'),
     Input('aggregate-dropdown', 'value'), Input('aggregate-col-dropdown', 'value'),
     Input('colorscale-dropdown', 'value'), Input('colorscale-transform-dropdown', 'value'),
     Input('nbins-slider', 'value'), Input('gpu-toggle', 'on')
     ]
)
def update_plots(
        selected_map, selected_dti, selected_credit, selected_upb, selected_delinquency,
        aggregate, aggregate_column, colorscale_name, transform, nbins, gpu_enabled
):
    t0 = time.time()

    # Get delayed dataset from client
    if gpu_enabled:
        df_d = get_dataset(client, 'c_df_d')
    else:
        df_d = get_dataset(client, 'pd_df_d')

    figures_d = delayed(build_updated_figures)(
        df_d, selected_map, selected_dti, selected_credit, selected_upb,
        selected_delinquency, aggregate, aggregate_column, colorscale_name,
        transform, nbins, bounds)

    figures = figures_d.compute()

    (choropleth, credit_histogram, delinquency_histogram,
     dti_histogram, n_selected_indicator, upb_histogram) = figures

    print(f"Update time: {time.time() - t0}")
    return (
        n_selected_indicator, choropleth, dti_histogram, credit_histogram,
        upb_histogram, delinquency_histogram
    )


# Query string helpers
def bar_selection_to_query(selection, column, bounds, nbins):
    """
    Compute pandas query expression string for selection callback data

    Args:
        selection: selectedData dictionary from Dash callback on a bar trace
        column: Name of the column that the selected bar chart is based on
        bounds: Dictionary from columns to (min, max) tuples
        nbins: Number of histogram bins

    Returns:
        String containing a query expression compatible with DataFrame.query. This
        expression will filter the input DataFrame to contain only those rows that
        are contained in the selection.
    """
    point_inds = [p['pointIndex'] for p in selection['points']]
    bin_edges = np.linspace(*bounds[column], nbins)
    xmin = bin_edges[min(point_inds)]
    xmax = bin_edges[max(point_inds) + 1]
    xmin_op = "<="
    xmax_op = "<=" if xmax == bin_edges[-1] else "<"
    return f"{xmin} {xmin_op} {column} and {column} {xmax_op} {xmax}"


def build_query(selections, exclude=None):
    """
    Build pandas query expression string for cross-filtered plot

    Args:
        selections: Dictionary from column name to query expression
        exclude: If specified, column to exclude from combined expression

    Returns:
        String containing a query expression compatible with DataFrame.query.
    """
    other_selected = {sel for c, sel in selections.items() if c != exclude}
    if other_selected:
        return ' and '.join(other_selected)
    else:
        return None


# Plot functions
def build_colorscale(colorscale_name, transform):
    """
    Build plotly colorscale

    Args:
        colorscale_name: Name of a colorscale from the plotly.colors.sequential module
        transform: Transform to apply to colors scale. One of 'linear', 'sqrt', 'cbrt',
        or 'log'

    Returns:
        Plotly color scale list
    """
    colors = getattr(sequential, colorscale_name)
    if transform == "linear":
        scale_values = np.linspace(0, 1, len(colors))
    elif transform == "sqrt":
        scale_values = np.linspace(0, 1, len(colors)) ** 2
    elif transform == "cbrt":
        scale_values = np.linspace(0, 1, len(colors)) ** 3
    elif transform == "log":
        scale_values = (10 ** np.linspace(0, 1, len(colors)) - 1) / 9
    else:
        raise ValueError("Unexpected colorscale transform")
    return [(v, clr) for v, clr in zip(scale_values, colors)]


def build_choropleth(
        df, aggregate, aggregate_column, colorscale_name, colorscale_transform, selected_zips
):
    """
    Build choropleth figure

    Args:
        df: pandas or cudf DataFrame
        aggregate: Aggregate operation (count, mean, etc.)
        aggregate_column: Column to perform aggregate on. Ignored for 'count' aggregate
        colorscale_name: Name of plotly colorscale
        colorscale_transform: Colorscale transformation
        clear_selection: If true, clear choropleth selection. Otherwise leave
            selection unchanged

    Returns:
        Choropleth figure dictionary
    """
    # Perform aggregation
    if aggregate == "count":
        zip_aggregates = df.groupby('zip').zip.count()
    else:
        grouper = df.groupby('zip')[aggregate_column]
        zip_aggregates = getattr(grouper, aggregate)()

    if isinstance(df, cudf.DataFrame):
        zip_aggregates = zip_aggregates.to_pandas()

    # Filter down to zip codes that we have geojson for
    zip_aggregates = zip_aggregates[zip_aggregates.index.isin(valid_zip3s)]

    # Build zero-padded zip3 strings
    zip_strs = zip_aggregates.index.astype(str).str.zfill(3)

    # Build colorscale
    colorscale = build_colorscale(colorscale_name, colorscale_transform)

    # Compute selected points
    if selected_zips is None:
        selectedpoints = None
    else:
        selected_mask = zip_aggregates.index.isin(selected_zips)
        selectedpoints = np.nonzero(selected_mask)[0]

    if aggregate == "count":
        colorbar_title = aggregate
    else:
        column_label = column_labels[aggregate_column]
        colorbar_title = f"{aggregate}({column_label})"

    # Build Figure
    fig = {
        "data": [{
            "type": "choroplethmapbox",
            "geojson": zip3_url,
            "featureidkey": "properties.ZIP3",
            "locations": zip_strs,
            "z": zip_aggregates.values,
            "colorscale": colorscale,
            "selectedpoints": selectedpoints,
            "colorbar": {"title": {
                "text": colorbar_title, "side": "right", "font": {"size": 14}
            }}
        }],
        "layout": {
            "mapbox": {
                "style": "dark",
                "accesstoken": token,
                "zoom": 3,
                "center": {"lat": 37.0902, "lon": -95.7129},
                'pitch': 0,
                'bearing': 0,
            },
            "uirevision": True,
            "margin": {"r": 140, "t": 26, "l": 0, "b": 0},
            'template': template,
        }
    }

    return fig


def build_histogram(df, column, nbins, bounds, selections, query_cache):
    """
    Build histogram figure

    Args:
        df: pandas or cudf DataFrame
        column: Column name to build histogram from
        nbins: Number of histogram bins
        bounds: Dictionary from columns to (min, max) tuples
        selections: Dictionary from column names to query expressions
        query_cache: Dict from query expression to filtered DataFrames

    Returns:
        Histogram figure dictionary
    """
    query = build_query(selections, column)
    if query in query_cache:
        df = query_cache[query]
    elif query:
        df = df.query(query)
        query_cache[query] = df

    if isinstance(df, cudf.DataFrame):
        bin_edges = cupy.linspace(*bounds[column], nbins)
        counts = cupy.asnumpy(cupy.histogram(df[column], bin_edges)[0])
        bin_edges = cupy.asnumpy(bin_edges)
    else:
        bin_edges = np.linspace(*bounds[column], nbins)
        counts = np.histogram(df[column], bin_edges)[0]

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    fig = {
        'data': [{
            'type': 'bar', 'x': centers, 'y': counts,
            'marker': {'color': text_color}
        }],
        'layout': {
            'yaxis': {
                'type': 'log',
                # 'range': [0, None],
                'range': [0, 8],  # Up to 100M
                'title': {
                    'text': "Count"
                }
            },
            'selectdirection': 'h',
            'dragmode': 'select',
            'template': template,
            'uirevision': True,
        }
    }
    if column not in selections:
        fig['data'][0]['selectedpoints'] = False

    return fig


def build_updated_figures(
        df, selected_map, selected_dti, selected_credit, selected_upb,
        selected_delinquency, aggregate, aggregate_column,
        colorscale_name, colorscale_transform, nbins, bounds
):
    """
    Build all figures for dashboard

    Args:
        df: pandas or cudf DataFrame
        selected_map: selectedData for choropleth figure
        selected_dti: selectedData for dti histogram
        selected_credit: selectedData for credit history histogram
        selected_upb: selectedData for unpaid balance histogram
        selected_delinquency: selectedData for delinquency histogram
        aggregate: Aggregate operation for choropleth (count, mean, etc.)
        aggregate_column: Aggregate column for choropleth
        colorscale_name: Colorscale name from plotly.colors.sequential
        colorscale_transform: Colorscale transformation ('linear', 'sqrt', 'cbrt', 'log')
        nbins: Number of histogram bins
        bounds: Dictionary from columns to (min, max) tuples

    Returns:
        tuple of figures in the following order
        (choropleth, credit_histogram, delinquency_histogram,
            dti_histogram, n_selected_indicator, upb_histogram)
    """
    selected = {
        col: bar_selection_to_query(sel, col, bounds, nbins)
        for col, sel in zip([
            'dti', 'borrower_credit_score',
            'current_actual_upb', 'delinquency_12_prediction'
        ], [
            selected_dti, selected_credit, selected_upb, selected_delinquency
        ]) if sel and sel.get('points', [])
    }

    array_module = cupy if isinstance(df, cudf.DataFrame) else np

    all_hists_query = build_query(selected)

    if selected_map:
        selected_zips = array_module.array(
            [int(p['location']) for p in selected_map['points']])
    else:
        selected_zips = None

    if selected_zips is not None:
        zips_array = array_module.asarray(df['zip'])

        # Perform isin in fixed length chunks to limit memory usage
        isin_mask = array_module.zeros(len(zips_array), dtype=np.bool)
        stride = 32
        for i in range(0, len(selected_zips), stride):
            zips_chunk = selected_zips[i:i+stride]
            isin_mask |= array_module.isin(zips_array, zips_chunk)
        df_map = df[isin_mask]
    else:
        df_map = df

    choropleth = build_choropleth(
        df.query(all_hists_query) if all_hists_query else df, aggregate,
        aggregate_column, colorscale_name, colorscale_transform, selected_zips)

    # Build indicator figure
    n_selected_indicator = {
        'data': [{
            'type': 'indicator',
            'value': len(
                df_map.query(all_hists_query) if all_hists_query else df_map
            ),
            'number': {
                'font': {
                    'color': text_color
                }
            }
        }],
        'layout': {
            'template': template,
            'height': row_heights[0],
            'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10}
        }
    }
    query_cache = {}
    delinquency_histogram = build_histogram(
        df_map, 'delinquency_12_prediction', nbins, bounds, selected, query_cache,
    )

    credit_histogram = build_histogram(
        df_map, 'borrower_credit_score', nbins, bounds, selected, query_cache
    )

    upb_histogram = build_histogram(
        df_map, 'current_actual_upb', nbins, bounds, selected, query_cache
    )

    dti_histogram = build_histogram(
        df_map, 'dti', nbins, bounds, selected, query_cache
    )

    return (choropleth, credit_histogram, delinquency_histogram,
            dti_histogram, n_selected_indicator, upb_histogram)


# gunicorn entry point
def get_server():
    init_client()
    return app.server


if __name__ == '__main__':
    # development entry point
    init_client()
    app.run_server(debug=True)
