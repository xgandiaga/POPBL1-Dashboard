import pandas as pd
import dash
from sklearn.ensemble import RandomForestRegressor
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import hdbscan

import plotly.express as px
import threading

data = pd.read_csv('work_data_prepro_all.csv')

event = threading.Event()
temp = data.copy()

app = dash.Dash(__name__)


app.layout = html.Div(children=[

    html.Div(
        [
            html.H3('Clustering interface', id='title'),
        ],
        className="pager"),

    html.Div(
        [
            # la parte del control
            html.Div(
                [html.Div([
                    html.Div([
                        html.Label('X-Axis variable'),
                        dcc.Dropdown(
                            id='variable1',
                            clearable=False,
                            value='event.duration',
                            options=[
                                {'label': 'Event duration',
                                    'value': 'event.duration'},
                                {'label': 'Network bytes',
                                    'value': 'network.bytes'},
                                {'label': 'Network packets',
                                    'value': 'network.packets'},
                            ],
                        ),
                    ], className="option"),

                    html.Div([
                        html.Label('Y-Axis variable'),
                        dcc.Dropdown(
                            id='variable2',
                            clearable=False,
                            value='network.bytes',
                            options=[
                                {'label': 'Event duration',
                                    'value': 'event.duration'},
                                {'label': 'Network bytes',
                                    'value': 'network.bytes'},
                                {'label': 'Network packets',
                                    'value': 'network.packets'},
                            ],
                        ),
                    ], className="option"),

                    html.Div([
                        html.Label('Algorithm'),
                        dcc.Dropdown(
                            id='algorythm',
                            clearable=False,
                            value='DBSCAN',
                            options=[
                                {'label': 'DBSCAN', 'value': 'DBSCAN'},
                                {'label': 'OPTICS', 'value': 'OPTICS'},
                                {'label': 'HDBSCAN', 'value': 'HDBSCAN'},
                            ],
                        ),
                    ], className="option"),

                    html.Div([
                        html.Label('Number of samples'),
                        dcc.Slider(
                            id='num_samples',
                            marks={i: '{}'.format(10 ** i)
                                   for i in range(len(data.index))},
                            max=len(data.index),
                            value=1000,
                            step=1,
                            updatemode='drag'
                        ),

                        html.Label(id='updatemode-output-container',
                                   style={'margin-top': 20}),
                    ], className="option"),

                    html.Div([
                        html.Label('Minimum samples for a cluster',
                                   style={'margin-top': 20}),
                        dcc.Input(
                            id='min_samples',
                            placeholder='Enter a value...',
                            type='number',
                            value=50,
                            min=1,
                            max=len(data.index),
                            step=1,
                            debounce=True),
                    ], id='min_samples_div', className="option"),

                    html.Div([
                        html.Label('Epsilon(Minimum distance between points)'),
                        dcc.Input(
                            id='epsilon',
                            placeholder='Enter a value...',
                            type='number',
                            value=0.3,
                            debounce=True),
                    ], id='epsilon_div', className="option"),

                    html.Div([
                        html.Label('Minimun cluster size'),
                        dcc.Input(
                            id='min_cluster_size',
                            placeholder='Enter a value...',
                            type='number',
                            value=10,
                            min=2,
                            step=1,
                            debounce=True),
                    ], className="option", id='min_cluster_size_div', style={'display': 'none'}),

                    html.Div([
                        html.Label('XI'),
                        dcc.Input(
                            id='xi',
                            placeholder='Enter a value...',
                            type='number',
                            value=0.05,
                            min=0,
                            max=1,
                            step=0.01,
                            debounce=True),
                    ], className="option", id='xi_div', style={'display': 'none'}),

                    html.Div([
                        html.Button('Reset', id='reset'),
                    ], className="option"),
                ], className="control_container"),
                ], className="control"),

            # la parte de los datos
            html.Div(
                [
                    dcc.Loading([
                        html.Div(
                            [
                                html.Div([
                                    dcc.Loading([
                                        dcc.Graph(
                                            id='clustering',
                                            figure={},
                                        ),
                                    ])], className="scatter"),

                                html.Div([
                                    dcc.Loading([
                                        dcc.Graph(
                                            id='clusterpie',
                                            figure={},
                                        ),
                                    ])], className="pie"),

                            ],
                            className="graph"),

                        html.Div(
                            [
                                html.Div([
                                    dcc.Loading([
                                        dcc.Graph(
                                            id='legit1',
                                            figure={},
                                        ),
                                    ])
                                ], className="comparison"),

                                html.Div([
                                    dcc.Loading([
                                        dcc.Graph(
                                            id='alltraffic1',
                                            figure={},
                                        ),
                                    ])
                                ], className="comparison"),

                                html.Div([
                                    dcc.Loading([
                                        dcc.Graph(
                                            id='legit2',
                                            figure={},
                                        ),
                                    ])
                                ], className="comparison"),

                                html.Div([
                                    dcc.Loading([
                                        dcc.Graph(
                                            id='alltraffic2',
                                            figure={},
                                        ),
                                    ])
                                ], className="comparisonLast"),

                            ], className="info"),
                    ])],
                className="data"),
        ],
        className="data_container"),
], className="container")


@app.callback([
    Output('variable1', component_property='value'),
    Output('variable2', component_property='value'),
    Output('algorythm', component_property='value'),
    Output('num_samples', component_property='value'),
    Output('min_samples', component_property='value'),
    Output('epsilon', component_property='value'),
    Output('min_cluster_size', component_property='value'),
    Output('xi', component_property='value'),
    Output('num_samples', component_property='value')],
    [Input('reset', 'n_clicks')])
def reset(n_clicks):
    return 'network.bytes', 'event.duration', 'DBSCAN', 100, 10, 0.5, 5, 0.05, 1000


@app.callback(
    [Output('legit1', 'figure'),
     Output('alltraffic1', 'figure'),
     Output('legit2', 'figure'),
     Output('alltraffic2', 'figure'), ],
    [Input('variable1', 'value'), Input(
        'variable2', 'value'), Input('num_samples', 'value')]
)
def legit(variable1, variable2, num_samples):

    data = pd.read_csv('work_data_prepro.csv', nrows=num_samples)
    data2 = pd.read_csv('work_data_prepro_all.csv', nrows=num_samples)

    trace1 = go.Histogram(x=data[variable1], nbinsx=20, histnorm='percent')
    trace2 = go.Histogram(x=data2[variable1], nbinsx=20, histnorm='percent')
    trace3 = go.Histogram(x=data[variable2], nbinsx=20, histnorm='percent')
    trace4 = go.Histogram(x=data2[variable2], nbinsx=20, histnorm='percent')

    figure1 = {
        'data': [trace1],
        'layout': go.Layout(title='Distribution of ' + str(variable1))}

    figure2 = {
        'data': [trace2],
        'layout': go.Layout(title='Distribution of all traffic for ' + str(variable1))}

    figure3 = {
        'data': [trace3],
        'layout': go.Layout(title='Distribution of ' + str(variable2))}

    figure4 = {
        'data': [trace4],
        'layout': go.Layout(title='Distribution of all traffic for ' + str(variable2))}

    return figure1, figure2, figure3, figure4


@app.callback([
    Output('epsilon_div', component_property='style'),
    Output('min_cluster_size_div', component_property='style'),
    Output('xi_div', component_property='style')],
    [Input('algorythm', 'value'), Input('reset', 'n_clicks')])
def hide_and_seek(value, n_clicks):
    if n_clicks is not None:
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    if value == 'DBSCAN':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    if value == 'OPTICS':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'block'}
    if value == 'HDBSCAN':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}


@app.callback(Output('updatemode-output-container', 'children'),
              [Input('num_samples', 'value')])
def display_value(value):
    return 'Linear Value: {}'.format(value)


@app.callback(
    [Output('clustering', 'figure'),
     Output('clusterpie', 'figure')],
    [Input('variable1', 'value'), Input('variable2', 'value'), Input('algorythm', 'value'), Input('num_samples', 'value'), Input('min_samples', 'value'),
     Input('epsilon', 'value'), Input('min_cluster_size', 'value'), Input('xi', 'value')]
)
def callback_variables(variable1, variable2, algorythm, num_samples, min_samples, epsilon, min_cluster_size, xi):

    data = pd.read_csv('work_data_prepro.csv', nrows=num_samples)

    traces = []
    temp = data.copy()
    X = data.copy()
    X = X.astype("int64")

    # Scaling the data to bring all the attributes to a comparable level
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if algorythm == 'DBSCAN':
        # eps = radio , min_samples = muestras minimas
        db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_noise_ = list(labels).count(-1)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

    if algorythm == 'OPTICS':

        # Building the OPTICS Clustering model
        optics_model = OPTICS(min_samples=min_samples, xi=xi,
                              min_cluster_size=min_cluster_size)
        optics_model = optics_model.fit(X)

        # Training the model
        labels = optics_model.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

    if algorythm == 'HDBSCAN':
        X_normal = normalize(X)
        X_df = pd.DataFrame(X_normal)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, gen_min_span_tree=True)
        clusterer = clusterer.fit(X_df)
        labels = clusterer.labels_

    temp['cluster'] = labels

    for i in temp.cluster.unique():
        data_by_cluster = temp[temp['cluster'] == i]
        traces.append(dict(
            x=data_by_cluster[variable1],
            y=data_by_cluster[variable2],
            text=data_by_cluster['cluster'],
            mode='markers',
            opacity=0.8,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i
        )),

        figure = {
            'data': traces,
            'layout': dict(
                xaxis={'title': str(variable1)},
                yaxis={'title': str(variable2)},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
            )
        }

    values1 = []
    labels1 = []
    traces1 = []
    for i in temp.cluster.unique():
        # print('Cluster '+str(i))
        # print(temp[temp['cluster']==i]['cluster'].count())
        values1.append((temp[temp['cluster'] == i]['cluster']).count())
        labels1.append('Cluster '+str(i))
        # print(values)
        # print(labels)

    traces1.append(dict(
        values=values1,
        labels=labels1,
        type='pie'
    ))

    figure2 = {
        'data': traces1,
        'layout': dict(
            showlegend=True,
            legend={'x': 0, 'y': 1},
            hovermode='closest',
        )
    }

    return figure, figure2


if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True)
