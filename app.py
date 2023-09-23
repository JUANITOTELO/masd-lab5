# Dash app that receives a 2 by 2 matrix of numbers and show it in a latex matrix form.

from dash import html, dcc, Output, Input, Dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_latex as dl
from dash.dependencies import Input, Output
import numpy as np
import plotly.figure_factory as ff


def dY_dt(a,b,X,Y):
    return a*X+b*Y

def dX_dt(c,d,X,Y):
    return c*X+d*Y

def generate_plot(A, name):
    # check if image exists
    # if os.pth.isfile(name):
    #     returan
    X_range = np.linspace(-10,10,10)
    Y_range = np.linspace(-10,10,10)

    X, Y = np.meshgrid(X_range, Y_range)

    X_slopes, Y_slopes = dX_dt(A[0][0],A[0][1],X,Y), dY_dt(A[1][0],A[1][1],X,Y)
    fig = go.Figure(ff.create_streamline(X_range, Y_range, X_slopes, Y_slopes, arrow_scale=.5, density=1.7))
    fig.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        title=f'Phase space of the matrix {name}'
    )
    return fig

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

server = app.server

app.layout = dbc.Container(
    [
        html.H1('2 by 2 matrix properties', style={'textAlign': 'center'}),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H2('Enter the values of the matrix', style={'margin-top': '20px'}),
                            dbc.Col(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                # input of type number with no arrows
                                                dbc.Input(id='a', type='number', value=None, style={'textAlign': 'center'}),
                                                width=5
                                            ),
                                            dbc.Col(
                                                dbc.Input(id='b', type='number', value=None, style={'textAlign': 'center'}),
                                                width=5
                                            )
                                        ],
                                        style={'margin-top': '20px'}
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Input(id='c', type='number', value=None, style={'textAlign': 'center'}),
                                                width=5
                                            ),
                                            dbc.Col(
                                                dbc.Input(id='d', type='number', value=None, style={'textAlign': 'center'}),
                                                width=5
                                            )
                                        ],
                                        align='center',
                                        justify='left',
                                        style={'margin-top': '20px'}
                                    )
                                ],
                                width='auto',
                                align='center',
                                style={'margin-top': '20px'}
                            ),
                            html.Div(id='output-markdown', style={'margin-top': '20px'}),
                            html.Div(id='output-canonical-form', style={'margin-top': '20px'}),
                            html.Div(id='output-eigenvector', style={'margin-top': '20px'}),
                            html.Div(id='output-eigenvalues', style={'margin-top': '20px'}),
                            html.Div(id='output-type-of-system', style={'margin-top': '20px'}),
                            html.Div(id='output-solution', style={'margin-top': '20px'})
                        ]
                    ),
                    width=5
                ),
                dbc.Col(
                    dcc.Graph(id='output-graph', figure={}, style={'margin-top': '20px', 'height': '70vh', 'width': '70vh'}, config={'displayModeBar': False, 'scrollZoom': False, 'staticPlot': True, 'editable': False, 'responsive': True, 'displaylogo': False}),
                    width='auto'
                )
            ],
            align='space-between',
            style={'margin-top': '20px', 'justify-content': 'space-around'}
        )
    ],
    fluid=True,
    style={'margin-top': '20px', 'align-items': 'center', 'justify_content': 'center'}
)

@app.callback(
    Output('output-markdown', 'children'),
    Input('a', 'value'),
    Input('b', 'value'),
    Input('c', 'value'),
    Input('d', 'value')
)
def update_markdown(a, b, c, d):
    if a != None and b != None and c != None and d != None:
        if a != 0 and b != 0 and c != 0 and d != 0:
            return dl.DashLatex(r'$$\boldsymbol{A} = \begin{bmatrix} %s & %s \\ %s & %s \end{bmatrix}$$' % (a, b, c, d))


@app.callback(
    Output('output-eigenvector', 'children'),
    Output('output-eigenvalues', 'children'),
    Output('output-type-of-system', 'children'),
    Input('a', 'value'),
    Input('b', 'value'),
    Input('c', 'value'),
    Input('d', 'value')
)
def update_eigenproperties(a, b, c, d):
    if a != None and b != None and c != None and d != None:
        if (a != 0 and b != 0 and c != 0 and d != 0) or (a == 0 or b == 0 or c == 0 or d == 0):
            a, b, c, d = float(a), float(b), float(c), float(d)
            matrix = np.array([[a, b], [c, d]])
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            out_eigenvectors = dl.DashLatex(r'$$\boldsymbol{\lambda} = \begin{bmatrix} %s \\ %s \end{bmatrix} ~~~~~ \boldsymbol{\lambda} = \begin{bmatrix} %s \\ %s \end{bmatrix}$$' % (np.round(eigenvectors[0][0],4), np.round(eigenvectors[0][1],4), np.round(eigenvectors[1][0],4), np.round(eigenvectors[1][1],4)))
            out_eigenvalues = dl.DashLatex(r'$$\lambda_1 = %s ~~~~~ \lambda_2 = %s$$' % (np.round(eigenvalues[0],4), np.round(eigenvalues[1],4)))
            e_type = 'Saddle'
            if np.iscomplex(eigenvalues[0]) and np.iscomplex(eigenvalues[1]):
                e_type = 'Spiral'
                if eigenvalues[0].real == 0 and eigenvalues[1].real == 0:
                    e_type = 'Center'
                elif eigenvalues[0].real < 0 and eigenvalues[1].real < 0:
                    e_type = 'Sink Spiral'
                elif eigenvalues[0].real > 0 and eigenvalues[1].real > 0:
                    e_type = 'Source Spiral'     
            else:
                if eigenvalues[0] < 0 and eigenvalues[1] < 0:
                    e_type = 'Sink'
                elif eigenvalues[0] > 0 and eigenvalues[1] > 0:
                    e_type = 'Source'
            out_type_of_system = dl.DashLatex(r'$$\text{Type of system:  \bf{%s}}$$' % (e_type))

            return out_eigenvectors, out_eigenvalues, out_type_of_system
    return {}, {}, {}

# plot of the phase space of the matrix
@app.callback(
    Output('output-graph', 'figure'),
    Input('a', 'value'),
    Input('b', 'value'),
    Input('c', 'value'),
    Input('d', 'value')
)
def update_graph(a, b, c, d):
    if a != None and b != None and c != None and d != None:
        if a != 0 and b != 0 and c != 0 and d != 0:
            a, b, c, d = float(a), float(b), float(c), float(d)
            fig = generate_plot([[a, b], [c, d]], 'A')
            return fig
        elif a == 0 or b == 0 or c == 0 or d == 0:
            a, b, c, d = float(a), float(b), float(c), float(d)
            fig = generate_plot([[a, b], [c, d]], 'A')
            return fig
    return {}

# canonical form of the matrix
@app.callback(
    Output('output-canonical-form', 'children'),
    Input('a', 'value'),
    Input('b', 'value'),
    Input('c', 'value'),
    Input('d', 'value')
)
def update_canonical_form(a, b, c, d):
    if a != None and b != None and c != None and d != None:
        if a != 0 and b != 0 and c != 0 and d != 0:
            a, b, c, d = float(a), float(b), float(c), float(d)
            matrix = np.array([[a, b], [c, d]])
            _, eigenvectors = np.linalg.eig(matrix)
            P = np.array([eigenvectors[0], eigenvectors[1]])
            P_inv = np.linalg.inv(P)
            Ac = P_inv * matrix * P
            out_canonical_form = dl.DashLatex(rf'$$\boldsymbol{{A_c}} = \begin{{bmatrix}} {Ac[0][0]:.4f} & {Ac[0][1]:.4f} \\ {Ac[1][0]:.4f} & {Ac[1][1]:.4f} \end{{bmatrix}}$$')
            return out_canonical_form
        elif a == 0 or b == 0 or c == 0 or d == 0:
            a, b, c, d = float(a), float(b), float(c), float(d)
            matrix = np.array([[a, b], [c, d]])
            _, eigenvectors = np.linalg.eig(matrix)
            P = np.array([eigenvectors[0], eigenvectors[1]])
            P_inv = np.linalg.inv(P)
            Ac = P_inv * matrix * P
            out_canonical_form = dl.DashLatex(rf'$$\boldsymbol{{A_c}} = \begin{{bmatrix}} {Ac[0][0]:.4f} & {Ac[0][1]:.4f} \\ {Ac[1][0]:.4f} & {Ac[1][1]:.4f} \end{{bmatrix}}$$')
            return out_canonical_form
    return {}

# solution of the equation
@app.callback(
    Output('output-solution', 'children'),
    Input('a', 'value'),
    Input('b', 'value'),
    Input('c', 'value'),
    Input('d', 'value')
)
def update_solution(a, b, c, d):
    if a != None and b != None and c != None and d != None:
        if a != 0 and b != 0 and c != 0 and d != 0:
            a, b, c, d = float(a), float(b), float(c), float(d)
            matrix = np.array([[a, b], [c, d]])
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            out_solution = dl.DashLatex(fr'$$\boldsymbol{{X}}(t) = c_1 e^{{{eigenvalues[0]:.4f} t}} \begin{{bmatrix}} {eigenvectors[0][0]:.4f} \\ {eigenvectors[0][1]:.4f} \end{{bmatrix}} + c_2 e^{{{eigenvalues[0]:.4f} t}} \begin{{bmatrix}} {eigenvectors[1][0]:.4f} \\ {eigenvectors[1][1]:.4f} \end{{bmatrix}}$$')
            return out_solution
        elif a == 0 or b == 0 or c == 0 or d == 0:
            a, b, c, d = float(a), float(b), float(c), float(d)
            matrix = np.array([[a, b], [c, d]])
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            out_solution = dl.DashLatex(fr'$$\boldsymbol{{X}}(t) = c_1 e^{{{eigenvalues[0]:.4f} t}} \begin{{bmatrix}} {eigenvectors[0][0]:.4f} \\ {eigenvectors[0][1]:.4f} \end{{bmatrix}} + c_2 e^{{{eigenvalues[0]:.4f} t}} \begin{{bmatrix}} {eigenvectors[1][0]:.4f} \\ {eigenvectors[1][1]:.4f} \end{{bmatrix}}$$')
            return out_solution
    return {}

if __name__ == '__main__':
    # run as production
    app.run_server(host='0.0.0.0',debug=False, port=8050)
