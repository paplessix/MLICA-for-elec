import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output  # for callbacks
import plotly.graph_objects as go
import plotly.express as px
import os 
from mlca_for_elec.env.env import HouseHold
from mlca_for_elec.env.env import Microgrid
import json


# Import micro-grid 

print("Start loading household profiles")
folder_path = "config\household_profile/"
houses = []
for file in os.listdir(folder_path)[:3]:
    if file.endswith(".json"):
        household = json.load(open(folder_path+"/"+ file))
    house = HouseHold(household)
    generation_path = "data\solar_prod\Timeseries_55.672_12.592_SA2_1kWp_CdTe_14_44deg_-7deg_2020_2020.csv"
    consumption_path = f"data/consumption/Reference-{house.param['consumption']['type']}.csv"
    spot_price_path = "data/spot_price/2020.csv"
    fcr_price_path = "data/fcr_price/random_fcr.csv"
    house.load_data(generation_path,consumption_path, spot_price_path,fcr_price_path)
    houses.append(house)
print(f"Loaded {len(houses)} households")
print("Start compute social welfare")
print([house.ID for house in houses])
microgrid_1 =json.load(open("config\microgrid_profile/default_microgrid.json"))
MG = Microgrid(houses, microgrid_1)





app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])   #initialising dash app
df = px.data.stocks() #reading stock price dataset 



app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.H1("Dynamically rendered tab content"),
        html.Hr(),
        dbc.Button(
            "Regenerate graphs",
            color="primary",
            id="button",
            className="mb-3",
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="Scatter", tab_id="scatter"),
                dbc.Tab(label="Histograms", tab_id="histogram"),
            ],
            id="tabs",
            active_tab="scatter",
        ),
        html.Div(id="tab-content", className="p-4"),
    ]
)

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)

def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "scatter":
            return dcc.Graph(id = 'line_plot')
        elif active_tab == "histogram":
            return dbc.Row(
                [   dcc.Dropdown( id = 'dropdown',options = [{'label':0, 'value':0 },{'label': 1, 'value':1}], value = 0),
                    dbc.Col(dcc.Graph(id = 'line_plot'), width=6),
                    dbc.Col(dcc.Graph(id = 'line_plot'), width=6),
                ]
            )
    return "No tab selected"


@app.callback(Output(component_id='line_plot', component_property= 'figure'),
              [Input(component_id='dropdown', component_property= 'value')])

def stock_prices(dropdown_value):
    # Function for creating line chart showing Google stock prices over time 
    fig = go.Figure([go.Scatter(y=MG.households[dropdown_value].data.consumption, name = "Consumption"),
                     go.Scatter(y=MG.households[dropdown_value].data.generation, name = 'Generation')]) #creating line chart
    fig.update_layout(title = 'Prices over time',
                      xaxis_title = 'Dates',
                      yaxis_title = 'Prices'
                      )
    return fig  

if __name__ == '__main__': 
    app.run_server(debug=True)