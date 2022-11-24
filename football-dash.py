# =====================================================================
# IMPORTS
# ---------------------------------------------------------------------
# general imports
import numpy as np
import pandas as pd
from urllib.request import urlopen
import json

# Dash imports
from dash import Dash, dcc, html, ctx
# import dash_core_components as dcc
# import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# visualization imports
import matplotlib as mpl
from matplotlib import cm
import plotly.graph_objects as go

# =====================================================================
# CONSTANTS
# ---------------------------------------------------------------------
season_str = "2021-22"
positions = ["DF", "MF", "FW"]
positions_str = ", ".join(positions)
positions_long = {"DF": "Defenders", "MF": "Midfielders", "FW": "Forwards"}
descr_chr = ["player", "nationality", "position", "team", "comp_level", "age", "minutes_90s", "dominant_foot"]
MIN_90s = 10

features_groups = {
    "DF": {
        "def_actions": ["tackles_plus_interceptions_per90","clearances_per90","tackles_won_per90","ball_recoveries_per90", "aerials_won_per90"],
        "press": ["pressures_per90", "pressures_mid_3rd_per90"],
        "pass": ["passes_completed_per90", "passes_completed_long_per90", "crosses_into_penalty_area_per90"],
        "prog": ["progressive_passes_per90","carry_progressive_distance_per90","dribbles_completed_per90"],
        "atk_actions": ["sca_per90","gca_per90","assisted_shots_per90","shots_total_per90","xg_per90","xa_per90"] 
    },
    "MF": {
        "def_actions": ["tackles_plus_interceptions_per90","clearances_per90","tackles_won_per90","ball_recoveries_per90"],
        "press": ["pressures_per90","tackles_att_3rd_per90"],
        "pass": ["passes_completed_per90","passes_into_penalty_area_per90","crosses_into_penalty_area_per90"],
        "prog": ["progressive_passes_per90","carry_progressive_distance_per90","dribbles_completed_per90"],
        "atk_actions": ["sca_per90","gca_per90","assisted_shots_per90","shots_total_per90","goals_per90","xg_per90","xa_per90"] 
    },
    "FW": {
        "def_actions": ["tackles_won_per90","ball_recoveries_per90"],
        "press": ["pressures_per90","tackles_att_3rd_per90"],
        "pass": ["passes_completed_per90","passes_into_penalty_area_per90"],
        "prog": ["carry_progressive_distance_per90","dribbles_completed_per90"],
        "atk_actions": ["sca_per90","gca_per90","assisted_shots_per90","shots_total_per90","goals_per90","xg_per90","xa_per90"]
    }
}

cluster_labels_list = {
    "DF": ["Attacking Defenders", "Carriers & Builders", "Pressers", "Sweepers"],
    "MF": ["Tacklers", "Creators", "Progressers", "Second Strikers"],
    "FW": ["Finishers", "Pressers", "Assisting Forwards"]
}

n_clusters = {k: len(v) for k,v in cluster_labels_list.items()}

dff_filename = "https://raw.githubusercontent.com/yuvalaltman/hostedFiles/main/dff_2021-22.csv"
data_prcnt_groups_filename = "https://raw.githubusercontent.com/yuvalaltman/hostedFiles/main/data_prcnt_2021-22.json"
cluster_labels_groups_filename = "https://raw.githubusercontent.com/yuvalaltman/hostedFiles/main/cluster_labels_2021-22.json"

country_placeholder = "<country>"
flag_img_template = "https://raw.githubusercontent.com/ninjanye/flag-icon-css/cc4ba085e285f449cd12f0babe4387e169932447/flags/4x3/"+country_placeholder+".svg"
countries_filename = "https://raw.githubusercontent.com/yuvalaltman/hostedFiles/main/countries.json"
countries_fullname_filename = "https://raw.githubusercontent.com/yuvalaltman/hostedFiles/main/countries_fullname.json"

main_title = "Players from Europe's Top 5 Leagues"

# =====================================================================
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def dict_values_ravel(d):
    """flatten all values of dictionary d
       and return them as a list"""
    return [item for sublist in d.values() for item in sublist]

def get_player_color(pos, player_cluster, cmap="tab20c"):
    return mpl.colors.to_hex(
        cm.get_cmap(cmap)(
            np.insert(
                np.array(list(n_clusters.values())).cumsum(), 0, 0
            )[positions.index(pos)] + player_cluster
        )
    )

def read_json_inner(filename):
    response = urlopen(filename)
    return json.loads(response.read())

def read_json(filename, fmt="list"):
    res = read_json_inner(filename)
    data_out = {}
    for pos in positions:
        if fmt == "list":
            data_out[pos] = res[pos]
        elif fmt == "dict":        
            data_out[pos] = pd.DataFrame(res[pos], columns=descr_chr + features_list_groups[pos])
    return data_out

# =====================================================================
# DATA LOADING AND PROCESSING
# ---------------------------------------------------------------------

features_list_groups = {pos: dict_values_ravel(features_groups[pos]) for pos in positions}

ticklabels_clean_groups = {}
for pos in positions:
    ticklabels = [item for sublist in list(features_groups[pos].values()) for item in sublist]
    ticklabels_clean = [tl[:-6].replace("_"," ") for tl in ticklabels]
    ticklabels_clean = ["key passes" if tl=="assisted shots" else tl for tl in ticklabels_clean]
    ticklabels_clean_groups[pos] = ticklabels_clean
    del ticklabels_clean
    
cluster_labels_groups = read_json(cluster_labels_groups_filename, fmt="list")
data_prcnt_groups = read_json(data_prcnt_groups_filename, fmt="dict")
dff = dff = pd.read_csv(dff_filename)
countries = read_json_inner(countries_filename)
countries_fullname = read_json_inner(countries_fullname_filename)

# =====================================================================
# DASH APP
# ---------------------------------------------------------------------
          
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="football-dash",
)
server = app.server

# -------------------------------------------------------------------------------------

def is_small_screen(screen_width):
    return True if screen_width < small_screen_width else False

# -------------------------------------------------------------------------------------

small_screen_width = 768 # pixels [https://getbootstrap.com/docs/5.1/layout/grid/]

marker_size = {"small": 6, "large": 12}

font_size = {
    "tickfont": {"small": 8, "large": 10},
    "cluster_label": {"small": 14, "large": 20},
    "player_title": {"small": 12, "large": 16},
    "player_subtitle": {"small": 10, "large": 14}
}

text_color = {
    "light_mode": "rgba(0, 0, 0)",
    "dark_mode": "rgba(255, 255, 255)"
}


flag_img = {
    "location": {
        "x": {"small": 0.5, "large": 0.5},
        "y": {"small": 0.93, "large": 1.2}
    },
    "size": {
        "x": {"small": 0.1, "large": 0.05},
        "y": {"small": 0.1, "large": 0.05}
    },
    "opacity": 0.8
}

config_graph = {"displaylogo": False}

# -------------------------------------------------------------------------------------  

# define "about" tooltip modal buttons for the presented graphs

def get_modal(button_title, modal_header, modal_body, id_suffix):
    return html.Div([
        dbc.Button(
            button_title,
            color="primary",
            id="open-"+id_suffix,
            n_clicks=0,
            size="sm",
            className="m-1"
        ),
        dbc.Modal([
            dbc.ModalHeader(modal_header),
            dbc.ModalBody(modal_body),
            dbc.ModalFooter(dbc.Button("Close", id="close-"+id_suffix, className="ms-auto", n_clicks=0))
        ],
            id=id_suffix,
            is_open=False
        )])

modal_scatter_button = "About scatter graph"
modal_scatter_header = "Player Similarity Graph"
modal_scatter_body = html.Div([
    html.P(
        "The graph visualizes players based on their similarities: players with similar performance attributes " +\
        f"are plotted in proximity. For each position ({positions_str}), a new scatter graph is generated. " +\
        "The graph can be filtered by league and by players' age."
    ),
    html.P(
        "Explore players by hovering over or clicking the points in the graph. " +\
        "You can also search for specific players in the search box above the graph. "
    ),
    html.P(
        "The scatter graph displays 2D embeddings learned by a t-SNE model, clustered with k-Means.",
        style={"fontStyle": "italic"})
])
modal_scatter_id = "about-scatter"
modal_scatter = get_modal(modal_scatter_button, modal_scatter_header, modal_scatter_body, modal_scatter_id)

modal_radar_button = "About player graph"
modal_radar_header = "Player Radar Graph"
modal_radar_body = html.Div([
    html.P([
        "Player attributes are compared to those of other players in the same position and are presented "+\
        "as percentiles in the graph. For example, a player with a 0.95 ",
        html.Span("passes completed ", style={"fontStyle": "italic"}),
        "value, is in the 95",html.Sup("th"), " percentile when comparing to other players in the same position, in terms " +\
        "of completed passes."
    ])
])
modal_radar_id = "about-radar"
modal_radar = get_modal(modal_radar_button, modal_radar_header, modal_radar_body, modal_radar_id)

@app.callback(
    Output(modal_scatter_id, "is_open"),
    [Input("open-"+modal_scatter_id, "n_clicks"), Input("close-"+modal_scatter_id, "n_clicks")],
    [State(modal_scatter_id, "is_open")],
)
def toggle_modal_scatter(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output(modal_radar_id, "is_open"),
    [Input("open-"+modal_radar_id, "n_clicks"), Input("close-"+modal_radar_id, "n_clicks")],
    [State(modal_radar_id, "is_open")],
)
def toggle_modal_radar(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# -------------------------------------------------------------------------------------    

player_search_placeholder="Search player..."

player_search = dcc.Dropdown(
    dff["player"],
    id="player-search",
    placeholder=player_search_placeholder,
    className="m-1",
)

# -------------------------------------------------------------------------------------  

countries_all_str = "All"
countries_fullname[countries_all_str] = countries_all_str
country_search_placeholder = countries_all_str

def get_country_search_options(df):
    return [
        {
            "label": html.Div(
                [
                    html.Img(src=flag_img_template.replace(country_placeholder, countries[i]), height=20),
                    html.Div(countries_fullname[i], style={'font-size': 15, 'padding-left': 10}),
                ],
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            ),
            "value": i
        }
        if i in countries.keys()
        else {"label": countries_fullname[i], "value": i}
        for i in np.insert(sorted(df["nationality"].unique()), 0, countries_all_str)
    ]

country_search = dcc.Dropdown(
    id="country-search",
    options=get_country_search_options(dff),
    value=countries_all_str,
    placeholder=country_search_placeholder,
    clearable=False
)

# -------------------------------------------------------------------------------------  

filters = html.Div(
    [
        dbc.Button("Options", id="open-filters", n_clicks=0, className="m-1", size="sm"),
        dbc.Offcanvas([
            html.Div(
                [
                    dbc.Label("Position:"),
                    dcc.Dropdown(
                    id="crossfilter-position",
                    options=[{"label": i, "value": i} for i in positions],
                    value=positions[0],
                    clearable=False
                    )
                ]
            ),
            html.Div(
                [
                    html.Br(),
                    dbc.Label("League:"),
                    dcc.Dropdown(
                        id="crossfilter-league",
                        options=[{"label": i, "value": i} for i in np.insert(dff["league"].unique(), 0, "All leagues")],
                        value="All leagues",
                        clearable=False
                    )
                ]
            ),
            html.Div(
                [
                    html.Br(),
                    dbc.Label("Ages:"),
                    dcc.RangeSlider(
                        id="age-range-slider",
                        value=[dff["age"].min(), dff["age"].max()],
                        min=dff["age"].min(),
                        max=dff["age"].max(),
                        step=1,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                        allowCross=False
                    )
                ]
            ),
            html.Div(
                [
                    html.Br(),
                    dbc.Label("Update player graph on:"),
                    dbc.RadioItems(
                        options=[
                            {"label": "hover", "value": "hover"},
                            {"label": "click (mobile-friendly)", "value": "click"}
                        ],
                        value="hover",
                        id="player-update-prompt",
                        inline=True
                    )
                ]
            ),
            html.Div(
                [
                    html.Br(),
                    dbc.Label("Nationality:"),
                    country_search
                ]
            ),
        ],
            id="filters",
            title="Options",
            is_open=False,
            scrollable=True,
        ),
    ]
)

@app.callback(
    Output("filters", "is_open"),
    Input("open-filters", "n_clicks"),
    [State("filters", "is_open")],
)
def toggle_filters(n1, is_open):
    if n1:
        return not is_open
    return is_open

# ------------------------------------------------------------------------------------- 

app.layout = dbc.Container([
    
    # **************************************************
    # get user window width
    dcc.Location(id="url"),
    html.Div(id="viewport-container", hidden=True),
    # **************************************************
    
    dbc.Tooltip(
        html.P([
            "You can change the position in ",
            html.Span("Options", style={"fontStyle": "italic"})
        ]),
        target="tooltip-target"
    ),
    
    dbc.Row([
        dbc.Col([
            html.H4(
                main_title,
                id="main-title",
                style={
                    "textAlign": "center",
                    "fontWeight": "bold",
                    "fontStretch": "expanded"
                }
            )
        ])
    ],
        justify="center"
    ),
    
    dbc.Row([
        dbc.Col([
            html.P([
                f"Min. {MIN_90s} 90-minutes completed | {season_str} Season | Data from ",
                html.A("Fbref.com", href="https://fbref.com/en/", target="_blank")
            ],
            style={
                "textAlign": "center",
                "fontStretch": "expanded"
            }
            )
        ])
    ]),
        
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup(
                [filters, modal_scatter, modal_radar],
                size="sm",
                className="d-grid d-md-flex justify-content-md-center flex-wrap"
            )], lg=3, align="center")
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            player_search
        ], lg=3, align="center")
    ], justify="center"),
        
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(
                    id="crossfilter-scatter",
                    hoverData={
                        "points": [
                            {
                                "customdata": np.stack(
                                    (
                                        dff["player"].iloc[0],
                                        dff["position"].iloc[0]
                                    ),
                                    axis=-1
                                )
                            }
                        ]
                    },
                    clickData={
                        "points": [
                            {
                                "customdata": np.stack(
                                    (
                                        dff["player"].iloc[0],
                                        dff["position"].iloc[0]
                                    ),
                                    axis=-1
                                )
                            }
                        ]
                    },
                    config=config_graph
                )],
            ),
        ], xs=10, align="center")
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            html.Div(
                [dcc.Graph(id="player-radar", config=config_graph)],
                style={"veritcalAlign": "center","horizontalAlign": "center"}
            ),
        ], xs=10, align="center")
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button(
                    "About this graph's variables",
                    color="secondary",
                    id="open-about-radarVars",
                    n_clicks=0,
                    size="sm",
                    className="m-1"
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("The Variables in Player Graph"),
                        dbc.ModalBody(
                            html.Div([html.P([""])]),
                            id="about-radarVars-modalBody"
                        ),
                        dbc.ModalFooter(dbc.Button("Close", id="close-about-radarVars", className="ms-auto", n_clicks=0))
                    ],
                    id="about-radarVars",
                    is_open=False,
                    scrollable=True
                )
            ])
        ], align="center")
    ], justify="center"),
    
], fluid=True)

# -------------------------------------------------------------------------------------

@app.callback(
    Output("about-radarVars", "is_open"),
    [Input("open-about-radarVars", "n_clicks"), Input("close-about-radarVars", "n_clicks")],
    [State("about-radarVars", "is_open")],
)
def toggle_modal_radar(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# -------------------------------------------------------------------------------------
# get the user screen width as a clientside function
app.clientside_callback(
    """
    function(href) {
        var w = window.innerWidth;
        return (w);
    }
    """,
    Output('viewport-container', 'children'),
    Input('url', 'href')
)

# -------------------------------------------------------------------------------------

@app.callback(
    [
        Output("crossfilter-scatter", "figure"),
        Output("player-search", "options"),
        Output("country-search", "options")
    ],
    [
        Input("crossfilter-position", "value"),
        Input("crossfilter-league", "value"),
        Input("age-range-slider", "value"),
        Input("viewport-container", "children"),
        Input("player-search", "value"),
        Input("country-search", "value")
    ]
)
def update_graph(position, league, ages, screen_width, player, country):
    triggered_id = ctx.triggered_id
    
    fig = go.Figure()
    
    ms = marker_size["small"] if is_small_screen(screen_width) else marker_size["large"]
    marker_opacity = 0.8
    selected_marker_opacity = 0.9
    unselected_marker_opacity = 0.5*marker_opacity
       
    dff_f = dff[dff["position"] == position]
    dff_f = dff_f[(dff_f["age"]>=ages[0]) & (dff_f["age"]<=ages[1])]
    if not league.startswith("All "):
        dff_f = dff_f[dff_f["league"] == league]
    if not country == countries_all_str:
        dff_f = dff_f[dff_f["nationality"] == country]
    
    ind_player = None
    player_cluster = None
    idx_player = None
    
    if triggered_id == "player-search":
        if (player is not None) and (player != player_search_placeholder):
            ind_player = dff_f.index[dff_f["player"]==player].values[0]
            player_cluster = dff_f["cluster"][ind_player]
            idx_player = np.where(dff_f[dff_f["cluster"] == player_cluster]["player"]==player)[0][0]

            fig.add_annotation(
                {
                    "font":
                    {
                        "color": text_color["light_mode"],
                        "size": 10,
                    },
                    "x": dff_f["x"][dff_f["player"] == player].values[0],
                    "y": dff_f["y"][dff_f["player"] == player].values[0],
                    "xref": "x",
                    "yref": "y",
                    "showarrow": False,
                    "text": "<span style=" +\
                    "\"text-shadow: 0em 0em 1.0em" +\
                    get_player_color(position, player_cluster) +\
                    "\">" +\
                    player.replace(" ","<br>") +\
                    "</span>",
                    "textangle": 0,
                    "opacity": 0.9
                }
            )      
    
    for i in range(n_clusters[position]):
        fig.add_scattergl(
            x = dff_f["x"][dff_f["cluster"] == i],
            y = dff_f["y"][dff_f["cluster"] == i],
            mode = "markers",
            marker = {
                "size": ms,
                "color": get_player_color(position, i),
                "opacity": marker_opacity,
                "line": {
                    "color": text_color["light_mode"],
                    "width": 0.1
                }
            },
            name = cluster_labels_groups[position][i],
            text = dff_f["player"][dff_f["cluster"] == i],
            hovertemplate = "%{text}<br>",
            customdata = np.stack(
                (
                    dff_f["player"][dff_f["cluster"] == i],
                    [position] * len(dff_f[dff_f["cluster"] == i])
                ),
                axis=-1
            ),
        )
                
        if np.sum(dff_f["cluster"] == i)>0:
            fig.add_annotation(
                {
                    "font":
                    {
                        "color": text_color["light_mode"],
                        "size": font_size["cluster_label"]["small"] if is_small_screen(screen_width) else font_size["cluster_label"]["large"],
                    },
                    "x": dff_f["x"][dff_f["cluster"] == i].mean(),
                    "y": dff_f["y"][dff_f["cluster"] == i].mean(),
                    "xref": "x",
                    "yref": "y",
                    "showarrow": False,
                    "text": cluster_labels_groups[position][i],
                    "textangle": 0,
                    "opacity": 0.3
                }
            )
            
    if triggered_id == "player-search":
        if (idx_player is not None) and (player_cluster is not None):
            fig.update_traces(
                selectedpoints=[idx_player],
                selected_marker_opacity=selected_marker_opacity,
                selected_marker_size=int(1.5*ms),
                unselected_marker_opacity=unselected_marker_opacity,
                unselected_marker_size=int(0.75*ms),
                selector={"name": cluster_labels_groups[position][player_cluster]}
            )

            other_clusters = [cluster_labels_groups[position][i] for i in range(len(cluster_labels_groups[position])) if i!=player_cluster]
            for oc in other_clusters:
                fig.update_traces(
                    selectedpoints=[],
                    unselected_marker_opacity=unselected_marker_opacity,
                    unselected_marker_size=int(0.75*ms),
                    selector={"name": oc}
                )
        
            
    fig.update_layout(
        autosize = True,
        showlegend=False,
        xaxis = {"visible": False},
        yaxis = {"visible": False},
        hovermode = "closest",
        dragmode=False,
        margin = {"l": 0, "b": 0, "t": 0, "r": 0},
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)"
    )
    
    return fig, dff_f["player"], get_country_search_options(dff_f)
    
# -------------------------------------------------------------------------------------

def create_player_radar(player, position, screen_width):
    
    ind_player = data_prcnt_groups[position]["player"]==player
    player_data = data_prcnt_groups[position].loc[ind_player, features_list_groups[position]].values[0]
    player_cluster = dff[(dff["player"]==player) & (dff["position"]==position)]["cluster"].values[0]
    color = get_player_color(position, player_cluster)
    
    theta=[tl.replace(" ","<br>") for tl in ticklabels_clean_groups[position]]
    
    fig = go.Figure(
        data=go.Scatterpolar(
            name=player,
            r=player_data,
            theta=theta,
            fill="toself",
            fillcolor=color,
            marker={"color": color},
            hovertemplate="%{theta}:<br>" + "%{r}<extra></extra>",
            opacity=0.5
        )
    )

    player_additional_data = {k: v for (k, v) in zip(descr_chr, data_prcnt_groups[position].loc[ind_player, descr_chr].values[0])}
    
    title = player +\
    "<br><sup><br>"+\
    player_additional_data["team"]+\
    " ("+\
    player_additional_data["comp_level"]+\
    ") | "+\
    str(player_additional_data["age"])+\
    " years old | "+\
    player_additional_data["dominant_foot"]+\
    "</sup>"
    
    if player_additional_data["nationality"] in countries.keys():
        fig.add_layout_image(
            {
                "source":flag_img_template.replace(country_placeholder, countries[player_additional_data["nationality"]]),
                "x":flag_img["location"]["x"]["small"] if is_small_screen(screen_width) else flag_img["location"]["x"]["large"],
                "y":flag_img["location"]["y"]["small"] if is_small_screen(screen_width) else flag_img["location"]["y"]["large"],
                "sizex":flag_img["size"]["x"]["small"] if is_small_screen(screen_width) else flag_img["size"]["x"]["large"],
                "sizey":flag_img["size"]["y"]["small"] if is_small_screen(screen_width) else flag_img["size"]["y"]["large"],
                "xanchor":"center",
                "yanchor":"bottom",
                "opacity":flag_img["opacity"]
            }
        )
    else:
        title = title.replace("<br><sup><br>", "<br><sup>" + player_additional_data["nationality"] +"<br>")

    fig.update_layout(
        polar={
            "radialaxis": {
                "showticklabels": False,
                "gridcolor": "rgba(235, 235, 235, 0.0)"
            },
            "angularaxis": {
                "tickfont_size": font_size["tickfont"]["small"] if is_small_screen(screen_width) else font_size["tickfont"]["large"],
                "gridcolor": "rgba(235, 235, 235, 0.0)"
            },
            "bgcolor": "rgba(235, 235, 235, 0.3)"
        },
        showlegend=False,
        autosize=True,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        margin = {"l": 50, "b": 10, "t": 10, "r": 50} if is_small_screen(screen_width) else {"l": 80, "b": 80, "t": 100, "r": 80},
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": font_size["player_title"]["small"] if is_small_screen(screen_width) else font_size["player_title"]["large"]}
        },
    )
    
    fig.add_annotation(
        {
            "font":
            {
                "color": text_color["light_mode"],
                "size": font_size["player_subtitle"]["small"] if is_small_screen(screen_width) else font_size["player_subtitle"]["large"]
            },
            "x": 0.5,
            "y": -0.2,
            "showarrow": False,
            "text": "<sup>" + str(player_additional_data["minutes_90s"])+ " 90s played in "+ season_str + "</sup>",
            "textangle": 0,
            "opacity": 0.5
        }
    )

    fig.add_annotation(
        {
            "font":
            {
                "color": text_color["light_mode"],
                "size": font_size["cluster_label"]["small"] if is_small_screen(screen_width) else font_size["cluster_label"]["large"]
            },
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
            "text": cluster_labels_groups[position][player_cluster].replace("s "," ")[:-1],
            "textangle": 0,
            "opacity": 0.3
        }
    )    

    return fig

# -------------------------------------------------------------------------------------

# update the player radar graph based on either:
# - hover data
# - click data
# - player searched

@app.callback(
    Output("player-radar", "figure"),
    [
        Input("crossfilter-scatter", "hoverData"),
        Input("crossfilter-scatter", "clickData"),
        Input("player-search", "value"),
        Input("crossfilter-position", "value"),
        Input("player-update-prompt", "value"),
        Input("viewport-container", "children")
    ]
)
def update_player_radar(hoverData, clickData, player_search, position_opt, update_prompt, screen_width):
    triggered_id = ctx.triggered_id
    
    # if there is an active result for a player search, generate the player radar graph
    # based on this search result
    if triggered_id == "player-search":
        if (player_search is not None) and (player_search != player_search_placeholder):
            return create_player_radar(player_search, position_opt, screen_width)
    
    # if not in an active search result mode, generate the player radar graph
    # based on either the hover or click data, given the update_prompt
    if update_prompt == "click":
        player_name = clickData["points"][0]["customdata"][0]
        position = clickData["points"][0]["customdata"][1]
    else:
        player_name = hoverData["points"][0]["customdata"][0]
        position = hoverData["points"][0]["customdata"][1]
        
    return create_player_radar(player_name, position, screen_width)

# -------------------------------------------------------------------------------------

# automatically change the "player update prompt" method to a mobile-friendly one
# given the current user screen width

@app.callback(
    Output("player-update-prompt", "value"),
    Input("viewport-container", "children")
)
def auto_change_player_update_prompt(screen_width):
    return "click" if is_small_screen(screen_width) else "hover"

# -------------------------------------------------------------------------------------

# clear player selection dropdown value whenever options change.

@app.callback(
    Output("player-search", "value"),
    [
        Input("crossfilter-position", "value"),
        Input("open-filters", "n_clicks")
    ]
)
def clear_player_search(position, n_clicks):
    if n_clicks>0:
        return player_search_placeholder
    pass

# -------------------------------------------------------------------------------------

# change the main title to display the current position selected
@app.callback(
    Output("main-title", "children"),
    [Input("crossfilter-position", "value")]
)
def update_main_title(position):
    new_txt = html.P([
        html.Span(
            positions_long[position],
            id="tooltip-target",
            style = {"color": get_player_color(position, 0)}
        ),
        " " + main_title.split(" ", 1)[1]
    ])
    return new_txt

# -------------------------------------------------------------------------------------

# change the text of the "About this graph's variables" modal, based on the displayed position

radar_vars_info_txt = {
    "tackles_plus_interceptions_per90": "tackling an opponent, and intercepting the ball",
    "clearances_per90": "clearing the ball from a dangerous position",
    "tackles_won_per90": "how many tackles by the player ended successfuly",
    "ball_recoveries_per90": "stealing the ball from an opponent",
    "aerials_won_per90": "winning the ball in the air",
    "pressures_per90": "pressing the opponent",
    "pressures_mid_3rd_per90": "pressing the opponent in the middle of the pitch",
    "passes_completed_per90": "passes succussfuly arriving to a teammate",
    "passes_completed_long_per90": "successful passes of over than 30 yards",
    "crosses_into_penalty_area_per90": "attempts of passing the ball into the opponents box",
    "progressive_passes_per90": "completed passes that move the ball towards the opponent's goal",
    "carry_progressive_distance_per90": "distance that completed passes have traveled towards the opponent's goal",
    "dribbles_completed_per90": "dribbles completed successfuly",
    "sca_per90": "offensive actions directly leading to a shot, such as passes, dribbles and drawing fouls",
    "gca_per90": "offensive actions directly leading to a goal, such as passes, dribbles and drawing fouls",
    "assisted_shots_per90": "passes that directly lead to a shot",
    "shots_total_per90": "shots at goal, excluding penalties",
    "xg_per90": "expected goals",
    "xa_per90": "expected assists",
    "tackles_att_3rd_per90": "tackles in the attacking third of the pitch",
    "passes_into_penalty_area_per90": "completed passes into the opposition's box",
    "goals_per90": "goals scored",
}

@app.callback(
    Output("about-radarVars-modalBody", "children"),
    [Input("crossfilter-position", "value")]
)
def get_radar_vars_info_txt(position):
    txt = []
    ticklabels = [item for sublist in list(features_groups[position].values()) for item in sublist]
    for tl in ticklabels:
        key = tl.replace("_per90", "").replace("_", " ")
        if key == "assisted shots":
            key = "key passes"
        val = radar_vars_info_txt[tl]
        txt.append(
            html.P(
                [
                    html.Span(
                        key,
                        style={
                            "fontWeight": "bold",
                            "fontStyle": "italic",
                            "textDecoration": "underline"
                        }
                    ),
                    ": " + val
                ]
            )
        )
    return html.Div(txt)

# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(dev_tools_hot_reload=False)
