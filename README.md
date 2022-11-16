# football-dash
football-dash is a web app project aimed to allow for analyzing and comparing football players from Europ's Top 5 Leagues.

## The app
The app contains two main graphs: a scatter plot and a player's radar plot:
- Scatter plot: each data point in the graph is a single player. Similar players, in terms of the performances and attributes, are more likely to be plotted near each other. For each position (DF, MF, FW), a new scatter plot is generated, and in each such plot the players are clustered into different groups, e.g., different types of defenders, etc. You can explore the different players by hovering over the data points (or clicking them, when in a mobile environement), or by searching a player in the search box situated above the graph. The graph can be filtered by league and players' ages (in "Options").
- Player's radar plot: selecting a player (by hovering, clicking, or searching) updates the player's radar plot. This plot displays the player's attributes, with values denoting percentiles in the range [0, 1]: a midfielder with *completed passes*=0.95 is in the top 95<sup>th</sup> percentile of all midfielders in the analysis (i.e., midfielders in Top 5 Leagues) for *completed passes*, or, this midfielder is better in terms of *completed passes* than 95% of midfielders in the analysis.

## The model
The scatter graphs in the app (one for each position, i.e., DF, MF, FW) visualize 2D t-SNE embeddings, clustered by k-Means.
