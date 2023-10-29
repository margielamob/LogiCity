import numpy as np
import plotly.graph_objects as go

# Load the 2D points
points = np.loadtxt('core/road_graph.txt')

# Create a scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=points[:, 1],  # x-coordinates are now the second dimension
    y=points[:, 0],  # y-coordinates are the first dimension
    mode='markers+text',
    text=[str(i) for i in range(len(points))],
    textposition="top center",
    marker=dict(size=5, color='red'),
))

# Update layout for better appearance and reverse the direction of the y-axis
fig.update_layout(
    title="2D Points Visualization",
    xaxis_title="X Coordinate",
    yaxis_title="Y Coordinate",
    showlegend=False,
    yaxis=dict(autorange='reversed')  # This line ensures the y-axis points downwards
)

fig.show()