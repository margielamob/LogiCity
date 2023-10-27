import numpy as np
import plotly.graph_objects as go

# Load the 2D points
points = np.loadtxt('core/road_graph.txt')

# Create a scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=points[:, 0],
    y=points[:, 1],
    mode='markers+text',  # 'markers' for dots, 'text' for labels
    text=[str(i) for i in range(len(points))],  # List of labels for each point
    textposition="top center",  # Position of the labels relative to the markers
    marker=dict(size=5, color='red'),  # Adjust size and color as needed
))

# Update layout for better appearance
fig.update_layout(
    title="2D Points Visualization",
    xaxis_title="X Coordinate",
    yaxis_title="Y Coordinate",
    showlegend=False,
    yaxis=dict(autorange='reversed'),
)

fig.show()
