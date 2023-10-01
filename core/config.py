# City
BUILDING_TYPES = ["House", "Office", "Gas Station", "Garage", "Store"]
BUILDING_PROB = [0.2, 0.2, 0.05, 0.15, 0.4]
LABEL_MAP = {
    -1: 'Overlap',
    0: 'Under Construction',
    1: 'Walking Street',
    2: 'Traffic Street',
    2.5: 'Mid Lane',
    3: 'House',
    4: 'Gas Station',
    5: 'Office',
    6: 'Garage',
    7: 'Store',
    8: 'Pedestrian',
    9: 'Car'
}
COLOR_MAP = {
            -1: [100, 100, 100],
            0: [200, 200, 200],       # Grey for empty
            1: [152, 216, 170],           # Green for walking street
            2: [168, 161, 150],        # Red for traffic street
            2.5: [0, 215, 255],        # Red for traffic street
            3: [255, 204, 112],       # house
            4: [34, 102, 141],        # gas station
            5: [255, 250, 221],       # office
            6: [142, 205, 221],       # garage
            7: [255, 63, 164]         # store
        }
# 0: block, 1: building, 2: street, 3+: agent
BLOCK_ID = 0
BUILDING_ID = 1
STREET_ID = 2
BASIC_LAYER = 3
INTERSECTION_CODE = -1
MID_LINE_CODE_PLUS = 0.5
NUM_OF_BLOCKS = 25
WALKING_STREET_WID = 3
TRAFFIC_STREET_WID = 7

# Agents
AGENT_TYPES = ["Pedestrian", "Car"]
AGENT_GLOBAL_PATH_PLUS = 0.1
AGENT_GOAL_PLUS = 0.3
AGENT_START_PLUS = -0.2