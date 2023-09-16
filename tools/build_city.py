import yaml
import numpy as np

def list_representer(dumper, data):
    return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=False)

yaml.add_representer(list, list_representer)

BUILDING_TYPES = ["House", "Office", "Gas Station", "Garage", "Store"]
probabilities = [0.2, 0.2, 0.05, 0.15, 0.4]
np.random.seed(42)

base_yaml_file = 'config/maps/v0.1.yaml'

with open(base_yaml_file, 'r') as file:
    city_config = yaml.safe_load(file)

city_config['buildings'] = []
# Traffic Streets
for j in range(48, 200, 48):
    # vertical
    traf_stree = {
        'position': [7, j],
        'length': 243,
        'orientation': 'vertical',
        'type': 'Traffic Street',
        'directions': 2,
        'width': 7
    }
    city_config['streets'].append(traf_stree)
    # horison
    traf_stree = {
        'position': [j, 7],
        'length': 243,
        'orientation': 'horizontal',
        'type': 'Traffic Street',
        'directions': 2,
        'width': 7
    }
    city_config['streets'].append(traf_stree)

# Walking
for j in range(45, 200, 48):
    # vertical
    traf_stree = {
        'position': [10, j],
        'length': 230,
        'orientation': 'vertical',
        'type': 'Walking Street',
        'directions': 2,
        'width': 3
    }
    city_config['streets'].append(traf_stree)
    traf_stree = {
        'position': [10, j+10],
        'length': 230,
        'orientation': 'vertical',
        'type': 'Walking Street',
        'directions': 2,
        'width': 3
    }
    city_config['streets'].append(traf_stree)
    # horison
    traf_stree = {
        'position': [j, 10],
        'length': 230,
        'orientation': 'horizontal',
        'type': 'Walking Street',
        'directions': 2,
        'width': 3
    }
    city_config['streets'].append(traf_stree)
    traf_stree = {
        'position': [j+10, 10],
        'length': 230,
        'orientation': 'horizontal',
        'type': 'Walking Street',
        'directions': 2,
        'width': 3
    }
    city_config['streets'].append(traf_stree)

# inter walking and buildings
ind = list(range(26, 200, 48))
ind.append(219)

for i, col in enumerate(ind):
    for j, row in enumerate(ind):
        # buildings
        block_id = i*5 + j + 1
        size = 16
        if i == 4 or j == 4:
            size = 17
        # vertical
        traf_stree = {
        'position': [10 + i*48, row],
        'length': 35 if i != 4 else 38,
        'orientation': 'vertical',
        'type': 'Walking Street',
        'directions': 2,
        'width': 3
        }
        if i == 4 and j != 4:
            traf_stree['width'] = 1
            traf_stree['position'][1] = row+1
        if j==4:
            traf_stree['width'] = 4
            traf_stree['position'][1] = row
        city_config['streets'].append(traf_stree)
        v_wid = traf_stree['width']
        v_s = traf_stree['position'][1]

        traf_stree = {
        'position': [col, 10 + j*48],
        'length': 35 if j != 4 else 38,
        'orientation': 'horizontal',
        'type': 'Walking Street',
        'directions': 2,
        'width': 3
        }
        if j == 4 and i != 4:
            traf_stree['width'] = 1
            traf_stree['position'][0] = col+1
        if i==4:
            traf_stree['width'] = 4
            traf_stree['position'][0] = col
        city_config['streets'].append(traf_stree)
        h_wid = traf_stree['width']
        h_s = traf_stree['position'][0]

        chosen_building = np.random.choice(BUILDING_TYPES, p=probabilities)
        chosen_building = str(chosen_building)
        # top-left
        building = {
            'position': [v_s - size, h_s - size],
            'size': [size, size],
            'block': block_id,
            'type': chosen_building,
            'height': np.random.randint(10, 30)
        }
        city_config['buildings'].append(building)

        chosen_building = np.random.choice(BUILDING_TYPES, p=probabilities)
        chosen_building = str(chosen_building)
        # top-right
        building = {
            'position': [v_s + v_wid, h_s - size],
            'size': [size, size],
            'block': block_id,
            'type': chosen_building,
            'height': np.random.randint(10, 30)
        }
        city_config['buildings'].append(building)

        chosen_building = np.random.choice(BUILDING_TYPES, p=probabilities)
        chosen_building = str(chosen_building)
        # bottom-left
        building = {
            'position': [v_s - size, h_s + h_wid],
            'size': [size, size],
            'block': block_id,
            'type': chosen_building,
            'height': np.random.randint(10, 30)
        }
        city_config['buildings'].append(building)

        chosen_building = np.random.choice(BUILDING_TYPES, p=probabilities)
        chosen_building = str(chosen_building)
        # bottom-right
        building = {
            'position': [v_s + v_wid, h_s + h_wid],
            'size': [size, size],
            'block': block_id,
            'type': chosen_building,
            'height': np.random.randint(10, 30)
        }
        city_config['buildings'].append(building)


with open('TEST.yaml', 'w') as file:
    yaml.dump(city_config, file, default_flow_style=False)


