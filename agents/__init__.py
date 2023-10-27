from .basic import Agent
from .pedestrian import Pedestrian
from .car import Car
from .bus import Bus

Agent_mapper = {
    'Pedestrian': Pedestrian,
    'Private_car': Car,
    'Bus': Bus
}