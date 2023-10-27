from .basic import Agent
from .pedestrian import Pedestrian
from .car import Car

Agent_mapper = {
    'Pedestrian': Pedestrian,
    'Private_car': Car
}