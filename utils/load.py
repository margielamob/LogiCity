import yaml
from core.city import City
from core.building import Building
from core.street import Street
from core.config import *
from agents import Agent_mapper
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class CityLoader:
    @staticmethod
    def from_yaml(map_yaml_file, agent_yaml_file, rule_yaml_file, rule_type, debug=False):

        cached_observation = {
                "Time_Obs": {},
                "Static Info": {
                    "Agents": {},
                    "Logic": {
                        "Predicates": [],
                        "Rules": [],
                        "Groundings": {},
                    }
                }
            }

        with open(map_yaml_file, 'r') as file:
            city_config = yaml.load(file, Loader=yaml.Loader)
            logger.info("Get map info from {}".format(map_yaml_file))
        with open(agent_yaml_file, 'r') as file:
            agent_config = yaml.load(file, Loader=yaml.Loader)
            logger.info("Get agent info from {}".format(agent_yaml_file))
            for keys in agent_config.keys():
                city_config[keys] = agent_config[keys]

        # Create a city instance with the specified grid size
        city = City(grid_size=(WORLD_SIZE, WORLD_SIZE), local_planner=rule_type, rule_file=rule_yaml_file)
        cached_observation["Static Info"]["Logic"]["Predicates"] = list(city.local_planner.predicates.keys())
        cached_observation["Static Info"]["Logic"]["Rules"] = city.local_planner.data["rules"]
        for predicate in city.local_planner.predicates.keys():
            cached_observation["Static Info"]["Logic"]["Groundings"][predicate] = []

        # Add streets to the city
        logger.info("Constructing {} streets".format(len(city_config["streets"])))
        for street_data in tqdm(city_config["streets"]):
            street = Street(
                position=tuple(street_data["position"]),
                length=street_data["length"],
                orientation=street_data["orientation"],
                type=street_data["type"],
                directions=street_data["directions"],
                width=street_data["width"]
            )
            city.add_street(street)
        
        # Add buildings to the city
        logger.info("Constructing {} buildings".format(len(city_config["buildings"])))
        for building_data in tqdm(city_config["buildings"]):
            building = Building(
                block=building_data["block"],
                position=tuple(building_data["position"]),
                size=tuple(building_data["size"]),
                height=building_data["height"],
                type=building_data["type"],
            )
            city.add_building(building)
        
        # Mid lane for car planning
        city.add_mid()
        city.add_intersections()

        # Add agents to the city
        logger.info("Adding {} agents".format(len(city_config["agents"])))
        for agents_data in tqdm(city_config["agents"]):
            agent = Agent_mapper[agents_data["type"]](
                type=agents_data["type"],
                size=agents_data["size"],
                id=agents_data["id"],
                global_planner=agents_data['gplanner'],
                concepts=agents_data['concepts'] if 'concepts' in agents_data else None,
                world_state_matrix=city.city_grid,
                debug=debug
            )
            city.add_agent(agent)
            agent_name = "{}_{}".format(agents_data["type"], agent.layer_id)
            cached_observation["Static Info"]["Agents"][agent_name] = {
                "id": agent.layer_id,
                "type": agent.type,
                "size": agent.size,
            }
            for predicate in city.local_planner.predicates.keys():
                # ONLY Arity-1 predicate is supported
                cached_observation["Static Info"]["Logic"]["Groundings"][predicate].append(agent_name)

        logger.info("Done!")
        city.logic_grounds = cached_observation["Static Info"]["Logic"]["Groundings"]
        return city, cached_observation
