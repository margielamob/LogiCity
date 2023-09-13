import yaml
from core import City, Building, Street
from agents import Agent_mapper
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class CityLoader:
    @staticmethod
    def from_yaml(yaml_file):
        with open(yaml_file, 'r') as file:
            city_config = yaml.safe_load(file)
            logger.info("Get map info from {}".format(yaml_file))

        # Create a city instance with the specified grid size
        city = City(grid_size=tuple(city_config["grid_size"]))

        # Add buildings to the city
        logger.info("Constructing {} buildings".format(len(city_config["buildings"])))
        for building_data in tqdm(city_config["buildings"]):
            building = Building(
                position=tuple(building_data["position"]),
                size=tuple(building_data["size"]),
                height=building_data["height"],
                type=building_data["type"],
            )
            city.add_building(building)

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
        
        # Add agents to the city
        logger.info("Adding {} agents".format(len(city_config["agents"])))
        for agents_data in tqdm(city_config["agents"]):
            agent = Agent_mapper[agents_data["type"]](
                type=tuple(agents_data["type"]),
                size=agents_data["size"],
                id=agents_data["id"],
                world_state_martix=city.city_grid
            )
            city.add_street(agent)

        logger.info("Done!")
        return city
