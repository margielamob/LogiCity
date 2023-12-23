from problog.program import PrologString
from problog import get_evaluatable
import time

def problog_string():
    agents = [f'agent_{i+1}' for i in range(8)]
    intersections = [f'intersection_{i+1}' for i in range(32)]

    # Start building the ProbLog model string
    problog_model = []

    # Define agents and intersections
    problog_model.append('% Define agents and intersections')
    for agent in agents:
        problog_model.append(f'agent({agent}).')
    for intersection in intersections:
        problog_model.append(f'intersection({intersection}).')

    # Add probabilistic facts about agents being pedestrians
    problog_model.append('% Probabilistic facts about agents being pedestrians')
    for agent in agents:
        prob_probability = 1.0  # Example probability, adjust as needed
        problog_model.append(f'{prob_probability}::is_pedestrian({agent}).')

    # Add probabilistic facts about intersections having empty cars
    problog_model.append('% Probabilistic facts about intersections')
    for intersection in intersections:
        prob_probability = 1.0  # Example probability, adjust as needed
        problog_model.append(f'{prob_probability}::is_inter_car_empty({intersection}).')

    # Add probabilistic facts about agents being at intersections
    problog_model.append('% Probabilistic facts about agents being at intersections')
    for agent in agents:
        for intersection in intersections:
            prob_probability = 1.0
            problog_model.append(f'{prob_probability}::is_at({agent}, {intersection}).')
    # Define rules
    problog_model.append('''
    % Rule for stopping
    should_stop(Agent) :-
        agent(Agent), intersection(Intersection),
        is_pedestrian(Agent), not is_inter_car_empty(Intersection), is_at(Agent, Intersection).
    ''')

    # Define queries
    problog_model.append('% Queries')
    for agent in agents:
        problog_model.append(f'query(should_stop({agent})).')

    # Combine into a single string
    problog_model_str = '\n'.join(problog_model)

    # Print or use the generated ProbLog model
    print(problog_model_str)
    return problog_model_str

def main():
    problog_model_str = problog_string()
    problog_model = PrologString(problog_model_str)
    s = time.time()
    problog_eval = get_evaluatable().create_from(problog_model)
    result = problog_eval.evaluate()
    print("Time: ", time.time() - s)
    print(result)

if __name__ == '__main__':
    main()