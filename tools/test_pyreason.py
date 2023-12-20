import pyreason as pr
import networkx as nx
import time

def test_hello_world(graph_path):
    # Modify pyreason settings to make verbose and to save the rule trace to a file
    pr.settings.verbose = True     # Print info to screen

    # Load all the files into pyreason
    pr.load_graphml(graph_path)
    pr.add_rule('popular(x) <-1 popular(y), Friends(x,y), owns(y,z), owns(x,z)', 'popular_rule')
    pr.add_fact(pr.Fact('popular-fact', 'Mary', 'popular', [1, 1], 0, 2))

    # Run the program for two timesteps to see the diffusion take place
    interpretation = pr.reason(timesteps=2)

    # Display the changes in the interpretation for each timestep
    dataframes = pr.filter_and_sort_nodes(interpretation, ['popular'])
    for t, df in enumerate(dataframes):
        print(f'TIMESTEP - {t}')
        print(df)
        print()

    assert len(dataframes[0]) == 1, 'At t=0 there should be one popular person'
    assert len(dataframes[1]) == 2, 'At t=0 there should be two popular people'
    assert len(dataframes[2]) == 3, 'At t=0 there should be three popular people'

    assert dataframes[0].iloc[0].component == 'Mary' and dataframes[0].iloc[0].popular == [1, 1], 'Mary should have popular bounds [1,1] for t=0 timesteps'
    assert dataframes[1].iloc[0].component == 'Mary' and dataframes[1].iloc[0].popular == [1, 1], 'Mary should have popular bounds [1,1] for t=1 timesteps'
    assert dataframes[2].iloc[0].component == 'Mary' and dataframes[2].iloc[0].popular == [1, 1], 'Mary should have popular bounds [1,1] for t=2 timesteps'

    assert dataframes[1].iloc[1].component == 'Justin' and dataframes[1].iloc[1].popular == [1, 1], 'Justin should have popular bounds [1,1] for t=1 timesteps'
    assert dataframes[2].iloc[2].component == 'Justin' and dataframes[2].iloc[2].popular == [1, 1], 'Justin should have popular bounds [1,1] for t=2 timesteps'

    assert dataframes[2].iloc[1].component == 'John' and dataframes[2].iloc[1].popular == [1, 1], 'John should have popular bounds [1,1] for t=2 timesteps'

def test_intersect():
    # Add nodes, including agents and intersections
    g = nx.Graph()
    g.add_nodes_from(['Agent_Ped_{i}' for i in range(4)])
    g.add_nodes_from(['Agent_Car_{i}' for i in range(4)])
    g.add_nodes_from(['Intersections_{i}' for i in range(32)])
    # 1. Try Add edges, [0, 1]
    # for type in ['Ped', 'Car']:
    #     for i in range(4):
    #         for j in range(32):
    #             g.add_edge('Agent_{type}_{i}'.format(type=type, i=i), 'Intersections_{j}'.format(j=j), IsAt=0)
    # load init graph
    pr.load_graph(g)
    # 2. Add rules
    pr.add_rule('Stop(x) <- IsPedestrian(x), IsIntersection(y), IsAt(x, y), IsOccupied(y)', 'Stop_rule')
    # 3. Add facts
    for i in range(4):
        pr.add_fact(pr.Fact('agent-type-fact', 'Agent_Ped_{i}', 'IsPedestrian', [1, 1], 0, 100))
        pr.add_fact(pr.Fact('agent-type-fact', 'Agent_Car_{i}', 'IsPedestrian', [0, 0], 0, 100))

    for i in range(32):
        pr.add_fact(pr.Fact('intersection-type-fact', 'Intersections_{i}', 'IsIntersection', [1, 1], 0, 100))
        pr.add_fact(pr.Fact('intersection-occupied-fact', 'Intersections_{i}', 'IsOccupied', [0, 0], 0, 100))

    # 4. Run the program for two timesteps to see the diffusion take place
    s = time.time()
    interpretation = pr.reason(timesteps=0)
    print('Time taken: ', time.time() - s)
    dataframes = pr.filter_and_sort_nodes(interpretation, ['Stop'])
    for t, df in enumerate(dataframes):
        print(f'TIMESTEP - {t}')
        print(df)
        print()

test_intersect()