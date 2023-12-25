from z3 import *
import time

def logicity_demo():
    # Define 8 agents and 32 intersections
    Agent = DeclareSort('Agents')
    Intersection = DeclareSort('Intersections')

    agents = [Const(f'agent_{i}', Agent) for i in range(8)]
    intersections = [Const(f'intersection_{i}', Intersection) for i in range(32)]
    dummyAgent = Const('dummyAgent', Agent)
    dummyIntersection = Const('dummyIntersection', Intersection)

    # Define predicates
    IsPedestrian = Function('IsPedestrian', Agent, BoolSort())
    IsInterCarEmpty = Function('IsInterCarEmpty', Intersection, BoolSort())
    IsAt = Function('IsAt', Agent, Intersection, BoolSort())
    Stop = Function('Stop', Agent, BoolSort())

    # Initialize solver
    strt = time.time()
    s = Solver()

    # Define the rule
    for agent in agents:
        rule = Exists([dummyIntersection], \
                        And(IsPedestrian(agent), Not(IsInterCarEmpty(dummyIntersection)), IsAt(agent, dummyIntersection))
                                    ) == Stop(agent)
        s.add(rule)
    # rule = ForAll([dummyAgent, dummyIntersection], \
    #               Implies(And(IsPedestrian(dummyAgent), Not(IsInterCarEmpty(dummyIntersection)), IsAt(dummyAgent, dummyIntersection)), \
    #                       Stop(dummyAgent)
    #                      )
    #              )

    # rule = ForAll([dummyAgent, dummyIntersection], \
    #               Exists([dummyIntersection], \
    #                         And(IsPedestrian(dummyAgent), Not(IsInterCarEmpty(dummyIntersection)), IsAt(dummyAgent, dummyIntersection)) == \
    #                         Stop(dummyAgent)
    #                         )
    #                 )
    print("Time: ", time.time() - strt)
    s.add(rule)
    # Specify conditions, IsPedestrian, IsIntersect, IsInterCarEmpty
    for i in range(4):
        s.add(IsPedestrian(agents[i]))  # Agents 1-3 are pedestrians

    for i in range(4, 8):
        s.add(Not(IsPedestrian(agents[i])))  # Agents 3-8 are not pedestrians

    for i in range(31):
        s.add(IsInterCarEmpty(intersections[i]))  # Intersections 1-32 are not empty

    s.add(Not(IsInterCarEmpty(intersections[31])))  # Intersections 32 is not empty

    # Specify IsAt
    for agent in agents[1:]:
        for intersection in intersections:
            s.add(Not(IsAt(agent, intersection)))

    s.add(IsAt(agents[0], intersections[31]))
    for intersection in intersections[:-1]:
        s.add(Not(IsAt(agents[0], intersection)))
    # s.add(Not(Stop(agents[2])))
    # s.add(Stop(agents[0]))
    # Check if there is a solution
    if s.check() == sat:
        m = s.model()
        for agent in agents:
            if m.evaluate(Stop(agent)):
                print(f"Agent {agent} must stop")
        print("Time: ", time.time() - strt)
    else:
        print("No solution found")

def stack_demo():
    A = DeclareSort('A')
    B = DeclareSort('B')
    C = DeclareSort('C')

    p = Function('p', A, B, C, BoolSort())
    q = Function('q', A, B, C, BoolSort())

    dummyA = Const('a', A)
    dummyB = Const('b', B)
    dummyC = Const('c', C)

    teaches = Function('teaches', A, B, BoolSort())
    constraint1 = ForAll([dummyA, dummyB], teaches(dummyA, dummyB) == Exists([dummyC], \
        Or(p(dummyA, dummyB, dummyC), q(dummyA, dummyB, dummyC))))

    s = Solver()
    s.add(constraint1)
    print(s.check())
    print(s.model())

def stack_demo_extended():
    # Sort Declarations
    A = DeclareSort('A')
    B = DeclareSort('B')
    C = DeclareSort('C')

    # Function Declarations
    p = Function('p', A, B, C, BoolSort())
    q = Function('q', A, B, C, BoolSort())
    teaches = Function('teaches', A, B, BoolSort())

    # Declare specific entities
    a1 = Const('a1', A)
    a2 = Const('a2', A)
    b1 = Const('b1', B)
    b2 = Const('b2', B)
    c1 = Const('c1', C)
    c2 = Const('c2', C)
    # Variables for quantified expression
    dummyA = Const('dummyA', A)
    dummyB = Const('dummyB', B)
    dummyC = Const('dummyC', C)

    # Solver Initialization
    s = Solver()

    # Specify Grounded Predicates
    # Example: Setting some predicates to be true or false for specific entities
    s.add(p(a1, b1, c1))
    s.add(Not(p(a2, b2, c2)))
    s.add(q(a1, b2, c1))
    s.add(Not(q(a2, b1, c2)))

    # Constraint Definition
    constraint1 = ForAll([dummyA, dummyB], teaches(dummyA, dummyB) == Exists([dummyC], \
        Or(p(dummyA, dummyB, dummyC), q(dummyA, dummyB, dummyC))))
    s.add(constraint1)

    # Check Satisfiability and Print the Model
    print(s.check())
    if s.check() == sat:
        print(s.model())


def learn_z3():
    Agent = DeclareSort('Agents')
    Intersection = DeclareSort('Intersections')

    agents = [Const(f'agent_{i}', Agent) for i in range(2)]
    intersections = [Const(f'intersection_{i}', Intersection) for i in range(2)]

    dummyAgent = Const('dummyAgent', Agent)
    dummyIntersection = Const('dummyIntersection', Intersection)

    IsAt = Function('IsAt', Agent, Intersection, BoolSort())
    IsPedestrian = Function('IsPedestrian', Agent, BoolSort())
    IsInterCarEmpty = Function('IsInterCarEmpty', Intersection, BoolSort())
    Stop = Function('Stop', Agent, BoolSort())

    s = Solver()
    s.add(IsPedestrian(agents[0]))
    s.add(IsPedestrian(agents[1]))

    s.add(Not(IsInterCarEmpty(intersections[0])))
    s.add(IsInterCarEmpty(intersections[1]))

    
    s.add(IsAt(agents[0], intersections[0]))
    s.add(IsAt(agents[1], intersections[1]))
    s.add(Not(IsAt(agents[0], intersections[1])))
    s.add(Not(IsAt(agents[1], intersections[0])))
    # s.add(Not(Stop(agents[1])))
    # s.add(Not(IsAt(agents[1], intersections[1])))

    # 1. Forall + Implies
    # s.add(ForAll([dummyAgent, dummyIntersection], \
    #              Implies(And(IsPedestrian(dummyAgent), IsAt(dummyAgent, dummyIntersection), Not(IsInterCarEmpty(dummyIntersection))
    #                          )
    #                      , Stop(dummyAgent)
    #                     )   
    #              )
    #      )
    
    # 2. Exists + ==
    s.add(ForAll([dummyAgent], \
                 Stop(dummyAgent) == Exists([dummyIntersection], \
                                            And(IsPedestrian(dummyAgent), IsAt(dummyAgent, dummyIntersection), Not(IsInterCarEmpty(dummyIntersection)))
                                            )
                 )
         )
    
    
    if s.check() == sat:
        m = s.model()
        for agent in agents:
            if m.evaluate(Stop(agent)):
                print(f"Agent {agent} must stop")
    else:
        print("No solution found")  


if __name__ == '__main__':
    logicity_demo()
    # stack_demo()
    # stack_demo_extended()