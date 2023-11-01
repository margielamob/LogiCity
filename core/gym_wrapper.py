# reward definition
# 1) absolute distance to the goal
# 2) penalty off street (different agent different reward...1 agent first)
# 3) penalty of not stopping at the stopsign? 

# S -> A_1, A_2, ...A_n? -----> S' (R)
# 1 ----------------- 2
# 1 --|           |--- 2
#     |           |
#     |           |
#     |___________|
# T(s'|s, a)

# offline RL: off-policy evaluation
# online
    # Mmodel-based
    # Model-free
        # on-policy: Policy Gradient (REINFORCE ---> TRPO ---> PPO V(s))
        # off-policy: Q Learning + Policy Improvement
    
