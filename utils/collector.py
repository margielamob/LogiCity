
def collect_trajectories(env, model, n_steps):
    # Initialize state, action, reward, and done lists
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    # Reset the environments
    obs = env.reset()

    for _ in range(n_steps):
        # Use the model to determine the action
        action, _ = model.predict(obs, deterministic=True)
        
        # Take a step in the environment
        new_obs, reward, done, infos = env.step(action)

        # Store state, action, reward, next_state, and done signal
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(new_obs)
        dones.append(done)

        # Update state
        obs = new_obs

    return states, actions, next_states, rewards, dones
