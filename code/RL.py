import numpy as np
import sys
from environment import Environment
from collections import defaultdict
import plotting


# def discretize_state(state):
#     Sx, Sy = round(state["satellite_position"][0], 1), round(state["satellite_position"][1], 1)
#     Vx, Vy = round(state["satellite_velocity"][0], 1), round(state["satellite_velocity"][1], 1)
#     fuel = round(state["fuel"])
    
#     debris_state = []
#     for d in state["debris_state"].values():
#         d_Sx, d_Sy = round(d["debris_positions"][0], 1), round(d["debris_positions"][1], 1)
#         d_Wx, d_Wy = round(d["debris_velocities"][0], 1), round(d["debris_velocities"][1], 1)
#         debris_state.append((d_Sx, d_Sy, d_Wx, d_Wy))

#     return (Sx, Sy, Vx, Vy, fuel, tuple(debris_state))


def discretize_state(state, bins):
    """
    Discretize a continuous state into bins.

    Args:
        state (dict): The state dictionary.
        bins (dict): A dictionary specifying the bins for each state component.

    Returns:
        tuple: A discretized state tuple.
    """
    # Discretize satellite position and velocity
    Sx = np.digitize(state["satellite_position"][0], bins["satellite_position"][0])
    Sy = np.digitize(state["satellite_position"][1], bins["satellite_position"][1])
    # Vx = np.digitize(state["satellite_velocity"][0], bins["satellite_velocity"][0])
    Vy = np.digitize(state["satellite_velocity"][1], bins["satellite_velocity"])
    fuel = np.digitize(state["fuel"], bins["fuel"])

    # Discretize debris state
    debris_state = []
    for d in state["debris_state"].values():
        d_Sx = np.digitize(d["debris_positions"][0], bins["debris_positions"][0])
        d_Sy = np.digitize(d["debris_positions"][1], bins["debris_positions"][1])
        d_Wx = np.digitize(d["debris_velocities"][0], bins["debris_velocities"][0])
        d_Wy = np.digitize(d["debris_velocities"][1], bins["debris_velocities"][1])
        debris_state.append((d_Sx, d_Sy, d_Wx, d_Wy))

    return (Sx, Sy, Vy, fuel, tuple(debris_state))

# # State discretization
# bins = {
#     "satellite_position": [np.linspace(0, 10, 20), np.linspace(-2, 2, 20)],
#     "satellite_velocity": [np.linspace(0.5, 0.5, 1), np.linspace(-3, 3, 20)],
#     "fuel": np.linspace(0, 5, 5),
#     "debris_positions": [np.linspace(0, 10, 20), np.linspace(-2, 2, 20)],
#     "debris_velocities": [np.linspace(-5, 5, 10), np.linspace(-5, 5, 10)]
# }


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        if np.max(Q[observation])==0 :
            best_action = np.random.choice(list(range(len(Q[observation]))))
        else :
            best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn




def q_learning(env: Environment, num_episodes, bins, discount_factor=0.5, alpha=0.5, epsilon=1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(len(env.action_space)))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, len(env.action_space))
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Implement this!
        current_state = env.reset()
        current_observation = discretize_state(current_state, bins)
        episode_reward = 0
        num_iter = 0

        epsilon = 1/(i_episode+1)  # Reduce ε gradually
        # epsilon = 0.99*epsilon


        while True:
            num_iter+=1

            action_index = np.random.choice(list(range(len(env.action_space))), p=policy(current_observation))
            action = env.action_space[action_index]
            new_state, reward, done, termination_condition = env.step(action)
            new_observation = discretize_state(new_state, bins)

            # print("action:", action, "discretized_state: ", current_observation, "value: ", Q[current_observation])

            #Update the Q function
            Q[current_observation][action_index] += alpha*(reward + discount_factor*max([Q[new_observation][a] for a in list(range(len(env.action_space)))]) - Q[current_observation][action_index])

            policy = make_epsilon_greedy_policy(Q, epsilon, len(env.action_space))


            episode_reward += reward
            current_state = new_state
            current_observation = new_observation
            

            if done:
                # termination_reward = (termination_condition=="Reached maximum time steps")*50 + (termination_condition=="Collision with debris")*-100 + (termination_condition=="Exceeded maximum orbit")*-0.1
                # print(termination_reward)
                stats.episode_rewards[i_episode] = episode_reward #+ termination_reward
                stats.episode_lengths[i_episode] = num_iter + 1
                break
    
    return Q, stats
