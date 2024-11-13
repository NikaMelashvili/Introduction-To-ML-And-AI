import numpy as np

# R matrix
R = np.matrix([
    [-1, -1, -1, -1, 0, -1, -1],
    [-1, -1, -1, 0, -1, 0, -1],
    [-1, -1, -1, 0, -1, -1, 100],
    [-1, 0, 0, -1, 0, -1, -1],
    [-1, -1, -1, 0, -1, -1, 100],
    [-1, 0, -1, -1, -1, -1, 100],
    [-1, -1, -1, -1, -1, -1, 100]])

# Q matrix
Q = np.matrix(np.zeros([7, 7]))

# Gamma (learning parameter).
gamma = 0.8

# Initial state. (Usually to be chosen at random)
initial_state = 1

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

# This function chooses at random which action to be performed within the range
# of all the available actions.
def sample_next_action(available_actions):
    next_action = int(np.random.choice(available_actions, 1)[0])
    return next_action

def update_abstraction(action_param):
    max_index = np.where(Q[action_param,] == np.max(Q[action_param,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1)[0])
    else:
        max_index = int(max_index[0])  # Ensures single-element extraction
    return max_index

# This function updates the Q matrix according to the path selected and the Q
# learning algorithm
def update(current_state_param, action_param, gamma_param):

    max_index = update_abstraction(action_param)
    max_value = Q[action_param, max_index]

    # Q learning formula
    Q[current_state_param, action_param] = R[current_state_param, action_param] + gamma_param * max_value

# Training

# Train over 10 000 iterations. (Re-iterate the process above).
for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state, action, gamma)

# Normalize the "trained" Q matrix
print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# Testing

# Goal state = 5
# Best sequence path starting from 2 > 2, 3, 1, 5
current_state = 1
steps = [current_state]

while current_state != 6:
    next_step_index = update_abstraction(current_state)
    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print("Selected path:")
print(steps)
