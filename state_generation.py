import numpy as np
import math

# Set the number of qubits to use
num_qubits = 4

# Returns input parameters to create arbitrary pure state
def create_random_state(num_qubits, seed = 12):
    np.random.seed(seed)
    num_values = 2 ** (num_qubits + 1) - 2
    state = np.random.rand(num_values) * math.pi
    return state

# Returns target state vectors 
def get_target_states(num_qubits, num_solutions):
    # Initialize target state
    target_states= [ [0.0] * (2**num_qubits) for _ in range(num_solutions)]
    
    # Currently defining specific target states with num_qubits == num_solutions
    assert num_qubits == num_solutions

    if num_qubits == 1:
        # 1 qubit / 2 solution - for testing 
        target_states[0][0] = 1.0
    elif num_qubits == 2:
        # 2 qubits / 2 solutions - for testing 
        target_states[0][1] = 1.0
        target_states[1][3] = 1.0
    elif num_qubits == 4:
        # 4 qubits / 4 solutions 
        target_states[0][2] = 1.0
        target_states[1][4] = 1.0
        target_states[2][9] = 1.0
        target_states[3][11] = 1.0  
    else:
        print("Target states undefined for {} qubits".format(num_qubits))
    return target_states

# Inputs statevector, outputs corresponding density matrix
def pure_state_density_matrix(state_vector):
    return np.outer(state_vector, np.conjugate(state_vector))

