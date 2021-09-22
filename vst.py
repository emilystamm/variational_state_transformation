import pennylane as qml
from pennylane.optimize import RotosolveOptimizer
from pennylane import numpy as np
from math import log
import csv, os, time
from state_generation import num_qubits, pure_state_density_matrix

# Initialize device 
dev = qml.device("default.qubit", wires=num_qubits)
# dev = qml.device('qiskit.ibmq', wires=num_qubits, backend='ibmq_belem')

def is_non_zero_file(fpath): return os.path.isfile(fpath) and os.path.getsize(fpath) > 0        

# Variational model class
class VariationalStateTransformation():
    def __init__(self, 
            input_states,
            target_states,
            circuit_structure='linear', 
            num_layers=2, 
            iterations=1,
            step_record=1
        ):
        # Set basics 
        self.num_qubits = num_qubits
        self.num_solutions = len(target_states)
        # Circuit 
        self.num_layers = num_layers
        self.circuit_structure=circuit_structure
        # Training
        self.iterations = iterations 
        self.optimizer = RotosolveOptimizer()
        self.train_time = 0
        # Input states
        self.input_states = input_states
        # Target output states
        self.target_states = target_states
        # Parameters
        self.best_parameters = np.zeros((self.num_qubits, self.num_layers, 3))
        self.best_cost = self.cost(self.best_parameters)
        self.loss = 0
        self.step_record = step_record
        self.output_probs = None
    
    # Define layer in circuit
    def layer(self, params, layer_index):
        if self.circuit_structure == 'full':
            self.RXRZ_layer(params, layer_index)
            self.full()   
        elif self.circuit_structure == 'ddql_bc':
            self.RXRZ_layer(params, layer_index)
            self.ddql_bc(params, layer_index)
        elif self.circuit_structure == 'linear' or self.circuit_structure == 'circular':
            self.RXRZ_layer(params, layer_index)            
            self.linear_or_circular(circular=self.circuit_structure)
        elif self.circuit_structure == 'partial':
            self.RXRZ_layer(params, layer_index)
            self.partial_entanglment(layer_index)
        else:
            print("Error, invalid circuit structure.")

    # Define circuit gates
    def circuit_gates(self, params, input_state):
        # State Preparation - prepare input state 
        qml.templates.ArbitraryStatePreparation(input_state, wires=list(range(self.num_qubits)))

        # Parametrized Circuit - apply layers 
        for layer_index in range(self.num_layers):
            self.layer(params, layer_index)    
         
    # Define circuit
    def circuit(self, params, input_state, output_state = None):

        @qml.qnode(dev)
        # Measure to obtain probabilities 
        def probs_measurement(params, input_state):
            # Circuit
            self.circuit_gates(params, input_state)
            # Measurement     
            return qml.probs(wires = [i for i in range(self.num_qubits)])

        # Measure expectation value wrt output state density matrix 
        @qml.qnode(dev)
        def expval_measurement(params, input_state, output_state):
            # Circuit
            self.circuit_gates(params, input_state)
            # Measurement     
            return qml.expval(qml.Hermitian(output_state, wires=range(self.num_qubits)))

        # If output state is given, return expectation value, else return probabilities 
        if output_state is not None: return expval_measurement(params, input_state, output_state)
        else: return probs_measurement(params, input_state)

    # Cost function to minimize 
    def cost(self, params):
        self.loss = 0
        for index in range(self.num_solutions): 
            output = self.circuit(params, self.input_states[index], pure_state_density_matrix(self.target_states[index]))
            self.loss -= output
        self.loss  = self.loss  / self.num_solutions
        return self.loss

    # Train 
    def train(self):
        print("Begin training.")
        # Start timer
        start = time.time()
        costs = []
        # Initialize parameters - size num_qubits x num_layers x 3 
        #   (3 for ddql_bc, only 2 for others used)
        params = np.random.normal(0, np.pi, (self.num_qubits, self.num_layers, 3))
        params = np.array(params)
        # For each iteration i
        for i in range(self.iterations):
            # Update parameters from optimizer 
            params = self.optimizer.step(lambda params: self.cost(params), params)
            # Keep track of the best parameters 
            if self.loss < self.best_cost:
                self.best_cost = self.loss
                self.best_parameters = params 
            # Print progress and save best cost
            if i % self.step_record == self.step_record - 1:
                print("Cost after {} steps is {}".format(i + 1, self.best_cost))
                costs += [self.best_cost]
            # Calculate train time
            self.train_time = time.time() - start 
        return costs
    
    # Test parameters
    def test(self, test_probs=True):
        print("Begin testing.")
        self.outputs = [self.circuit(self.best_parameters, self.input_states[i], pure_state_density_matrix(self.target_states[i])) for i in range(self.num_solutions)]
        if test_probs: self.output_probs = [self.circuit(self.best_parameters, self.input_states[i]) for i in range(self.num_solutions)]

    # Print results
    def compare_target_to_output(self):
        print('\nResults')
        print('=' * 75)
        for i in range(self.num_solutions):
            print('Target Vector\t\t\t{}\nOutput Probability\t\t{}\nFidelity\t\t\t{}\n'.format(
                self.target_states[i], self.output_probs[i],  self.outputs[i])
            )
        print('=' * 75)
        print('Average Fidelity:\t\t{}\n'.format(np.average(self.outputs)))
        
    # Write to file
    def write(self, file):
        header_row = [
            'TrainTime', 'AvgFidelity', 'NumQubits', 'Iterations', 'EntanglementType', 'NumLayers',
            'BestParameters'
        ]
        with open(file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
            if not is_non_zero_file(file): writer.writerow(header_row)
            result_row = [
                self.train_time, - self.best_cost , self.num_qubits, self.iterations, self.circuit_structure, self.num_layers, 
                self.best_parameters
            ]
            writer.writerow(result_row)

    # Not yet used
    def negative_log_likelihood(self, output):
            epsilon = .0001
            return log(np.maximum(epsilon, output))

    # Layer of X, Z rotations
    def RXRZ_layer(self, params, layer_index):
        for i in range(self.num_qubits):
            qml.RX(params[i, layer_index, 0], wires=i)
            qml.RZ(params[i, layer_index, 1], wires=i)

    # Different entanglement circuits
    def full(self):
        for i in range(self.num_qubits - 1):
            for j in range(i):
                qml.CNOT(wires=[i, j])  

    def ddql_bc(self, params, layer_index):  
        for i in range(1, self.num_qubits - 1):
            qml.IsingXX(params[i, layer_index, 2], wires=[0, i]) 
        if layer_index < self.num_layers - 1:
            qml.RZ(params[0, layer_index, 2], wires=i)
    
    def linear_or_circular(self, circular = False):
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            # circular entanglement - addition to linear, entangle last and first
            if circular == 'circular':
                qml.CNOT(wires=[self.num_qubits - 1, 0])
    
    def partial_entanglment(self, layer_index):
        if layer_index == 0: self.linear_or_circular(circular=True)
        else:
            index = (layer_index % (self.num_qubits - 1)) + 1
            qml.CNOT(wires=[0, index])


