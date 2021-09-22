from state_generation import create_random_state, get_target_states
from vst import VariationalStateTransformation, num_qubits
from plot import plot_cost_fn
 
if __name__ == "__main__":
    # Set parameters
    num_solutions = num_qubits
    iterations = 10
    step_record = 1
    circuit_structure = 'linear'
    num_layers = 5
    # Input states
    input_states = [create_random_state(num_qubits=num_qubits, seed=i) for i in range(num_solutions)]
    print("Input State Parameters")
    print(input_states)

    # Initialize  model 
    model = VariationalStateTransformation(
        input_states=input_states,
        target_states=get_target_states(num_qubits, num_solutions),
        iterations=iterations, 
        circuit_structure=circuit_structure,
        num_layers=num_layers,
        step_record=step_record
    )
    # Train
    results = {}
    results[circuit_structure] = model.train()
    # Test
    model.test()
    # Print
    model.compare_target_to_output()
    # Write
    model.write('results.csv')
    # Plot
    title = 'Cost Function'
    subtitle = '{} circuit, {} qubits'.format(circuit_structure, str(num_qubits))
    file = 'plots/plot_{}_circuit_{}_qubits.png'.format(circuit_structure, str(num_qubits))
    plot_cost_fn(x=list(range(step_record, iterations+1, step_record)), y=results, title=title, file=file, subtitle=subtitle)

