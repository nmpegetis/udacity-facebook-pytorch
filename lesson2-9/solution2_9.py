import pandas as pd

# Set initial values w1, w2, and bias for the linear equation 3*x1+4*x2-10
w1 = 3.0
w2 = 4.0
bias = -10.0

# Inputs and outputs
misclassified_inputPoint_coords = (1, 1)
outputs = []

condition = False
times = 0
learning_rate = 0.1
is_correct_string = ''
while condition == False:
    calculated_w1 = (w1 + learning_rate * misclassified_inputPoint_coords[0])
    calculated_w2 = (w2 + learning_rate * misclassified_inputPoint_coords[1])
    calculated_bias = (bias + learning_rate * 1)
    x1 = misclassified_inputPoint_coords[0]
    x2 = misclassified_inputPoint_coords[1]
    linear_combination = calculated_w1 * x1 + calculated_w2 * x2 + calculated_bias
    output = int(linear_combination >= 0)
    condition = output
    is_correct_string = 'Yes' if condition == True else 'No'
    outputs.append([calculated_w1, calculated_w2, calculated_bias,
                    linear_combination, output, is_correct_string])
    times = times + 1
    w1 = calculated_w1
    w2 = calculated_w2
    bias = calculated_bias

# Print output
print(times)
output_frame = pd.DataFrame(outputs, columns=[
                            'Input 1', '  Input 2', '  Bias', '  Linear Combination', '  Activation Output', '  Is Correct'])
print(output_frame.to_string(index=False))
