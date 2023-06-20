import re
import matplotlib.pyplot as plt


file_path  = 'outputs/Stacked_TC_Khop_GNN_cls_lr2_r0/log_train.txt'

file_path  = '/scratch/uceepdg/new_outputs_mmw/Stacked_GraphAttention_parallel_MSGGNN_Transformer_cls_lr2_r0/'

file_path = '/scratch/uceepdg/new_outputs_mmw/Stacked_GraphAttention_parallel_MSGGNN_Transformer_cls_lr2_r0_lrs05/'
file_path = '/scratch/uceepdg/new_outputs_mmw/Stacked_TC_GraphAttention_parallel_MSGGNN_Transformer_cls_lr2_r0/'

file_path  = '/scratch/uceepdg/new_outputs_mmw/Stacked_TC_GraphAttention_parallel_MSGGNN_Transformer_cls_lr2_r0_lrs05/'

# Read the text file
with open(file_path + 'log_train.txt', 'r') as file:
    content = file.read()

# Extract X and Y values using regular expressions
x_values = re.findall(r'\n(\d+)\s+eval', content)
y_values = re.findall(r'accuracy:\s([\d.]+)', content)

# Convert values to integers and floats
x_values = [int(x) for x in x_values]
y_values = [float(y) for y in y_values]

# Plot the line plot
plt.plot(x_values, y_values, marker='o')
plt.xlabel('Epoch ')
plt.ylabel('BCE Loss ')
plt.title('Test Loss')


# Save the figure as a PNG file
plt.savefig('test_curve.png')

# Display the plot
plt.show()

"""
[Test] Loss   0.476299    Accuracy: 0.782719
Precision:  0.70761355248236 
Recall:  0.752746264389068 
F1 Score: 0.7294824906964278
AUC-ROC:  0.8441552088037368
Sampled Accuracy:  0.7822289037660263
"""