import matplotlib.pyplot as plt
import os
def  main():
    cases_tensorparallel=['combination-e8b-d8b', 'combination-e6b-d6b', 'combination-e6b-d4b', 'combination-e4b-d6b', 'combination-e4b-d4b', 'combination-e2b-d2b']
    cases_logquantization=['combination-log-e8b-d8b', 'combination-log-e6b-d6b', 'combination-log-e6b-d4b', 'combination-log-e4b-d6b', 'combination-log-e4b-d4b', 'combination-log-e2b-d2b']
    base_path_log = "/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/quantised-logarithmic"
    base_path_tensorparallel = "/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/quantised-tensorparallel"

    # Process all cases
    cases_to_process = ['combination-e8b-d8b', 'combination-e6b-d6b', 'combination-e6b-d4b', 'combination-e4b-d6b', 'combination-e4b-d4b', 'combination-e2b-d2b']
    
    for case in cases_to_process:
        i = cases_tensorparallel.index(case)
        
        # Create a new figure for each case
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows for tensor/log

        tensorparallel_path = os.path.join(base_path_tensorparallel, cases_tensorparallel[i])
        logquantization_path = os.path.join(base_path_log, cases_logquantization[i])    

        tensorparallel_images = [img for img in os.listdir(tensorparallel_path) if img.endswith('.png')][:3]
        logquantization_images = [img for img in os.listdir(logquantization_path) if img.endswith('.png')][:3]

        # Add titles for the rows
        axs[0, 1].set_title('Tensor Parallel', pad=10)
        axs[1, 1].set_title('Log Quantization', pad=10)

        # Plot the images
        for j in range(3):
            axs[0, j].imshow(plt.imread(os.path.join(tensorparallel_path, tensorparallel_images[j])))
            axs[1, j].imshow(plt.imread(os.path.join(logquantization_path, logquantization_images[j])))
            
            # Remove axes for cleaner look
            axs[0, j].axis('off')
            axs[1, j].axis('off')

        plt.tight_layout()
        fig.savefig(f"/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/quantization_scripts/graphs/comparison_{case}.png")
        plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":
    main()
