import os
import subprocess
import concurrent.futures

def run_command(command):
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)

def run_linear_probing(directory, script_name, max_workers=4):
    commands = []
    filenames = sorted(os.listdir(directory))
    for filename in filenames:
        if filename.endswith(".pt"):
            pretrained_model_path = os.path.join(directory, filename)
            command = [
                "python",
                script_name,
                "--pretrained_model_path", pretrained_model_path,
                "--linear_probing"
            ]
            commands.append(command)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_command, commands)

# Define the directory containing the files and the script name
directory = './mae_pretrained_model'
script_name = 'train_classifier.py'

# Define the number of concurrent workers (adjust based on your GPU memory)
max_workers = 21

run_linear_probing(directory, script_name, max_workers)