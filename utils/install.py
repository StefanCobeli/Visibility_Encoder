import os

def check_folder_exists(folder_paths):

    for folder_path in folder_paths:
        if os.path.isdir(folder_path):
            # print(f"The folder '{folder_path}' exists.")
            return True, folder_path
    print(f"The interface repository was not cloned yet.")
    return False, None


import subprocess

def clone_git_repo(repo_url, target_dir):
    """Clones a Git repository"""

    # Clone the repository
    clone_cmd = ["git", "clone", repo_url, target_dir]
    subprocess.run(clone_cmd, check=True)

    # Change directory to the cloned repository
    subprocess.run(["cd", target_dir], check=True)


def npm_build_and_run(interface_dir):
    """runs npm commands."""

    # Change directory to the cloned repository
    # subprocess.run(["cd", interface_dir], check=True)

    # Run npm install
    subprocess.run(["npm", "install"], cwd=interface_dir, check=True)

    # Run npm build
    subprocess.run(["npm", "run", "build"], cwd=interface_dir, check=True)

    # Run npm run dev
    subprocess.run(["npm", "run", "dev"], cwd=interface_dir, check=True)

    


# Example usage:
interface_was_cloned, path_to_interface   = check_folder_exists([pref + "visibility-data-generator" for pref in ["./", "../", "../../"]])
# Cloning github repository:
if not(interface_was_cloned):
    path_to_interface = "./visibility-data-generator"
    git_repo          = "https://github.com/rbv3/visibility-data-generator.git"
    print(f"Cloning interface at:  \n\t{path_to_interface}")
print(f"(OK) 1. Interface cloned at: \n\t{path_to_interface}")
# Installing / Setting up conda environment:
conda_environments_str = str(subprocess.run(["conda", "info", "--envs"], capture_output=True).stdout)
environement_name = "visibility_encoder"
if environement_name not in conda_environments_str:
    print("Creating conda environement for visibility analysis:")

# Activate conda environtment:
subprocess.run(["conda", "activate", environement_name], check=True)
print(f"(OK) 2. Conda environment created and activated: \n\t{environement_name}")



subprocess.run(["python", "server.py"], check=True)
print(f"(OK) 3. python server for visiblity computation is running.")

npm_build_and_run(path_to_interface)
print(f"(OK) 4. npm Interface is Running: \n\t{path_to_interface}")


