
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e


#################### 1. ######################
# #1. Cloning interface github repository: if it does not exist:
export flag=0
export path_to_interface=0
directories=("./visibility-data-generator" "../visibility-data-generator" "../../visibility-data-generator")
servers=("../" "../Visibility_Encoder" "../../Visibility_Encoder")
# export dir="visibility-data-generator"

# Check if each directory exists
for dir in "${!directories[@]}"; do
  if [ -d "${directories[$dir]}" ]; then
    export path_to_interface="${directories[$dir]}"
    export path_to_server="${servers[$dir]}"
    export flag=1

    echo "Found interfacxe in $path_to_interface"
    echo "Relative path to server is $path_to_server"

  else
    echo "Directory ${directories[$dir]} does not exist."
  fi
done

echo

if [ "$path_to_interface" = "0" ]; then
    export path_to_interface="./"
    echo "Clonig repsitory to: $path_to_interface"
    echo "git clone https://github.com/rbv3/visibility-data-generator.git $path_to_interface"
else
    echo "Pulling new git commits."
    cd $path_to_interface
    git pull
    echo "Current dirctory is:"
    # pwd
    # ls -l
fi


#################### 3. ######################
# 3. Running the interface server:
cd $path_to_interface
# npm install
# npm run build
# npm run dev &

sleep 1

echo "npm visual interface server running..."


#################### 4. ######################
# 4. Creating / Activating the server conda envrionment
cd $path_to_server
echo "\ncurrent path is"
pwd
#Check if conda environment exists.

if { conda env list | grep 'visibility_encoder'; } >/dev/null 2>&1; then 
  echo "Conda env already exists"; 
  # conda init
  eval "$(conda shell.bash hook)"
  echo "Conda initalized"
  conda activate visibility_encoder
  # Update conda environment:
  conda env update --name visibility_encoder --file environment.yml --prune

else 
  echo "doesn't exist"; 
  echo "Creating conda environment from environment.yml"
  conda env create -f environment.yml  
  eval "$(conda shell.bash hook)"
  echo "Conda initalized"
  conda activate visibility_encoder
fi



#Run python server
# python server.py 

sleep 1
echo "Python visual encoder computation running..."


cd $path_to_server
pwd

echo

# # Initialize conda
# echo "Initializing conda..."
# conda init bash

# # Activate the conda environment
# echo "Activating the conda environment..."
# conda activate env_name

# # Run the Python install script
# echo "Running the Python install script..."
# python install.py

# echo "Setup complete!"



# # Example usage:
# interface_was_cloned, path_to_interface   = check_folder_exists([pref + "visibility-data-generator" for pref in ["./", "../", "../../"]])
# #1. Cloning interface github repository:
# if not(interface_was_cloned):
#     path_to_interface = "./visibility-data-generator"
#     git_repo          = "https://github.com/rbv3/visibility-data-generator.git"
#     print(f"Cloning interface at:  \n\t{path_to_interface}")
# print(f"(OK) 1. Interface cloned at: \n\t{path_to_interface}")
# # Installing / Setting up conda environment:
# conda_environments_str = str(subprocess.run(["conda", "info", "--envs"], capture_output=True).stdout)
# environement_name = "visibility_encoder"
# if environement_name not in conda_environments_str:
#     print("Creating conda environement for visibility analysis:")

# # Activate conda environtment:
# subprocess.run(["conda", "activate", environement_name], check=True)
# print(f"(OK) 2. Conda environment created and activated: \n\t{environement_name}")



# subprocess.run(["python", "server.py"], check=True)
# print(f"(OK) 3. python server for visiblity computation is running.")

# npm_build_and_run(path_to_interface)
# print(f"(OK) 4. npm Interface is Running: \n\t{path_to_interface}")
