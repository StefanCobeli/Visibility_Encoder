
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


cd $path_to_interface
if [ "$path_to_interface" = "0" ]; then
    export path_to_interface="./"
    echo "Clonig repsitory to: $path_to_interface"
    echo "git clone https://github.com/rbv3/visibility-data-generator.git $path_to_interface"
else
    echo "Pulling new git commits."
    git pull
    echo "Current dirctory is:"
    # pwd
    # ls -l
fi





#################### 4. ######################
# 4. Creating / Activating the server conda envrionment
cd $path_to_server
echo "\ncurrent path is"
pwd
#Check if conda environment exists.
# Define environment and files
ENV_NAME="visibility_encoder"
# YML_FILE="environment.yml"
YML_FILE="environment_minimal.yml"

if { conda env list | grep $ENV_NAME; } >/dev/null 2>&1; then 
  echo "Conda env already exists"; 
  # conda init
  eval "$(conda shell.bash hook)"
  echo "Conda initalized"
  conda activate $ENV_NAME
  echo "\nActivate envirnmoent $ENV_NAME from $YML_FILE "
  # Update conda environment:
  # conda env update --name $ENV_NAME --file $YML_FILE --prune

else 
  # echo "doesn't exist"; 
  echo "\nCreating conda environment from $YML_FILE"
  conda env create -f $ENV_NAME  

  #Alternative manual environment creation:
  # echo "Parsing $ENV_NAME for pip-compatible packages..."
  
  # # # Extract packages from the $ENV_NAME file
  # # PACKAGES=$(grep -E "^[ ]*-[ ]*[a-zA-Z0-9._\-]+" $YML_FILE | sed 's/^[ ]*-[ ]*//')
  # # Extract only the package names (remove Conda-specific version constraints)
  # PACKAGES=$(grep -E "^[ ]*-[ ]*[a-zA-Z0-9._\-]+" $YML_FILE | sed -E 's/^[ ]*-[ ]*//' | sed -E 's/=[^=]+//g')


  # # Create a minimal Conda environment with Python
  # echo "Creating a minimal environment..."
  # conda create -n $ENV_NAME python=3.9 -y
  
  # # Activate the new environment
  # eval "$(conda shell.bash hook)"
  # conda activate $ENV_NAME

  # # Try to install each package via pip
  # echo "Installing packages using pip..."
  # for PACKAGE in $PACKAGES; do
  #   echo "Installing $PACKAGE..."
  #   if ! pip install $PACKAGE; then
  #     echo "Warning: Failed to install $PACKAGE. Skipping..."
  #   fi
  # done

  # echo "Environment creation completed with available packages."

  eval "$(conda shell.bash hook)"
  echo "Conda initalized"
  conda activate $ENV_NAME

  #For the first run make sure you install npm packages
  cd $path_to_interface
  npm install
  npm run build
  cd $path_to_server
fi

#################### 3. ######################
# 3. Running the interface server:

cd $path_to_interface
npm run dev &
server_pid=$! #catch id of process where server is running.

sleep 1

echo "npm visual interface server running..."


#################### 4. ######################
#4. Run python server
cd $path_to_server
python server.py 

sleep 1
echo "Python visual encoder computation running..."


pwd

echo

# Stop the server when the script exits
trap "kill $server_pid" EXIT

# Wait for user input to keep the script running
read -p "Press Enter to stop the server..."

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
