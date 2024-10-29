## Count model prams, maximum feature map,FLOPs
1. Clone the repository: `git clone https://github.com/wellyhsu/Count_params.git`.
2. Run the `install_dependencies.sh` file to create the Python 3.9 virtual environment and install some python package.
3. Run the `install_psbody.sh` file to install psbody package.
4. After into the Handmesh path, run the `python count_parameters.py`, get the result.

--------------------------------------------------------------------------------------------------------------------------

If we change the model in the future, we’ll provide a new "model.pth" file. Just update the "model_path" in count_parameters.py to get the new results.
