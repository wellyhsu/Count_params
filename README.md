## Count model prams, maximum feature map,FLOPs
1. Clone the repository: `git clone https://github.com/wellyhsu/Count_params.git`.
2. Run the `install_dependencies.sh` file to create the Python 3.9 virtual environment and install some python package.
3. Run the `install_psbody.sh` file to install psbody package.
4. After into the Handmesh path, run the `python count_parameters.py`, get the result.

--------------------------------------------------------------------------------------------------------------------------
https://drive.google.com/drive/folders/1LDl4UHSGrTkbcmJf_oIuoOIHdlAXw5Vj?usp=drive_link

Due to the large size of the model file, it can't be uploaded to GitHub, so you will need to download it from the google cloud. 
After downloading, please fill in the path of the downloaded model file in "model_path" in count_parameters.py.

If we change the model in the future, we’ll provide a new "model.pth" file. Just update the "model_path" in count_parameters.py to get the new results.
