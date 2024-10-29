## Count model prams, maximum feature map,FLOPs
1. Clone the repository: `git clone https://github.com/wellyhsu/Count_params.git`.
2. Run the `install_dependencies.sh` file to create the Python 3.9 virtual environment and install some python package.
3. Run the `install_psbody.sh` file to install psbody package.
4. After into the Handmesh path, run the `python count_parameters.py`, get the result.

--------------------------------------------------------------------------------------------------------------------------
https://drive.google.com/drive/folders/1LDl4UHSGrTkbcmJf_oIuoOIHdlAXw5Vj?usp=drive_link

Due to the large size of the model file, it can't be uploaded to GitHub, so you will need to download it from the google cloud. 
After downloading, please put it in this path `/Count_params/HandMesh/mobrecon/out/`.

If we change the model in the future, we will update the Python file. 
Just download the new file, replace the old one, and change the `control` parameter in count_parameters.py to obtain the new results.

![image](https://github.com/user-attachments/assets/d12a255a-1501-4a61-b4e1-c17c763767c5)

Example Result:

![image](https://github.com/user-attachments/assets/2d956364-cd73-4a5c-b33e-49c35dea7f5b)
![image](https://github.com/user-attachments/assets/1111c9df-d359-4080-be69-e156f35f8615)
![image](https://github.com/user-attachments/assets/72f041de-95e6-450d-8155-58e0523c0465)
