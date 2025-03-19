<div align="center">
<h1>1st Place 🏆 Solution Report of team CV for the Challenge "Core Values: A Geological Quest"</h1>
<h2>Challenge Website: <a href="https://thinkonward.com/app/c/challenges/core-values">Link</a></h2>
<h2>Team Member: Efstathios Karypidis</h2>
<h2>Contact: stathiskaripidis@gmail.com </h2>
</div>


# Short Description 
Our solution implements a two-phase methodology for geological image analysis. In the first phase, I developed a specialized neural network architecture leveraging state-of-the-art Vision Foundation Models, trained on both labeled and unlabeled datasets. The second phase employs an advanced prediction refinement pipeline that integrates the Segment Anything Model 2 (SAM2) to enhance segmentation accuracy and robustness.

# Installation 
1. Create a new conda environment and install the required packages:
```bash
conda create -n cvgq python=3.11
conda activate cvgq
pip3 install torch torchvision torchaudio
```
2. Clone and install Segment Anything Model 2 (SAM2) from the official repository:
```bash
git clone https://github.com/facebookresearch/sam2
cd sam2
pip install -e .
```
3. Sam has a problem and configs should be placed in the parent sam2 folder. Inside the parent sam2 folder, run the following command:
```bash
cp -r sam2/configs/ .
```
4. Download sam2 checkpoints
```bash
cd checkpoints
bash download_ckpts.sh
cd ../.. # Return to the root folder
```
5. Download the pretrained weights from the [release page](https://github.com/Sta8is/cvs/releases/tag/v0.1). Select the file `pretrained_models.zip` and place the checkpoints in the `checkpoints` and `final_checkpoints` folders.

# Structure
The repository is structured as follows:
```
cvs
│
├───data
│   ├───core-values-test-data
│   ├───train
│   ├───train_unlabeled
├───checkpoints
├───src
│   ├───model.py
│   ├───data.py
│   ├───transforms.py
│   ├───utils.py
│   ├───my_utils.py
├───pred_vis
├───train.py
├───train_semi.py
|───predict.py 
├───solution.ipynb
|───README.md
```
- `data`: contains the dataset folders. Download the dataset from the official challenge website and unzip the files. Note that file image001269.png from the unlabeled dataset is corrupted and is removed. 
- `checkpoints`: folder to save the trained model checkpoints.
- `final_checkpoints`: contains the the 5 trained model checkpoints used for the final submission.
- `src`: contains the source code files.
- `model.py`: contains the neural network architecture
- `train.py`: script to train the model on the labeled dataset.
- `train_semi.py`: script to train the model on both labeled and unlabeled datasets.
- `utils.py`: contains utility functions provided by the challenge organizers github repository.
- `my_utils.py`: contains utility functions implemented by me.
- `predict.py`: script to make predictions on the test dataset.
- `solution.ipynb`: Jupyter notebook containing the solution report.
- `pred_vis`: folder to save the prediction visualizations (I run the prediction script on the public test dataset and saved the predictions).
- `README.md`: this file.
# Usage
To understand and reproduce my solution, please refer to the `solution.ipynb` notebook. The notebook provides a detailed explanation of the methodology, implementation, some insights and results.

# Incremental Improvements
In order to show important incremental improvements, I provide a table with the experiments conducted during the development of the solution. The table includes the experiment id, description, public dice score, and the checkpoint file.

| Experiment Id | Description | Public Dice Score | Checkpoint |
| --- | --- | --- | --- |
| 1 | Supervised Only (10 epochs, CE loss) | 0.5569 | model_supervised.pth |
| 2 | Semi-Supervised (25 epochs, CE loss) | 0.5910 | --- |
| 3 | Semi-Supervised (50 epochs, CE loss) | 0.6058 | --- |
| 4 | Semi-Supervised (100 epochs, CE loss) | 0.6413 |  --- |
| 5 | Semi-Supervised (150 epochs, CE loss) | 0.6422 |  model_semi_blr2em5.pth |
| 6 | Semi-Supervised (150 epochs, blr 4e-5, CE loss) | 0.6441 |  model_semi.pth |
| 7 | Semi-Supervised (100 epochs, blr 4e-5, Ohem loss) | 0.6445 |  model_semi_100eps_ohem.pth |
| 8 | Semi-Supervised (150 epochs, blr 4e-5, Consistency Reg) | 0.6473 |  model_semi_creg.pth |
| 9 | Semi-Supervised (150 epochs, blr 4e-5, Combined loss) | 0.6486 |  model_semi_combloss.pth |
|10 | Ensemble of 6, 8, 9 | 0.6599 |  --- |
|11 | Ensemble of 6, 7, 8, 9 | 0.6642 |  --- |
|12 | Ensemble of 5, 6, 7, 8, 9 | 0.6655 |  --- |
|13 | Ensemble of 5, 6, 7, 8, 9, 11 + SAM2 AutoMaskGen Refine | 0.6718 |  --- |
|14 | Ensemble of 5, 6, 7, 8, 9, 11 + SAM2 AutoMaskGen Refine + Replace Confident(thresh 0.8) | 0.6868 |  --- |
|15 | Ensemble of 5, 6, 7, 8, 9, 11 + SAM2 AutoMaskGen Refine + Replace Confident(thresh 0.625) | 0.6951 |  --- |

# Candidate for Honorable Mention

- [x] Fastest GPU Inference. My algorithm performs inference on 70 images of the public test dataset in ~ 4 minutes. 
- [ ] Fastest CPU Inference. Not tested.
- [x] Best custom model that does not use a highly abstracted API. The main model architecture is implemented in PyTorch. Only for prediction refinement, I use the SAM2AutomaticMaskGenerator.
- [x] Most innovative use of unlabelled data. I use state-of-the-art semi supervised learning techniques to leverage the unlabeled data.
- [x] Best documentation. The solution report is detailed and provides a clear explanation of the methodology, implementation, and results.

