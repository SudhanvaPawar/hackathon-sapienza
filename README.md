# Hackathon Sapienza - Machine Learning Project

This repository contains the code for a machine learning project developed for the Hackathon Sapienza. The project uses a PyTorch-based Multi-Layer Perceptron (MLP) model to perform a classification task, with a special focus on the trade-offs between predictive accuracy, privacy (machine unlearning), and computational efficiency.

## Repository Structure

```
/
├─── data/                     # Contains datasets and model artifacts
│    ├─── *.csv                # Partitioned training data
│    ├─── forget_data.csv      # Data for the unlearning/forget phase
│    ├─── test_data.csv        # Test data
│    └─── model_artifact       # Pre-trained model artifact
├─── utils/                    # Python utility modules
│    ├─── functions.py         # Helper functions for data processing and training
│    └─── model.py             # Definition of the DynamicMLP model
├─── .gitignore                # Files ignored by Git
├─── data.zip                  # Compressed data archive
├─── environment.yml           # Conda environment configuration file
├─── main.py                   # Main script to run the evaluation pipeline
└─── requirements.txt          # List of dependencies for pip installation
```

## Setup and Installation

You can set up the development environment using either Conda (recommended) or pip.

### 1. Unzip the Data

First, unzip the data.zip archive to populate the data/ directory.

```bash
unzip data.zip
```

### 2. Environment Setup

#### Option A: Using Conda

This command will create a new Conda environment named hackathon-tim-env with all the necessary dependencies.

```bash
conda env create -f environment.yml
conda activate hackathon-tim-env
```

#### Option B: Using pip

If you are not using Conda, you can install the dependencies in a Python virtual environment using pip.

```bash
python3.8 -m venv hackathon-tim-env
source hackathon-tim-env/bin/activate
pip install -r requirements.txt
```

## Final Evaluation (Weighted Metric)

The final score for the hackathon is a weighted average of three distinct components, as defined in `utils/eval.py`. This score balances model performance, privacy-preservation, and efficiency.

The final score is calculated as follows:

- **45% - Precision@10** (`precision_val`): Measures the model's predictive accuracy. It checks if the true positive labels are within the top 10 highest-scored predictions.
- **45% - MIA Resistance** (`mia_auc`): Measures privacy. A Membership Inference Attack (MIA) tries to determine if a specific data point was used to train the model. An ideal, privacy-preserving model would have a MIA AUC score of 0.5, meaning the attacker's guesses are no better than random chance. The score is calculated as `1.0 - 2.0 * abs(mia_auc - 0.5)`, rewarding models with an AUC closer to 0.5.
- **10% - Execution Time** (`execution_time`): Measures the computational efficiency of the unlearning process. Faster execution is better, with a performance penalty applied if the time exceeds a set threshold.

## Submission Requirements

To submit your solution for evaluation, you must prepare a folder containing the following three files with **exact names, quantities, and extensions**:

### Required Files

1. **`execution_time.txt`**
   - Contains a single integer representing the total execution time (in seconds) of your unlearning method
   - Example: `45` or `127`

2. **`model_artifact`** (no extension)
   - A serialized pickle file containing your trained model and related metadata
   - Must include the following keys when loaded:
     - `state_dict`: The model's learned parameters
     - `architecture`: The model architecture specification
     - `best_hyperparameters`: Optimal hyperparameters used
     - `model_class_source`: Source code of the model class
   - Load using the provided utility:
     ```python
     def load_pickle(filepath):
         """Loads a pickle file supporting both standard pickle and Pandas DataFrame serialization."""
         try:
             with open(filepath, 'rb') as f:
                 payload = pickle.load(f)
             print("Artifact successfully loaded.")
             return payload
         except FileNotFoundError:
             print(f"Could not find the file at {filepath}. Please verify the path.")
             raise
     
     payload = load_pickle(artifact_path)
     state_dict = payload['state_dict']
     architecture = payload['architecture']
     best_params = payload['best_hyperparameters']
     model_class_source = payload['model_class_source']
     ```

3. **`validation_ids.csv`**
   - A CSV file containing the data points used in your validation set
   - Must have a header row with the column name: `user_id`
   - Each subsequent row contains a single user ID from your validation set
   - Example format:
     ```
     user_id
     12345
     67890
     54321
     ```

### Submission Instructions

#### First submissions
1. Create a folder with this naming convention GROUPNAME_VERSION (e.g. *TIMGROUP_V1*) containing all three files with the exact names and extensions specified above. (N.B. The dropbox name of the submitting person will be automatically appended and visible in the leaderboard. You will see *TIMGROUP_V1 Alessandro Sbandi*)-
2. Upload the folder to the submission link: **[Dropbox_link](https://www.dropbox.com/request/4pn0sf0fz0wy39vtv1o4)**, you can upload as many times as you want changing the versions to check your score.
3. Ensure all file names and extensions match exactly as specified.
4. Your submission will be automatically evaluated using these files, the leaderboard link is at the end of the README.

#### Final submissions
1. Create a folder with this naming convention GROUPNAME (e.g. *TIMGROUP*) containing all three files with the exact names and extensions specified above. (N.B. The dropbox name of the submitting person will be automatically appended and visible in the leaderboard. You will see *TIMGROUP Alessandro Sbandi*)-
2. Upload the folder to the submission link, which will be shared in this read me on the final competition day: [PLACEHOLDER], you can only upload ONCE, please inform us immediately if you mistakenly uploaded more than once.
3. Ensure all file names and extensions match exactly as specified.
4. Your submission will be automatically evaluated using these files, the leaderboard link is the same as the temporary one, and will be updated on the final day.

### Important Notes

- File names are case-sensitive and must match exactly
- Do not include additional files in your submission folder
- The execution time must be a valid integer (seconds)
- All user IDs in `validation_ids.csv` must correspond to actual data points in your validation set
- The `model_artifact` file will be loaded using pickle; ensure it is properly serialized

## Usage

The main script for the project is `main.py`. This script performs the following steps:

1. Loads the partitioned data from the `data/` folder
2. Preprocesses the data (handles missing values)
3. Loads a pre-trained DynamicMLP model from the artifact at `data/model_artifact`
4. Evaluates the model's performance on the training data
5. Runs the final evaluation pipeline, which calculates the weighted score summary (currently uses placeholder data)

To run the script:

```bash
python main.py
```

**Note:** The `main.py` script contains hardcoded sections and placeholders, particularly for splitting data into validation, test, and "forget" sets. These sections must be completed to run a full evaluation pipeline.


### Real-Time Leaderboard
# From today (20th of July to 21st of July)
Here you can always submit and check the leaderboard status.

**https://hackaton-sapienza-2e1ed445-dba5-42f6-8ad8-ce6e4cf-zhqvqvw7tq-ez.a.run.app/**

### The last day, the last submission (22nd of July)
You can submit ONCE and check the FINAL leaderboard, as described previously.

## Additional information

We expect each group to share their code as a github repository in the slack channel and we expect to find in the codebase:
- utility functions to calculate the evaluating metrics
- a main.py script which we can use to reproduce your results
- requirements files to reproduce the coding environment

We will run the code of the first three groups to ensure the equality of the submitted results.

## Contacts

For any doubt, feel free to contact us at: alessandro.sbandi@telecomitalia.it, marcoantonio.desideri@telecomitalia.it, mariadiana.calugaru@telecomitalia.it
