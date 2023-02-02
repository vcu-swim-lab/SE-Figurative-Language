# Paper: Shedding Light on Software Engineering-specific Metaphors and Idioms
In the following, we briefly describe the different components that are included in this project and the software required to run the experiments.

## Project Structure
The project includes the following files and folders:

  - __/dataset__: A folder that contains inputs that are used for the experiments.
    - annotated-dataset.csv: CSV file that contains 1248 annotated GitHub instances that contains figurative expressions.
    - emotion-train.csv: Train subset of emotion dataset.
    - emotion-train.csv: Test subset of emotion dataset.
    - sentiment-train-2.csv: Train subset of sentiment dataset.
    - sentiment-train-2.csv: Test subset of sentiment dataset.
    - civility-train.csv: Train subset of civility dataset.
    - civility-train.csv: Test subset of civility dataset.
 - __/annotation instructions__: A folder that contains the annotation related instructions.
    - Instructions for Annotation.pdf: contains the annotation instructions for metaphors and idioms verification and EMS constructions.
    - Instructions for Annotation (Different).pdf: contains the annotation instructions for DMS constructions.
 - __/RQ2__: A folder that contains the scripts regarding RQ2.
 - __/RQ3__: A folder that contains the scripts regarding RQ3.
 - requriments.txt: contains the The python libraries used in this experiment.


## Setup
1. setup virtual environment and activate it
2. `pip install -r requirements.txt'


## Running Experiments

### Running RQ2 scipt: 
`python run.py datapath model type`

Here, datapath = `dataset/annotated-dataset.csv`
model = `bert-base-uncased` or `roberta-base` or `jeniya/BERTOverflow`
type = `SE` or `General` or `any`


### Running RQ3 scipts: 
First, you need to run `contrastive_learning.py` for fine-tuning using contrastive learning and save the fine-tined model. The command is:

`python contrastive_learning.py datapath model model_savepath`

Here, datapath = `dataset/annotated-dataset.csv`
model = `bert-base-uncased` or `roberta-base` or `jeniya/BERTOverflow`
model_savepath = `saved model file name`

Then, you can run `finetune_llm.py` with or without loading the saved fine-tuned contrastive model. The command when running with/without contrastive learning model:
`python finetune_llm.py col model constrastive_flag constrastive_path data_type`

Here, col = `Anger` or `Joy` or, `Fear` or `Love` or `Sadness` or `Surprise` when data_type = `emotion`

col = `Positive` or `Negative` or, `Neutral` when data_type = `sentiment`

col = `Positive` or `Negative` or, `Neutral` or `Uncivil` when data_type = `civility`

constrastive_flag = `1` when you want to load the saved fine-tuned contrastived model, otherwise `0`

constrastive_path = `saved file path name` or empty string ("") based on constrastive_flag value.


