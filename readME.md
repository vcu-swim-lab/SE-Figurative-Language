# Shedding Light on Software Engineering-specific Metaphors and Idioms

This repository contains the data and code to reproduce the experiments from our paper titled "Shedding Light on Software Engineering-specific Metaphors and Idioms".


## Repository Structure

- __/annotation__: A folder contains resources related to dataset annotation and the annotated CSV file. 

- __/contrastive_learning__: Implementation of contrastive learning method. Before run RQ2 and RQ3, you must run the script from this folder and save the model weights which can be loaded later.

    - contrastive_learning.py: the script of contrastive learning.

    - Fig_Lan_Annotation.csv: contains the dataset of contrastive learning.

    - readMe.md: contains the readMe on how to run contrastive_learning.py.

- __/RQ1__: Folder that contains experiments for RQ1 on LLM figurative language interpretation using cosine similarity. 

    - run.py: the script of cosine similarity and other metrics.

    - annotated-dataset.csv: the dataset of this experiment.

    - readMe.md: contains the readMe on how to run run.py.

- __/RQ2__: Folder that contains experiments for RQ2 on improving affect analysis via fine-tuning.

    - __/github_emotion__: Folder that contains experiments for emotion classifcation.
    
        - emotion_classification.py: the script of emotion classification.

	    - github-train.csv: the train dataset of this experiment.

	    - github-test.csv: the test dataset of this experiment.

	    - readMe.md: contains the readMe on how to run emotion_classification.py.


    - __/github_incivility__: Folder that contains experiments for incivility classifcation.
    
        - incivility_classification.py: the script of incivility classification.

	    - incivility-train.csv: the train dataset of this experiment.

	    - incivility-test.csv: the test dataset of this experiment.

	    - readMe.md: contains the readMe on how to run incivility_classification.py.


- __/RQ3__: Folder that contains experiments for RQ3 on bug report prioritization.

    - bug_priority_classification.py: the script of bug priority classification.

    - priority-small-train.csv: the train dataset of this experiment.

    - priority-small-test.csv: the test dataset of this experiment.

    - readMe.md: contains the readMe on how to run bug_priority_classification.py.

- requirements.txt: Python package dependencies.

Each folder contains separate ReadME showing how to run each separate scripts.

