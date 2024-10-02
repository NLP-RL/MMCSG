## Yes, this is what I was looking for! Towards Multi-modal Medical Consultation Concern Summary Generation

The repository contains code and dataset for research article titled 'Yes, this is what I was looking for! Towards Multi-modal Medical Consultation Concern Summary Generation' published at 46th European Conference on Information Retrieval (ECIR 2024). 

### Abstract
Over the past few years, the use of the Internet for healthcare-related tasks has grown by leaps and bounds, posing a challenge in effectively managing and processing information to ensure its efficient utilization. During moments of emotional turmoil, we frequently turn to the internet as our initial source of support, choosing this over discussing our feelings with others due to the associated social stigma. In this paper, we propose a new task of multi-modal medical concern summary (MMCS) generation, which provides a short and precise summary of patients' major concerns brought up during the consultation. Nonverbal cues, such as patients' gestures and facial expressions, aid in accurately identifying patients' concerns. Doctors also consider patients' personal information, such as age and gender, in order to describe the medical condition appropriately. Motivated by the potential efficacy of patients' personal context and visual gestures, we propose a transformer-based multi-task, multi-modal intent-recognition, and medical concern summary generation (IR-MMCSG) system. Furthermore, we propose a multitasking framework for intent recognition and medical concern summary generation for doctor-patient consultations. We construct the first multi-modal medical concern summary generation (MM-MediConSummation) corpus, which includes patient-doctor consultations annotated with medical concern summaries, intents, patient personal information, doctor's recommendations, and keywords. Our experiments and analysis demonstrate (a) the significant role of patients' expressions/gestures and their personal information in intent identification and medical concern summary generation, and (b) the strong correlation between intent recognition and patients' medical concern summary generation.

![Working](https://github.com/NLP-RL/MMCSG/blob/main/MMCSG.png)

Full Paper: https://arxiv.org/abs/2401.05134

### Full Dataset Access

We provide the dataset for research and academic purposes. To request access to the dataset, please follow the instructions below:

1. **Fill Out the Request Form**: To access the corpus, you need to submit a request through our [Google Form](https://forms.gle/C5q7jDprPGsCuYcD6).

2. **Review and Approval**: After submitting the form, your request will be reviewed. If approved, you will receive an email with a link to download the dataset.

3. **Terms of Use**: By requesting access, you agree to:
    - Use the dataset solely for non-commercial, educational, and research purposes.
    - Not use the dataset for any commercial activities.
    - Attribute the creators of this resource in any works (publications, presentations, or other public dissemination) utilizing the dataset.
    - Not disseminate the dataset without prior permission from the appropriate authorities.

### Experiment 

#### Please install the dependencies (requirements.txt). To install the packages, 

    conda install --file requirements.txt

#### Training

    For medical concern summary generation: MMCSG > python IR-MMCS.py

    For doctors'summary generation: DSSG > python IR-MMCSG.py
    
#### Ablation Study

    For Ablation Study: Ablation Study > python file.py
    file (T: Text, I: image information, A: Audio, and P: Persoanlity informain)

#### Testing 

    Model weights will be saved in a folder named 'model'.

    Generated text will be saved in 'result' folder. 

### Citation Information 
If you find this code useful in your research, please consider citing:
~~~~

@inproceedings{tiwari2024yes,
  title={Yes, This Is What I Was Looking For! Towards Multi-modal Medical Consultation Concern Summary Generation},
  author={Tiwari, Abhisek and Bera, Shreyangshu and Saha, Sriparna and Bhattacharyya, Pushpak and Ghosh, Samrat},
  booktitle={European Conference on Information Retrieval},
  pages={210--225},
  year={2024},
  organization={Springer}
}


Please contact us @ abhisektiwari2014@gmail.com for any questions, suggestions, or remarks.
