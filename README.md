## Resume-Matching

#### The follwing repository matches resume from various categories to given job descriptions. 

### Datasets:
#### Resume Dataset: [Kaggle Resume Dataset]("https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset")
#### Job Descriptions: [Job Description Dataset]("https://huggingface.co/datasets/jacob-hugging-face/job-descriptions/viewer/default/train?row=0")

### Task Action
* <strong>PDF Extraction</strong>: A PDF extraction tool is used to parse the PDF storing the metadata associated with it.
* <strong>Job Description</strong>: Job Descriptions are chosen from the above mentioned dataset.
* <strong>Candidate Job Matching</strong>: Given n job descriptions find the most relevant resume among available candidates.

### Running the program
#### Download the resume datasets from the mentioned link.
#### Install the requirements.txt file. Python version used: 3.9.17
```
pip install requirements.txt
```
#### Run the following command 
```
python ./src/resume_match.py -n 2 -w bottom
```
#### Flag -n (integer) specify the number of job descriptions to test against and -w (str) denote the location in dataset to choose job descriptions from, choices are [top, bottom, shuffle]. 


### Note:
#### Parsing all the resumes (2484) may take from 10-20 minutes in an Intel-i5 CPU. 'test_resumes' folder contains one randomly sampled resume from each job category and can be used for quicker execution. 
