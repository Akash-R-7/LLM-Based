import os
import random
import argparse

from langchain.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

from datasets import load_dataset_builder, load_dataset

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




# Get the list of all files and directories
def get_files():
    resume_pdfs_path = "test_resumes/"
    # resume_pdfs_path = "data/data/"
    file_paths_ls = []
    for (root, dirs, file) in os.walk(resume_pdfs_path):
        for pdf in file:
            f = pdf
            file_path = root + f
            # file_path = root + "/" + f
            file_paths_ls.append(file_path)

    return file_paths_ls



def page_extractor(files_path): # PDF Parsing
    # dict of lists
    ct = 1
    pdfs_dict = {}
    for file in files_path:
        if ct % 200 == 0:
            print("200 resumes parsed.......")
        
        file_path_key = file[10:]
        pdfs_dict[file_path_key] = []
        loader = PDFMinerLoader(file)

        # Load Documents and split into chunks. Chunks are returned as Documents.
        pages = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap=0))
        for page in pages:
            data = page.page_content
            pdfs_dict[file_path_key].append(data)
        
        ct += 1

        # print('------------------------------')
    return pdfs_dict




def get_skills_and_edu(pdfs_dict: dict) -> dict:
    
    """ Extracts skills and education corresponding to each pdf """

    pdfs_relevant = {}
    for key, value in pdfs_dict.items(): # value -> list of page contents
        pdfs_relevant[key] = "" # for storing skills and education
        skill_flag = edu_flag = False
        for page in value:
            for each_line in page.split('\n'):
                # print(each_line)
                line_ls = each_line.lower().split(' ')
                
                if skill_flag and len(line_ls) <= 3 and ("experience" in line_ls or ("work" in line_ls and "history" in line_ls) or "education" in line_ls):
                    skill_flag = False
                if edu_flag and len(line_ls) <= 3 and ("experience" in line_ls or ("work" in line_ls and "history" in line_ls) or "skills" in line_ls):
                    edu_flag = False

                if skill_flag and each_line != '\n':
                    pdfs_relevant[key] += each_line + "\n"
                if edu_flag and each_line != '\n':
                    pdfs_relevant[key] += each_line + "\n"

                if len(line_ls) <= 3 and "skills" in line_ls: # Accounting for variations in heading 
                    skill_flag = True
                if len(line_ls) <= 3 and "education" in line_ls:
                    edu_flag = True
                
                
    return pdfs_relevant




def load_jd_dataset():
    dataset = load_dataset("jacob-hugging-face/job-descriptions", split="train")
    dataset.set_format(type='pandas')
    job_ds_features = dataset[:]
    return job_ds_features



def get_job_profiles(job_ds_features, num_of_jobs=5, where='top'):

    """ returns num_of_jobs job profiles from where location in dataset"""
    if where == "top":
        return job_ds_features.iloc[:num_of_jobs]
    elif where == "bottom":
        return job_ds_features.iloc[-num_of_jobs:]
    elif where == "shuffle": # random
        idx = random.choices(range(0, len(job_ds_features)), k=num_of_jobs)
        return job_ds_features.iloc[idx]
    


def make_resume_data(pdfs_relevant, text_splitter):
    resumes_data = list(pdfs_relevant.values())
    resumes_metadata = [{'doc_src': src} for src in pdfs_relevant.keys()]

    resume_documents = text_splitter.create_documents(resumes_data, metadatas=resumes_metadata)

    return resume_documents

def make_jd_data(n_jds, text_splitter):

    ## Chunking job_descriptions
    job_des = list(n_jds['job_description'])
    model_response = list(n_jds['model_response'])

    jobs_data = []
    for i in range(len(job_des)):
        needed_skills = model_response[i].split('\n')
        needed_skills = "\n".join([ns.strip(",") for ns in needed_skills][1:4])
        complete_job_des = needed_skills + "\n" + job_des[i] 
        
        jobs_data.append(complete_job_des)

    jobs_metadata = [{'company': comp, 'position': pos} for comp, pos in tuple(n_jds.apply(lambda row: (row['company_name'], row['position_title']), axis=1))]

    job_documents = text_splitter.create_documents(jobs_data, metadatas=jobs_metadata)

    return (jobs_data, jobs_metadata, job_documents)



def embed_docs(jd, resume_documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    job_embedding = embeddings.embed_documents([jd])

    query_res = [relv_data.page_content for relv_data in resume_documents]
    query_meta = [meta.metadata['doc_src'] for meta in resume_documents]
    cvs_embedding = embeddings.embed_documents(query_res)

    return (job_embedding, cvs_embedding, query_meta)



def get_similar_docs(job_embedding, cvs_embedding, query_meta):
    similarity_scores = cosine_similarity(job_embedding, cvs_embedding)[0]
    most_similar_index = np.argmax(similarity_scores)
    most_similar_resume = query_meta[most_similar_index]

    return most_similar_resume





def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_jds", required=True,
        help="number of job descriptions to match with")
    ap.add_argument("-w", "--where", required=True,
        help="where to choose job descriptions from", choices=["top", "bottom", "shuffle"])
    
    args = vars(ap.parse_args())
    # display a friendly message to the user

    files_path = get_files() # to be adjusted for all resumes

    pdfs_dict = page_extractor(files_path)
    print('Resume PDF extraction done.......')
    pdfs_relevant = get_skills_and_edu(pdfs_dict)

    job_ds_features = load_jd_dataset()
    print('Job descriptions loaded.......')
    n_jds = get_job_profiles(job_ds_features, int(args["num_jds"]), args["where"])

    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    )

    resume_docs = make_resume_data(pdfs_relevant, text_splitter)
    jd_data, jd_meta, jd_docs = make_jd_data(n_jds, text_splitter)
    print('Pre-processing done.......')
    print('Finding matching job descriptions and resumes.......')

    print('------------------------------')
    for i in range(len(jd_data)):
        job_embedding, cvs_embedding, query_meta = embed_docs(jd_data[i], resume_docs)
        most_matching_resume = get_similar_docs(job_embedding, cvs_embedding, query_meta)
        print("Most matching resume for the role of {0} at {1}: ".format(jd_meta[i]["position"], jd_meta[i]["company"]))
        print(most_matching_resume)
        print('------------------------------')

    print('Successful execution.......')

if __name__=="__main__":
    main()
