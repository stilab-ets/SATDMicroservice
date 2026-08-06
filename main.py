import json
import logging as log
import os

import time
from typing import List

from git import Repo

from Extractor import CommitAnalyzer
import yaml
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import logging

log.basicConfig(level=log.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
log.getLogger('pydriller').setLevel(log.WARNING)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

if __name__ == "__main__":

    # Initialize variable paths    
      
    conf_file = "./conf/conf.yml"
    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    # 1. Get all the collected projects from dataset 
    dataset = conf['dataset']
    repos_dir = conf["respos_clone"]  
    results = conf["results"] # to save outputs 
    projects = pd.read_excel(dataset)
    print(projects.columns)
    project_names = projects["project"].dropna().to_list()
    #projects = hcl["URL"].to_list()
    
       
    # Clone all projects in the dataset
    for project in set(project_names):   
            if project is None:   
                continue     
            repo_folder_path = repos_dir + project  
                              
            if not os.path.exists(repo_folder_path):                
                try:
                    Repo.clone_from(f"https://github.com/{project}.git", repo_folder_path)
                    print(f"downloading project: {project}")
                except:
                    log.info(f"unable to clone: {project}")
            else:
                log.info(f"{project} already exists.")
       
 
    # 2. identify the set of commits with these udpated files

    # 3. For each file of the commit we identify: 
    #     - type (e.g., java, py,, js), 
    #     - set of comments {text, Line}, 
    #     - FileEditType (added, MODIFIED, DELETED)
     
    languages = conf["languages"] # programming languages to analyze
    
    #until = datetime.now(timezone.utc) 
    for project in set(project_names):
        
        csv_file_name = project.replace("/", "__")
        repo_path = repos_dir + project 
        output_json =  f"./dataset/comments/comments_{csv_file_name}.json"        
       
        if not os.path.exists(output_json):
            log.info(f'++++++ Starting Launch comments extractor for {project}')
            # Record start time
            start_time = time.time()
            analyzer = CommitAnalyzer(repo_path, month=1, allowed_extensions=languages)
            analysis_results = analyzer.analyze_repository(output_json)              

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
        else:
            print('exist')
    