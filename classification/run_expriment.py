
#from data_loader import SATDDataLoader
from utils import aggregate_results

#from evaluation import Evaluator
from xgb.xgbClassifier import main
from xgb.xgb_tuning import main_tuning
#from rf.rf_classifier import main

if __name__ == "__main__":
    
    # run XGB experiment
    main(input_path="./data/SATD_Classification_version_normalized.xlsx", output_path="./results/xgb/xgb_full_results.xlsx")
    #main_tuning(input_path="./data/SATD_Classification_version_normalized.xlsx", output_path="./results/xgb/xgb_full_results.xlsx")
    # run RF experiment
    #main(input_path="./data/SATD_Classification_version_normalized.xlsx",
    #    output_path="./results/rf/rf_full_results.xlsx"
    #)
