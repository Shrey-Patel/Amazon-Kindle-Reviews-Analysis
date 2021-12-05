❖ llr.py - 		the script to measure Log Likelihood Ratio
❖ project_fns.py - 	holds functions used for the main project
❖ project.py - 		main file

Usage: 
project.py [-h]
	[--use_sklearn_features]
	[--test_size TEST_SIZE]
	[--num_folds NUM_FOLDS] [--no_stratify] [--seed SEED]
	[--plot_metrics]
	[--num_most_informative NUM_MOST_INFORMATIVE]
	[--infile INFILE]


Decriptions for Optional Arguments:
-h, --help					show this help message and exit
--use_sklearn_features 				Use sklearn's feature extraction
--test_size TEST_SIZE 				Proportion (from 0 to 1) of items held out for final testing
--num_folds NUM_FOLDS				Number of folds for cross-validation (use 2 for just a train/test split)
--no_stratify					Use stratified rather than plain cross-validation
--seed SEED 					Random seed
--plot_metrics 					Generate figures for evaluation
--num_most_informative NUM_MOST_INFORMATIVE	Number of most-informative features to show
--infile INFILE 				Input .csv file


To run the code:

python project.py [--optional_arguments] 
