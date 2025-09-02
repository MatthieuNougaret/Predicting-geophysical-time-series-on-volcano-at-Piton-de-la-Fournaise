clean_all:
	rm -rf models_seismicity/*
	rm -rf models_gnss/*
	rm -rf models_both/*

train_all:
	python train.py --PATH ./models_gnss --DATASET gnss
	python train.py --PATH ./models_seismicity --DATASET seismicity
	python train.py --PATH ./models_both --DATASET both

train_gnss:
	python train.py --PATH ./models_gnss --DATASET gnss

train_seis:
	python train.py --PATH ./models_seismicity --DATASET seismicity

train_both:
	python train.py --PATH ./models_both --DATASET both

opt_all:
	python optuna_hyperopt.py --DATASET gnss --NTRIALS 200
	python optuna_hyperopt.py --DATASET seismicity --NTRIALS 200
	python optuna_hyperopt.py --DATASET both --NTRIALS 200

opt_gnss:
	python optuna_hyperopt.py --DATASET gnss --NTRIALS 50

opt_seis:
	python optuna_hyperopt.py --DATASET seismicity --NTRIALS 50

opt_both:
	python optuna_hyperopt.py --DATASET both --NTRIALS 50

compile_runs:
	python compile_runs.py --PATH ./models_seismicity --DATASET seismicity
	python compile_runs.py --PATH ./models_gnss --DATASET gnss
	python compile_runs.py --PATH ./models_both --DATASET both

generate_tables:
	python generate_tables.py

copy_figs:
	cp ./models_seismicity/seismicity_R2_features.pdf ./selected_figures/seismicity_R2_features.pdf
	cp ./models_seismicity/seismicity_R2_future.pdf ./selected_figures/seismicity_R2_future.pdf
	cp ./models_gnss/gnss_R2_features.pdf ./selected_figures/gnss_R2_features.pdf
	cp ./models_gnss/gnss_R2_future.pdf ./selected_figures/gnss_R2_future.pdf
	cp ./models_both/both_R2_future.pdf ./selected_figures/both_R2_future.pdf