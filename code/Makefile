# Update the dataframe containing all data from AR6 phase

../data/processed/AU0.csv : ../data/raw/AU0 build_AU0_dataframe.py drrdTools.py
	python3 build_AU0_dataframe.py

.PHONY : backup
backup :
	rsync -varzub ../* /home/mbreyes/Dropbox/drrd_alfe_backup/
