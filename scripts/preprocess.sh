# sh scripts/preprocess.sh <inside|outside>

export PANGOLIN_WINDOW_URI=headless://

# preprocess datasets

python3 preprocess_data.py --data_dir data --source tooth_morphology/datasets --name ToothMorphology --split examples/splits/tooth_${1}_train.json --skip
