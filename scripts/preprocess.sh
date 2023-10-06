# sh scripts/preprocess.sh <inside|outside>

export PANGOLIN_WINDOW_URI=headless://

# preprocess datasets

python preprocess_data.py \
    --data_dir data \
    --source tooth_morphology/datasets \
    --name ToothMorphology \
    --split examples/splits/tooth_${1}_${2}.json \
    --unify_center \
    # --unify_scale
    # --skip \
