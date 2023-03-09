# sh scripts/train_dit.sh <inside|outside>

GPU_ID=0
preprocessed_data_dir=./data

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_deep_implicit_templates.py \
    -e examples/tooth_${1} \
    --debug \
    --batch_split 2 \
    -c latest \
    -d ${preprocessed_data_dir}
