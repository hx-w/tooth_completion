# sh scripts/train_dit.sh <inside|outside>

# rm -rf examples/tooth_${1}/Latent* examples/tooth_${1}/*Parameters \
#        examples/tooth_${1}/*Logs examples/tooth_${1}/TrainingMeshes* \
#        examples/tooth_${1}/code* \
#        examples/tooth_${1}/Recon*

GPU_ID=0
preprocessed_data_dir=./data

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_deep_implicit_templates.py \
    -e examples/tooth_${1} \
    --debug \
    --batch_split 2 \
    -c latest \
    -d ${preprocessed_data_dir}
