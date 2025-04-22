# LORA_agnews

Usage:

#Training

python train.py

#Validation

python validate.py \
  --model_path ./trained_models/results_lora/final_model \
  --batch_size 64 \
  --output_json validation_results.json \
  --output_plot val_confusion_matrix.png

#Testing

python test.py \
  --model_path ./trained_models/results_lora/final_model \
  --test_pickle test_unlabelled.pkl \
  --output_csv my_predictions.csv
