# Usage

```
# Distillation training
python train_distill.py

# Evaluate LSTM baseline
python test.py --model_type lstm --model_path ./MODEL

# Evaluate student after training
python test.py --model_type student --model_path ./STUDENT_MODEL
```