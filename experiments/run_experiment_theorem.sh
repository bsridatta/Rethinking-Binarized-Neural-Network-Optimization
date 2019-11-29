python3 ../bytorch/experiments/with_larq/claims/larq_theorem_cifar.py \
  --lr 0.01 \
  --epochs 1 \
  --optim SGD \
  --init glorot_uniform

python3 ../bytorch/experiments/with_larq/claims/larq_theorem_cifar.py \
  --lr 1 \
  --epochs 1 \
  --optim SGD \
  --init glorot_uniform

python3 ../bytorch/experiments/with_larq/claims/larq_theorem_cifar.py \
  --lr 0.01 \
  --epochs 1 \
  --optim SGD \
  --init scaled_glorot_uniform

python3 ../bytorch/experiments/with_larq/claims/larq_theorem_cifar.py \
  --lr 0.01 \
  --epochs 1 \
  --optim SGD \
  --init random_uniform

python3 ../bytorch/experiments/with_larq/claims/larq_theorem_cifar.py \
  --lr 0.01 \
  --epochs 1 \
  --optim Adam \
  --init random_uniform

sudo shutdown now