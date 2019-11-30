DEBUG_RUN=0 # 1=True, 0=False

# best parameters according to paper:
#   adaptivity rate: 10^−4 = 1e-4
#         threshold: 10^−8 = 1e-8

python3 ../research_seed/cifar/cifar_trainer.py \
  --gpus 0 \
  --nodes 1 \
  --num-data-loaders 4 \
  --max_nb_epochs 500 \
  --debug $DEBUG_RUN \
  --adaptivity-rate 1e-4 \
  --threshold 1e-8 \
  --batch_size 50 \
  --adam-lr 0.01 \
  --decay-n-epochs 100 \
  --decay-exponential 0.1 \
  --train-val-split 1 \
  --save-weights-every-n 5

sudo shutdown now