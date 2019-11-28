DEBUG_RUN=0 # 1=True, 0=False

python3 ../research_seed/cifar/cifar_trainer.py \
  --gpus 0 \
  --nodes 1 \
  --num-data-loaders 4 \
  --max_nb_epochs 500 \
  --debug $DEBUG_RUN \
  --adaptivity-rate 10e-4 \
  --threshold 10e-8 \
  --batch_size 50 \
  --adam-lr 0.01 \
  --decay-n-epochs 100 \
  --decay-exponential 0.1 \
  --train-val-split 0.9

sudo shutdown now