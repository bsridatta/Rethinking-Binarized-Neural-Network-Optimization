DEBUG_RUN=0 # 1=True, 0=False

# constant threshold 1e-6, test adativity rate 1e-2, 1e-3, 1e-4
python3 ../research_seed/cifar/cifar_trainer.py \
  --gpus 0 \
  --nodes 1 \
  --max_nb_epochs 100 \
  --debug $DEBUG_RUN \
  --adaptivity-rate 1e-2 \
  --threshold 1e-6 \
  --batch_size 50 \
  --adam-lr 0.01 \
  --decay-n-epochs 500 \
  --decay-exponential 0.1 \
  --train-val-split 0.9

python3 ../research_seed/cifar/cifar_trainer.py \
  --gpus 0 \
  --nodes 1 \
  --max_nb_epochs 100 \
  --debug $DEBUG_RUN \
  --adaptivity-rate 1e-3 \
  --threshold 1e-6 \
  --batch_size 50 \
  --adam-lr 0.01 \
  --decay-n-epochs 500 \
  --decay-exponential 0.1 \
  --train-val-split 0.9

python3 ../research_seed/cifar/cifar_trainer.py \
  --gpus 0 \
  --nodes 1 \
  --max_nb_epochs 100 \
  --debug $DEBUG_RUN \
  --adaptivity-rate 1e-4 \
  --threshold 1e-6 \
  --batch_size 50 \
  --adam-lr 0.01 \
  --decay-n-epochs 500 \
  --decay-exponential 0.1 \
  --train-val-split 0.9

# constant adaptivity rate 1e-3, test threshold 0, 1e-5, 1e-6

python3 ../research_seed/cifar/cifar_trainer.py \
  --gpus 0 \
  --nodes 1 \
  --max_nb_epochs 100 \
  --debug $DEBUG_RUN \
  --adaptivity-rate 1e-3 \
  --threshold 0 \
  --batch_size 50 \
  --adam-lr 0.01 \
  --decay-n-epochs 500 \
  --decay-exponential 0.1 \
  --train-val-split 0.9

python3 ../research_seed/cifar/cifar_trainer.py \
  --gpus 0 \
  --nodes 1 \
  --max_nb_epochs 100 \
  --debug $DEBUG_RUN \
  --adaptivity-rate 1e-3 \
  --threshold 1e-5 \
  --batch_size 50 \
  --adam-lr 0.01 \
  --decay-n-epochs 500 \
  --decay-exponential 0.1 \
  --train-val-split 0.9

python3 ../research_seed/cifar/cifar_trainer.py \
  --gpus 0 \
  --nodes 1 \
  --max_nb_epochs 100 \
  --debug $DEBUG_RUN \
  --adaptivity-rate 1e-3 \
  --threshold 1e-6 \
  --batch_size 50 \
  --adam-lr 0.01 \
  --decay-n-epochs 500 \
  --decay-exponential 0.1 \
  --train-val-split 0.9

sudo shutdown now