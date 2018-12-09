### run plain DCCA
python mnist.py --lr 0.001 --epochs 50 --weight-interval 300 --cca-reg 0.001 --weight-dir plain-batch64-weights --log-dir plain-batch64

### run ledoit with mu gradient
python mnist.py --lr 0.001 --epochs 50 --weight-interval 300 --cca-reg -1 --weight-dir ledoit-mu-batch64-weights --log-dir ledoit-mu-batch64 --mu-gradient