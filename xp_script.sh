
conda init zsh
conda deactivate
conda activate xai

python sklearn_train.py --lr 1e-6 --no_libelle False --substances True --hls 50  
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-7 --epochs 100000 --no_libelle False --substances True --hls 50 --ele 1000
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --no_libelle True --substances True --hls 50
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-5 --no_libelle False --substances False --hls 50
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-5 --no_libelle True --substances False --hls 50
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-6 --epochs 50000 --no_libelle False --substances True --hls 50 --ele 1000
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-7 --epochs 100000 --no_libelle False --substances True --hls 50 --ele 2000
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-5 --epochs 20000 --no_libelle False --substances True --hls 50 --tol 1e-2 --adaptive
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-4 --epochs 20000 --no_libelle False --substances True --hls 50
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-3 --epochs 20000 --no_libelle False --substances True --hls 50
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-3 --epochs 20000 --no_libelle False --substances True --hls 50 --tol 1e-2 --adaptive
killall -9 python3.8 python 
sleep 5 & wait %5

python sklearn_train.py --lr 1e-4 --epochs 20000 --no_libelle False --substances True --hls 50 --tol 1e-2 --adaptive
killall -9 python3.8 python 
sleep 5 & wait %5
