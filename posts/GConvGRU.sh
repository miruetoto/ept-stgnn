python r250224_GConvGRU.py --lags 4 4 4 --filters 8 8 8 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 8 4 8 --filters 8 8 8 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 12 4 12 --filters 8 8 8 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 16 4 16 --filters 8 8 8 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 20 4 20 --filters 8 8 8 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 24 4 24 --filters 8 8 8 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 28 4 28 --filters 8 8 8 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 28 4 28 --filters 8 8 8 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 4 4 4 --filters 16 16 16 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 8 4 8 --filters 16 16 16 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 12 4 12 --filters 16 16 16 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 16 4 16 --filters 16 16 16 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 20 4 20 --filters 16 16 16 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 24 4 24 --filters 16 16 16 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 28 4 28 --filters 16 16 16 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 28 4 28 --filters 16 16 16 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 4 4 4 --filters 24 24 24 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 8 4 8 --filters 24 24 24 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 12 4 12 --filters 24 24 24 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 16 4 16 --filters 24 24 24 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 20 4 20 --filters 24 24 24 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 24 4 24 --filters 24 24 24 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 28 4 28 --filters 24 24 24 --epochs 5 5 5 --num_simulations 20 &
python r250224_GConvGRU.py --lags 28 4 28 --filters 24 24 24 --epochs 5 5 5 --num_simulations 20 &

wait  # 모든 프로세스가 종료될 때까지 대기
echo "모든 실행이 완료됨!"