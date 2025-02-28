from v250224_2 import *  # 필요한 모듈 import
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
import os
import argparse

# GConvGRU를 활용한 순환 그래프 신경망 모델 정의
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

# 커맨드라인 인자 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Run RecurrentGCN simulations sequentially on a single GPU.")
    
    parser.add_argument("--lags", type=int, nargs=3, default=[4, 4, 4], help="Lags for STGCN and EPT-STGCN (e.g., --lags 4 4 4)")
    parser.add_argument("--filters", type=int, nargs=3, default=[4, 4, 4], help="Filters for STGCN and EPT-STGCN (e.g., --filters 4 4 4)")
    parser.add_argument("--epochs", type=int, nargs=3, default=[5, 5, 5], help="Epochs for STGCN and EPT-STGCN (e.g., --epochs 5 5 5)")
    parser.add_argument("--num_simulations", type=int, default=30, help="Number of simulations to run.")
    
    return parser.parse_args()

# 데이터 로드
df = pd.read_csv('data/data_eng_230710.csv')

# 주요 데이터 정의
y = df.loc[:, 'Bukchoncheon':'Gyeongju-si'].to_numpy()
yU = df.loc[:, 'Bukchoncheon_Upper':'Gyeongju-si_Upper'].to_numpy()
yP = np.divide(y, yU + 1e-10)  # 0으로 나누는 것을 방지
t = pd.to_datetime(df.loc[:, 'date'])
regions = list(df.loc[:, 'Bukchoncheon':'Gyeongju-si'].columns)

# 결과 저장 디렉토리 생성
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

# 개별 시뮬레이션 실행 함수 (순차 실행)
def run_simulation(sim_num, lags, filters, epochs):
    # 모델 초기화
    model = RecurrentGCN(node_features=lags[0], filters=filters[0])
    model_u = RecurrentGCN(node_features=lags[1], filters=filters[1])
    model_p = RecurrentGCN(node_features=lags[2], filters=filters[2])

    # 저장할 파일 이름 설정
    prefix = f"{save_dir}/DCRNN_lags[{lags[0]}-{lags[1]}-{lags[2]}]_filters[{filters[0]}-{filters[1]}-{filters[2]}]_epochs[{epochs[0]}-{epochs[1]}-{epochs[2]}]_sim{sim_num}.npy"

    # 모델 학습 및 결과 저장
    save_model_results(
        FXs=(y, yU, yP),
        t=t,
        regions=regions,
        tr_ratio=0.8,
        models=(model, model_u, model_p),
        lags=lags,
        epochs=epochs,
        simulation_number=sim_num,
        prefix=prefix,
    )

    print(f"Simulation {sim_num} 완료, 파일 저장: {prefix}")

    # GPU 메모리 정리
    del model, model_u, model_p
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 커맨드라인 인자 가져오기
    args = parse_args()
    
    num_simulations = args.num_simulations  # 실행할 시뮬레이션 개수

    print(f"Running {num_simulations} simulations sequentially with:")
    print(f"   - Lags: {args.lags}")
    print(f"   - Filters: {args.filters}")
    print(f"   - Epochs: {args.epochs}")

    # 순차적으로 실행 (하나씩 실행)
    for i in range(1, num_simulations + 1):
        run_simulation(i, args.lags, args.filters, args.epochs)

    print("모든 시뮬레이션 완료!")