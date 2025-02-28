# modules 
import numpy as np
import pandas as pd
import torch
import torch_geometric_temporal
import concurrent.futures
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from typing import Tuple, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

def makedict(FX: List[List[float]], W: Optional[List[float]] = None, node_ids: Optional[List[str]] = None) -> dict:
    """
    주어진 데이터를 기반으로 그래프 구조를 나타내는 딕셔너리를 생성한다.
    
    Args:
        FX (List[List[float]]): 각 행이 시간 단계를 나타내고 각 열이 노드를 나타내는 특성 행렬.
        W (Optional[List[float]], optional): 엣지 가중치 목록. 제공되지 않으면 노드 간의 상관 관계가 사용된다. 기본값은 None.
        node_ids (Optional[List[str]], optional): 노드 아이디 목록. 제공되지 않으면 'node0', 'node1', ..., 'nodeN-1'로 라벨링 된다. 기본값은 None.
    
    Returns:
        dict: 다음 키들을 포함하는 딕셔너리:
            - 'edges': 노드 인덱스 쌍으로 표현된 모든 엣지의 리스트.
            - 'node_ids': 노드 아이디 목록.
            - 'weights': 엣지 가중치 목록.
            - 'FX': 입력된 특성 행렬 (시간 x 노드).
    """
    # 입력된 데이터에서 시간 단계(T)와 노드 수(N)를 가져온다
    T, N = np.array(FX).shape
    
    # 가중치(W)가 제공되지 않으면 FX의 상관 행렬을 사용하여 계산한다
    if W is None:
        # 특성 데이터의 상관 행렬을 계산하고 이를 평탄화하여 리스트로 변환한다
        W = pd.DataFrame(FX).corr().to_numpy().reshape(-1).tolist()  # 상관 관계를 가중치로 사용
    
    # 노드 아이디가 제공되지 않으면 기본값으로 'node0', 'node1', ..., 'nodeN-1'을 생성한다
    if node_ids is None:
        node_ids = ['node' + str(n) for n in range(N)]
    
    # 그래프 정보를 포함하는 딕셔너리 생성
    dct = {
        # 'edges': 모든 노드 쌍을 포함하는 엣지 리스트 (자기 자신을 포함한 엣지도 포함)
        'edges': [[i, j] for i in range(N) for j in range(N)],
        
        # 'node_ids': 노드 아이디 목록
        'node_ids': node_ids,
        
        # 'weights': 엣지 가중치 리스트 (상관 관계 또는 제공된 리스트)
        'weights': W,
        
        # 'FX': 원본 특성 행렬 (시간 x 노드)
        'FX': FX
    }
    
    return dct

class Loader(object):
    """
    한국의 기상 관측소에서 제공한 2년 동안의 시간별 태양 복사 데이터 로드 및 처리 클래스.
    정점은 44개의 도시를 나타내며, 가중치가 있는 엣지는 태양 복사 강도 간의 관계 강도를 나타낸다.
    목표 변수는 회귀 작업을 지원한다.
    
    속성:
        _dataset (dict): 'edges', 'node_ids', 'weights', 'FX' 등을 포함하는 데이터셋.
    """
    
    def __init__(self, data_dict: Dict[str, Any]):
        """
        주어진 데이터셋으로 Loader를 초기화 한다.

        Args:
            data_dict (Dict[str, Any]): 그래프 데이터와 특성을 포함하는 데이터셋.
        """
        self._dataset = data_dict
    
    def _get_edges(self) -> None:
        """
        데이터셋에서 엣지 정보를 추출하고 전치(transpose) 한다.
        """
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self) -> None:
        """
        데이터셋에서 엣지 가중치(상관 계수)를 추출한다.
        엣지 가중치는 도시 간의 관계 강도를 나타낸다.
        """
        edge_weights = np.array(self._dataset["weights"]).T
        # 필요에 따라 스케일링을 추가할 수 있다 (예: min-max 스케일링).
        self._edge_weights = edge_weights

    def _get_targets_and_features(self) -> None:
        """
        훈련을 위한 특성과 목표 값을 추출 한다.
        특성은 이전 시간 단계들을 쌓아서 만들고, 목표는 다음 시간 단계를 사용한다.
        """
        stacked_target = np.stack(self._dataset["FX"])
        self.features = np.stack([
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ])
        self.targets = np.stack([
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ])

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """
        특성, 목표, 엣지, 가중치를 포함한 `StaticGraphTemporalSignal` 데이터셋 객체를 생성하고 반환 한다.
        
        Args:
            lags (int): 특성으로 사용할 이전 시간 단계의 수. 기본값은 4이다.

        Returns:
            StaticGraphTemporalSignal: 시간적 그래프 데이터를 포함하는 데이터셋.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        
        # StaticGraphTemporalSignal 객체 생성 및 반환
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        dataset.node_ids = self._dataset['node_ids']
        return dataset

# learn 

class Learner:
    """
    학습을 위한 Learner 클래스.
    주어진 훈련 데이터셋을 기반으로 모델을 학습시키고 예측 결과를 반환한다.
    
    Attributes:
        train_dataset: 훈련 데이터셋.
        model: 학습할 모델.
        lags: 훈련 데이터셋의 특성 차원.
    """
    
    def __init__(self, model, train_dataset, dataset_name: str = None):
        """
        Learner 클래스를 초기화 한다.
        
        Args:
            model: 학습할 모델.
            train_dataset: 훈련 데이터셋.
            dataset_name (str, optional): 데이터셋 이름. 기본값은 None.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.model = model.to(self.device)  # 모델을 GPU로 이동
        self.train_dataset = train_dataset
        self.lags = torch.tensor(train_dataset.features).shape[-1]
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def learn(self, epoch: int = 10) -> None:
        """
        주어진 모델을 사용하여 훈련 데이터를 기반으로 학습을 수행한다.
        
        Args:
            epoch (int, optional): 학습할 에폭 수. 기본값은 10.
        """
        self.model.train()
        
        # 주어진 에폭 수만큼 학습을 진행한다
        for e in range(epoch):
            for t, snapshot in enumerate(self.train_dataset):
                # 데이터와 목표값을 GPU로 이동
                node_features = snapshot.x.to(self.device)  # 입력 특성 (node features)
                target_values = snapshot.y.to(self.device)  # 목표값
                edge_index = snapshot.edge_index.to(self.device)
                edge_attr = snapshot.edge_attr.to(self.device)
                
                # 모델 예측값 계산
                yt_hat = self.model(node_features, edge_index, edge_attr)
                
                # 손실 계산 (MSE)
                cost = torch.mean((yt_hat.reshape(-1) - target_values.reshape(-1))**2)
                
                # 역전파와 최적화 진행
                cost.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # 에폭 진행 상황 출력
            print('{}/{}'.format(e + 1, epoch))                
    
    def __call__(self, dataset) -> dict:
        """
        주어진 데이터셋에 대해 모델을 사용하여 예측을 수행한다.
        
        Args:
            dataset: 예측할 데이터셋.
        
        Returns:
            dict: 특성(X), 실제 값(y), 예측 값(yhat)을 포함하는 딕셔너리.
        """
        node_features = torch.tensor(dataset.features).float().to(self.device)
        target_values = torch.tensor(dataset.targets).float().to(self.device)
        
        # 각 스냅샷에 대해 모델 예측값 계산
        yhat = torch.stack([self.model(snapshot.x.to(self.device), snapshot.edge_index.to(self.device), snapshot.edge_attr.to(self.device)) for snapshot in dataset]).detach().squeeze().float()
        # 결과를 CPU로 이동시켜 반환
        return {'X': node_features.cpu(), 'y': target_values.cpu(), 'yhat': yhat.cpu()}

def split_fit_merge_stgcn(
    FX: List[List[float]], 
    train_ratio: float,    
    model: torch.nn.Module, 
    lags: int, 
    epoch: int, 
    dataset_name: str = None
) -> np.ndarray:
    """
    ST-GCN 모델을 사용하여 시간적 그래프 데이터에 대해 예측을 수행한다.
    
    Args:
        FX (List[List[float]]): 각 행이 시간 단계를 나타내고 각 열이 노드를 나타내는 특성 행렬.
        train_ratio (float): 훈련 데이터셋과 테스트 데이터셋을 나눌 비율.
        model (torch.nn.Module): 학습할 모델.
        lags (int): 특성으로 사용할 이전 시간 단계의 수.
        epoch (int): 학습할 에폭 수.
        dataset_name (str, optional): 데이터셋 이름. 기본값은 None.
    
    Returns:
        np.ndarray: 예측된 값들의 배열.

    Note:
        이 함수는 FX 전체를 학습에 사용하지 않고, train_ratio에 따라 훈련(train)과 테스트(test) 데이터셋으로 나눈 후, 
        학습 및 예측을 수행한다. 기본적인 동작 순서는 다음과 같다.

        1. 주어진 FX 데이터를 train/test로 나눔
        2. train 데이터만을 이용하여 모델 학습 진행
        3. 학습된 모델을 사용하여 train과 test 데이터에 대해 각각 예측 수행
        4. train 예측 결과와 test 예측 결과를 합쳐 최종 출력값(yhat)으로 반환        
    """    
    # 주어진 FX 데이터를 기반으로 그래프 정보를 생성한다.
    dct = makedict(FX=FX.tolist())
    
    # 데이터셋 로더 초기화
    loader = Loader(dct)
    
    # 주어진 lags 값에 맞춰 데이터셋을 준비한다.
    dataset = loader.get_dataset(lags=lags)
    
    # 훈련 데이터셋과 테스트 데이터셋으로 분할한다.
    dataset_tr, dataset_test = torch_geometric_temporal.temporal_signal_split(dataset, train_ratio=train_ratio)
    
    # Learner 객체를 생성하고 모델을 학습한다.
    lrnr = Learner(model, dataset_tr, dataset_name=dataset_name)
    lrnr.learn(epoch)
    
    # 학습된 모델을 사용해 예측값을 계산한다.
    yhat = np.array(lrnr(dataset)['yhat'])
    
    # 예측값을 lags 만큼 앞에 추가하여 반환값의 길이를 맞춘다.
    yhat = np.concatenate([np.array([list(yhat[0])]*lags), yhat], axis=0)
    # yhat이 음수일 경우 0으로 설정
    yhat[yhat < 0] = 0
    return yhat

def split_fit_merge_eptstgcn(
    FXs: Tuple[List[List[float]], List[List[float]]],  # (yU, yP)
    train_ratio: float,     
    models: Tuple[torch.nn.Module, torch.nn.Module],  # (model_u, model_p)
    lags: Tuple[int, int],  # (lags_u, lags_p)
    epochs: Tuple[int, int],  # (epochs_u, epochs_p)
    dataset_name: str = None
) -> Tuple[np.ndarray, np.ndarray]:  # 반환값: 두 개의 예측값
    
    """
    EPT + ST-GCN 모델을 사용하여 시간적 그래프 데이터에 대해 예측을 수행한다.
    
    Args:
        FX (Tuple[List[List[float]], List[List[float]]]): 두 개의 특성 행렬 (yU, yP).
        train_ratio (float): 훈련 데이터셋과 테스트 데이터셋을 나눌 비율.
        models (Tuple[torch.nn.Module, torch.nn.Module]): 두 개의 모델 (model_u, model_p).
        lags (Tuple[int, int]): 각 모델에 대한 lags 값을 포함하는 튜플 (lags_u, lags_p).
        epochs (Tuple[int, int]): 각 모델에 대한 epoch 수를 포함하는 튜플 (epochs_u, epochs_p).        
        dataset_name (str, optional): 데이터셋 이름. 기본값은 None.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 두 개의 예측값 (yUhat, yPhat).
    
    Note:
        이 함수는 FX 전체를 학습에 사용하지 않고, train_ratio에 따라 훈련(train)과 테스트(test) 데이터셋으로 나눈 후, 
        학습 및 예측을 수행한다. 기본적인 동작 순서는 다음과 같다.
    
        1. 주어진 FX 데이터를 train/test로 나눔
        2. train 데이터만을 이용하여 model_u 및 model_p 학습 진행
        3. 학습된 모델을 사용하여 train과 test 데이터에 대해 각각 예측 수행
        4. train 예측 결과와 test 예측 결과를 합쳐 최종 출력값(yUhat, yPhat)으로 반환        
        
    """
    
    # FXs를 각각 yU, yP로 분리
    yU, yP = FXs
    
    # models를 각각 model_u, model_p로 분리
    model_u, model_p = models
    
    # lags를 각각 lags_u, lags_p로 분리
    lags_u, lags_p = lags
    
    # epochs를 각각 epoch_u, epoch_p로 분리
    epoch_u, epoch_p = epochs
    
    # 병렬로 실행
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        future_u = executor.submit(split_fit_merge_stgcn, yU, train_ratio, model_u, lags_u, epoch_u, dataset_name)
        future_p = executor.submit(split_fit_merge_stgcn, yP, train_ratio, model_p, lags_p, epoch_p, dataset_name)
        
        yUhat = future_u.result()  # 첫 번째 모델 예측값
        yPhat = future_p.result()  # 두 번째 모델 예측값
    
    return yUhat, yPhat

# simulations

def save_model_results(FXs, t, regions, tr_ratio, models, lags, epochs, prefix, simulation_number):
    """
    주어진 입력을 받아서 모델을 학습하고 결과를 npy 형식으로 저장하는 함수.
    y와 models는 튜플로 전달됩니다.
    
    Args:
        FXs (tuple): (y, yU, yP) 각 모델의 입력 데이터.
        t (List): 시간적 그래프 데이터.
        regions (List): 지역 정보.        
        tr_ratio (float): 훈련 데이터와 테스트 데이터의 비율.
        models (tuple): (model, model_u, model_p) 학습할 모델들.        
        lags (tuple): (lags1, lags2, lags3) 각 모델에 대한 lag 값들.
        epochs (tuple): (epochs1, epochs2, epochs3) 각 모델에 대한 epoch 값들.
        prefix (str): 파일 이름.
        simulation_number (int): 시뮬레이션 번호.
        
    Returns:
        None
    """
    # unpacking the data_tuple and models_tuple
    y, yU, yP = FXs
    model, model_u, model_p = models

    # Constants
    total_time_steps, total_regions = len(t), len(regions)
    test_size = int(np.floor(total_time_steps * (1 - tr_ratio)))  # 테스트 데이터셋의 길이
    train_size = total_time_steps - test_size  # 훈련 데이터셋의 길이

    # lags와 epochs 튜플을 언패킹
    lags1, lags2, lags3 = lags
    epochs1, epochs2, epochs3 = epochs
    
    # 병렬로 모델 학습 및 예측 처리
    with ThreadPoolExecutor() as executor:
        future_stgcn = executor.submit(split_fit_merge_stgcn, y, tr_ratio, model, lags1, epochs1, None)
        future_eptstgcn = executor.submit(split_fit_merge_eptstgcn, (yU, yP), tr_ratio, (model_u, model_p), (lags2, lags3), (epochs2, epochs3),  None)

        # 결과 가져오기
        yhat = future_stgcn.result()
        yUhat, yPhat = future_eptstgcn.result()

    # 훈련 데이터와 테스트 데이터 분리
    y_train, y_test = y[:train_size, :], y[train_size:, :]
    yhat_train, yhat_test = yhat[:train_size, :], yhat[train_size:, :]
    yU_train, yU_test = yUhat[:train_size, :], yUhat[train_size:, :]
    yP_train, yP_test = yPhat[:train_size, :], yPhat[train_size:, :]
    
    # 훈련 데이터 및 테스트 데이터 스택 쌓기
    train_data_stacked = np.stack((yhat_train, yU_train, yP_train), axis=0)
    test_data_stacked = np.stack((yhat_test, yU_test, yP_test), axis=0)
    
    # 저장할 파일 이름 설정 (모형 이름과 시뮬레이션 번호 반영)
    filename_train = f'{prefix}_simulation{simulation_number}_train.npy'
    filename_test = f'{prefix}_simulation{simulation_number}_test.npy'
    
    # NumPy 파일로 저장
    np.save(filename_train, train_data_stacked)
    np.save(filename_test, test_data_stacked)
    
    # 저장 완료 메시지 출력
    print(f"훈련 데이터 파일 '{filename_train}'이 저장되었습니다.")
    print(f"테스트 데이터 파일 '{filename_test}'이 저장되었습니다.")