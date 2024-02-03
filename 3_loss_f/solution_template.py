import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        print('shapes - ', self.X_train.shape, self.ys_train.shape, 
              self.X_test.shape, self.ys_test.shape)
        self.num_input_features = self.X_train.shape[1]
        print('num_input_features - ', self.num_input_features)
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train, self.key_query_ids_train, self.index_query_ids_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test, self.key_query_ids_test, self.index_query_ids_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test) 
        
        print('key_query_ids_train shape - ', self.key_query_ids_train.shape)
        print('key_query_ids_test shape - ', self.key_query_ids_test.shape)
        print('index_query_ids_train shape - ', self.index_query_ids_train.shape)
        print('index_query_ids_test shape - ', self.index_query_ids_test.shape)
        print('query_ids_train shape - ', self.query_ids_train.shape)
        print('query_ids_test shape - ', self.query_ids_test.shape)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        key_query_ids, index_query_ids = np.unique(inp_query_ids, return_inverse=True)
        print('len(key_query_ids)', str(len(key_query_ids)))
        # 0 итерация
        batch = inp_feat_array[np.where(index_query_ids==0)]
        scaled_X = StandardScaler().fit_transform(batch)

        for i in range(len(key_query_ids) - 1):
            batch = inp_feat_array[np.where(index_query_ids==i+1)]
            scaled_batch = StandardScaler().fit_transform(batch)
            scaled_X = np.concatenate((scaled_X, scaled_batch), axis=0)
        return [scaled_X, key_query_ids, index_query_ids]

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(num_input_features=listnet_num_input_features, 
                      hidden_dim=listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        # допишите ваш код здесь
        result = []
        for epoch in range(self.n_epochs):
            self._train_one_epoch()
            ndcgs = self._eval_test_set()
            print('epoch number - ', epoch)
            print(f"nDCG: {ndcgs:.4f}")
            result.append(ndcgs)
        return result

    def _calc_loss(self, batch_ys: torch.FloatTensor, batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        P_ys = torch.softmax(batch_ys, dim=0)
        P_pred = torch.softmax(batch_pred, dim=0)
        #return -torch.sum(P_ys * torch.log(P_pred))
        #print('P_ys.shape, P_pred.shape - ', P_ys.shape, P_pred.shape)
        #print(-torch.sum(P_ys * torch.log(P_pred/P_ys)))
        return -torch.sum(P_ys * torch.log(P_pred/P_ys))

    def _train_one_epoch(self) -> None:
        self.model.train()
        # допишите ваш код здесь
        #for i in range(len(self.key_query_ids_train)):
        rand_idx = torch.randperm(len(self.key_query_ids_train))
        for i in range(len(self.key_query_ids_train)):
            mask = np.where(self.query_ids_train==self.key_query_ids_train[i])
            # batch_X = self.X_train[np.where(self.index_query_ids_train==i)]
            # batch_y = self.ys_train[np.where(self.index_query_ids_train==i)]
            batch_X = self.X_train[mask]
            batch_y = self.ys_train[mask]
            self.optimizer.zero_grad()
            batch_pred = torch.reshape(self.model(batch_X), (-1,))
            #print(batch_y.shape, batch_pred.shape)
            assert(batch_y.shape == batch_pred.shape, 
                   'Ошибка в train\nВектора разного размера\nbatch_y - ', str(batch_y.shape),
                   '\nbatch_pred - ', str(batch_pred.shape))
            batch_loss = self._calc_loss(batch_y, batch_pred)
            #print(batch_loss)
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            # допишите ваш код здесь
            try:
                for i in range(len(self.key_query_ids_test)):
                    batch_X = self.X_test[np.where(self.index_query_ids_test==i)]
                
                    batch_y = self.ys_test[np.where(self.index_query_ids_test==i)]
                    #batch_pred = self.model(batch_X)
                    batch_pred = torch.reshape(self.model(batch_X), (-1,))
                    #print('batch_y.shape, batch_pred.shape', batch_y.shape, batch_pred.shape)
                    assert(batch_y.shape == batch_pred.shape, 
                           'Ошибка в test\nВектора разного размера\nbatch_y - ', str(batch_y.shape),
                           '\nbatch_pred - ', str(batch_pred.shape))
                    #curr_ndcgs
                    ndcgs.append(self._ndcg_k(batch_y, batch_pred, self.ndcg_top_k))
            except Exception as e:
                print('_eval_test_set error')
            else:
                return np.mean(ndcgs)
        
    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        _, indices = torch.sort(ys_pred, 0, descending=True)
        ys_true_sorted = ys_true[indices]
        res = 0
        try:
            for i, val in enumerate(ys_true_sorted[:ndcg_top_k]):
                val = float(2 ** val - 1) / math.log2(i+2)
                res += val
        except Exception as e:
            print('_dcg_k error')
            return 0.0
        else: 
            return float(res)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        ys_true_sorted_ideal, _ = torch.sort(ys_true, 0, descending=True)
        ideal_dcg = 0
        try:
            for i, val in enumerate(ys_true_sorted_ideal[:ndcg_top_k]):
                val = float(2 ** val - 1) / math.log2(i+2)
                ideal_dcg += val
            res = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        except Exception as e:
            print('_ndcg_k error')
            return 0.0
        else:
            return (res / float(ideal_dcg))