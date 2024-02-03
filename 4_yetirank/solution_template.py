import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.6,
                 max_depth: int = 10, min_samples_leaf: int = 5):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
        self.n_sample = int(subsample * self.X_train.shape[0])
        self.n_cols = int(colsample_bytree  * self.X_train.shape[1])
        print('n_sample', self.n_sample)
        print('n_cols', self.n_cols)

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
        self.ys_train = torch.reshape(self.ys_train, (-1, 1))
        self.ys_test = torch.reshape(self.ys_test, (-1, 1))
        
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

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        # допишите ваш код здесь
        np.random.seed(cur_tree_idx)
        
        for i in range(len(self.key_query_ids_train)):
#             if i ==7:
#                 print('i',i)
#                 print(self.ys_train[curr_idx])
            curr_idx = np.where(self.index_query_ids_train==i)
            curr_lamdas = self._compute_lambdas(y_true=self.ys_train[curr_idx], 
                                                y_pred=train_preds[curr_idx])
            if i == 0:
                idxs = curr_idx[0]
                lamdas = curr_lamdas
            else:
                idxs = np.concatenate([idxs, curr_idx[0]])
                lamdas = torch.cat([lamdas, curr_lamdas])      
        
        #lambda_update = self._compute_lambdas(y_true=self.ys_train, y_pred=train_preds)
        lambda_update = lamdas[idxs]
        #print('lambda_update', lambda_update)
        #print('sum lambda_update', torch.sum(lambda_update))
        #print('lambda_update', lambda_update.shape)
        
        tree_subsample_idx = torch.randperm(self.X_train.shape[0])[:self.n_sample]
        tree_cols_idx = torch.randperm(self.X_train.shape[1])[:self.n_cols]

        tree_X_train = self.X_train[tree_subsample_idx]
        tree_X_train = torch.index_select(tree_X_train, 1, tree_cols_idx)
        #print('tree_X_train - ', tree_X_train.shape)
        #print(tree_X_train)

        tree_ys_train = lambda_update[tree_subsample_idx]
        #print('tree_ys_train - ', tree_ys_train.shape)
        # print(tree_ys_train)
        
        tree_regressor = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        tree_regressor.fit(tree_X_train, tree_ys_train)
        
        return [tree_regressor, tree_cols_idx]

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        # допишите ваш код здесь
        ndcgs=[]
        key_query_ids, index_query_ids = np.unique(queries_list, return_inverse=True)
        #print(key_query_ids)
        for i in range(len(key_query_ids)):
            idx = np.where(index_query_ids==i)
            
            ys_true = true_labels[idx]
            ys_pred = preds[idx]
            #print('ys_true, ys_pred', ys_true, ys_pred)
            #print('ys_true, ys_pred', ys_true.shape, ys_pred.shape)
            
            ndcg = self._ndcg_k(ys_true=ys_true, ys_pred=ys_pred, ndcg_top_k=10)
            ndcgs.append(ndcg)
        #print(ndcgs)
            
        return np.mean(ndcgs)

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        self.trees = []
        self.cols_idxs = []
        ndcg = []
        for tree in range(self.n_estimators):
            print('tree n', tree)
            if tree == 0:
                train_preds = torch.zeros(self.X_train.shape[0], 1)
                test_preds = torch.zeros(self.X_test.shape[0], 1)
            curr_tree, curr_cols_idx = self._train_one_tree(cur_tree_idx=tree, train_preds=train_preds)
            self.trees.append(curr_tree)
            self.cols_idxs.append(curr_cols_idx)
            curr_predict = curr_tree.predict(torch.index_select(self.X_train, 1, curr_cols_idx))
            curr_predict = torch.reshape(torch.FloatTensor(curr_predict), (-1, 1))
            #print('curr_predict', curr_predict.shape)
            # print(curr_predict)
            train_preds -= self.lr * curr_predict
            #print('train_preds', train_preds.shape)
            #print(train_preds) 
            
            curr_test_predict = curr_tree.predict(torch.index_select(self.X_test, 1, curr_cols_idx))
            curr_test_predict = torch.reshape(torch.FloatTensor(curr_test_predict), (-1, 1))
            test_preds -= self.lr * curr_test_predict
            
            curr_ndcg = self._calc_data_ndcg(queries_list=self.query_ids_train, 
                                             true_labels=self.ys_train, preds=train_preds)
            curr_test_ndcg = self._calc_data_ndcg(queries_list=self.query_ids_test,
                                                 true_labels=self.ys_test, preds=test_preds)
            ndcg.append(curr_test_ndcg)
            
#             curr_ndcg = self._ndcg_k(ys_true=self.ys_train, ys_pred=train_preds, ndcg_top_k=10)
#             #print('train curr_ndcg', curr_ndcg)
            
#             curr_test_ndcg = self._ndcg_k(ys_true=self.ys_test, ys_pred=test_preds, ndcg_top_k=10)
#             ndcg.append(curr_test_ndcg)
#             #print('test curr_ndcg', curr_test_ndcg)
#             #print(ndcg)
            
        self.best_ndcg = np.max(ndcg)
        self.trees = self.trees[:np.argmax(ndcg)+1]
        self.cols_idxs = self.cols_idxs[:np.argmax(ndcg)+1]
        
        print(ndcg[:np.argmax(ndcg)+1])
        return self.best_ndcg

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        np.random.seed(0)
        preds = torch.zeros(data.shape[0], 1)
        for i in range(len(self.trees)):
            np.random.seed(i)
            print('tree n', i)
            curr_tree = self.trees[i]
            curr_cols_idx = self.cols_idxs[i]
            
            curr_predict = curr_tree.predict(torch.index_select(data, 1, curr_cols_idx))
            curr_predict = torch.reshape(torch.FloatTensor(curr_predict), (-1, 1))
            preds -= self.lr * curr_predict
        return preds

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # рассчитаем нормировку, IdealDCG
        #ideal_dcg = compute_ideal_dcg(y_true, ndcg_scheme=ndcg_scheme)
        y_true_sorted_ideal, _ = torch.sort(y_true, 0, descending=True)
        # print(y_true)
        # print(y_true_sorted_ideal)
        ideal_dcg = 0
        for i, val in enumerate(y_true_sorted_ideal):
            val = float(2 ** val - 1) / math.log2(i+2)
            ideal_dcg += val
        #print(ideal_dcg)
        try:
            N = 1 / ideal_dcg
        except:
            N = 0
            
        # рассчитаем порядок документов согласно оценкам релевантности
        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = self._compute_labels_in_batch(y_true)
            # посчитаем изменение gain из-за перестановок
            gain_diff = self._compute_gain_diff(y_true)

            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update

    def _compute_labels_in_batch(self, y_true):

        # разница релевантностей каждого с каждым объектом
        rel_diff = y_true - y_true.t()

        # 1 в этой матрице - объект более релевантен
        pos_pairs = (rel_diff > 0).type(torch.float32)

        # 1 тут - объект менее релевантен
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij

    def _compute_gain_diff(self, y_true):
        gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        return gain_diff

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        _, indices = torch.sort(ys_pred, 0, descending=True)
        ys_true_sorted = ys_true[indices]
        res = 0
        try:
            for i, val in enumerate(ys_true_sorted[:ndcg_top_k]):
                val = float(2 ** val - 1) / math.log2(i+2)
                res += val
        except Exception as e:
            #print('_dcg_k error')
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
            return (res / float(ideal_dcg))
        except Exception as e:
            #print('_ndcg_k error')
            return 0.0

    def save_model(self, path: str):
        state = {'trees': self.trees, 'cols_idxs': self.cols_idxs}
        f = open(path, 'wb')
        pickle.dump(state, f)

    def load_model(self, path: str):
        f = open(path, 'rb')
        state = pickle.load(f)
        self.trees = state['trees']
        self.cols_idxs = state['cols_idxs']