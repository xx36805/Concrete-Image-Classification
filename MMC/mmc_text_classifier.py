# Experiment framework
import operator
import random
import threading
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Optional

import numpy as np
from MMC.compressors import DefaultCompressor
from tqdm import tqdm
from MMC.zstd_compressor import ZstdCompressor, self_adaption
from MMC.compressorclassifier import CompressorClassifier


COMPRESSOR_PROVIDERS = {
    "ZSTD_CL12": lambda size: ZstdCompressor(size=size, compression_level=12),
    "ZSTD_CL10": lambda size: ZstdCompressor(size=size, compression_level=10),
    "ZSTD_CL9": lambda size: ZstdCompressor(size=size, compression_level=9),
    "ZSTD_CL6": lambda size: ZstdCompressor(size=size, compression_level=6),
    "ZSTD_CL3": lambda size: ZstdCompressor(size=size, compression_level=3)
}

class KnnExpText:
    def __init__(
        self,
        aggregation_function: Callable,
        compressor: DefaultCompressor,
        distance_function: Callable,
    ) -> None:
        self.aggregation_func = aggregation_function
        self.compressor = compressor
        self.distance_func = distance_function
        self.distance_matrix: list = []
        self.num = 0
        self.counter_lock = threading.Lock()


    def calc_dis_single_multi(self, train_data: list, datum: str) -> list:
        """
        Calculates the distance between `train_data` and `datum` and returns
        that distance value as a float-like object.

        Arguments:
            train_data (list): Training data as a list-like object.
            datum (str): Data to compare against `train_data`.

        Returns:
            list: Distance between `t1` and `t2`.
        """

        # similarity_score_bert = 0
        # for i in range(len(train_data)):
            # print(train_data[i])
            # similarity_score_bert = agg_by_concat_space_bert(train_data[i],datum)

        distance4i = []
        t1_compressed = self.compressor.get_compressed_len(datum)

        for j, t2 in tqdm(enumerate(train_data)):

            t2_compressed = self.compressor.get_compressed_len(t2)

            t1t2_compressed = self.compressor.get_compressed_len(
                self.aggregation_func(datum, t2)
            )

            distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
            distance4i.append(distance)

        return distance4i

    def calc_dis_single_multi_add(self, train_data: list, datum: str) -> list:
        """
        Calculates the distance between `train_data` and `datum` and returns
        that distance value as a float-like object.

        Arguments:
            train_data (list): Training data as a list-like object.
            datum (str): Data to compare against `train_data`.

        Returns:
            list: Distance between `t1` and `t2`.
        """
        distance4i = []
        t1_compressed_zstd = self.compressor.get_compressed_len_zstd(datum)
        for j, t2 in tqdm(enumerate(train_data)):

            t2_compressed_zstd = self.compressor.get_compressed_len_zstd(t2)
            t1t2_compressed_zstd = self.compressor.get_compressed_len_zstd(
                self.aggregation_func(datum, t2)
            )

            distance_add = self.distance_func(t1_compressed_zstd, t2_compressed_zstd, t1t2_compressed_zstd)

            distance4i.append(distance_add)
        return distance4i




    def combine_dis_acc_single(
        self,
        k: int,
        train_data: list,
        train_label: list,
        preds: list,
        preds2: list,
        test_data: list,
        test_label: list,  # int, as used in this application
        # datum: str,
        # label: Any,  # int, as used in this application
    ) -> tuple:
        """
        Calculates the distance and the accuracy of the algorithm for a single
        datum with training.

        Arguments:
            k (int?): TODO
            train_data (list): Training data to compare distances.
            train_label (list): Correct Labels.
            datum (str): Datum used for predictions.
            label (Any): Correct label of datum.

        Returns:
            tuple: prediction, and a bool indicating prediction correctness.
        """
        datum = test_data[self.num]
        label = test_label[self.num]
        # Support multi processing - must provide train data and train label
        train_data, train_label = obtain_label_data(train_data, train_label, preds[self.num], preds2[self.num])
        distance4i = self.calc_dis_single_multi(train_data, datum)
        sorted_idx = np.argpartition(np.array(distance4i), range(2))
        pred_labels = defaultdict(int)
        data_l = []
        for j in range(2):
            pred_l = train_label[int(sorted_idx[j])]
            data_l.append(train_data[int(sorted_idx[j])])
            pred_labels[pred_l] += 1
        sorted_pred_lab = sorted(
            pred_labels.items(), key=operator.itemgetter(1), reverse=True
        )
        print("==============================================")
        print(sorted_pred_lab)
        most_count = sorted_pred_lab[0][1]
        if_right = 0
        most_label = sorted_pred_lab[0][0]
        label_2 = -1
        if most_count == 1:
            label_2 = sorted_pred_lab[1][0]
        if most_label == label:
            if_right = 1


        # if if_right == 0:
        #     if most_count == 1:
        #         data_a, data_b, labels = alternative(data_l, train_data, train_label, sorted_pred_lab)
        #         distance4i_a = self.calc_dis_single_multi_add(data_a, datum)
        #         min_value = min(distance4i_a)
        #         distance4i_a = [1 if x == min_value else x for x in distance4i_a]
        #         distance4i_b = self.calc_dis_single_multi_add(data_b, datum)
        #         distance4i_ab = distance4i_a + distance4i_b
        #         sorted_idx = np.argpartition(np.array(distance4i_ab), range(1))
        #         pred_labels = defaultdict(int)
        #
        #         pred_l = labels[sorted_idx[0]]
        #         pred_labels[pred_l] += 1
        #         sorted_pred_lab = sorted(
        #             pred_labels.items(), key=operator.itemgetter(1), reverse=True
        #         )
        #     most_label = sorted_pred_lab[0][0]
        #     if most_count == 1 and preds[self.num] == label_2:
        #         most_label = label_2
        # if most_label == label:
            # if_right = 1


        # if most_count == 1 and preds2[self.num] == label_2:
        #     most_label = label_2
        # if most_count == 1 and preds[self.num] != label_2 and preds2[self.num] != label_2:
        #     most_label = preds[self.num]
        # for pair in sorted_pred_lab:
        #     if pair[1] < most_count:
        #         break
        #     if pair[0] == label:
        #         if_right = 1
        #         most_label = pair[0]

        pred = most_label
        if most_count != 2:
            pred = "-1"
        with self.counter_lock:
            self.num += 1
        return pred, if_right

def alternative(data_l: list,
                train_data: list,
                train_label: list,
                sorted_pred_lab: list,
                ):
    label_a = sorted_pred_lab[0][0]
    label_b = sorted_pred_lab[1][0]
    label_a_data = []
    label_b_data = []
    labels = []

    i, j = 0, 0
    for data, label in zip(train_data, train_label):
        if label == label_a and data != data_l[0]:
            i += 1
            label_a_data.append(data)
        if label == label_b and data != data_l[1]:
            j += 1
            label_b_data.append(data)
    for i in range(i):
        labels.append(label_a)
    for j in range(j):
        labels.append(label_b)
    return label_a_data, label_b_data, labels

def obtain_label_data(train_data: list,
                train_label: list,
                pred_lab_a: list,
                pred_lab_b: list,
                # pred_lab_c: list
                      ):

    label_a_data = []
    label_b_data = []
    label_c_data = []
    label_abc_data = []
    labels = []
    i, j, k = 0, 0, 0
    for data, label in zip(train_data, train_label):
        if label == pred_lab_a:
            i = i + 1
            label_a_data.append(data)
        if label == pred_lab_b:
            j = j + 1
            label_b_data.append(data)
        # if label == pred_lab_c:
        #     k = k + 1
        #     label_c_data.append(data)
    for i in range(i):
        labels.append(pred_lab_a)
    for j in range(j):
        labels.append(pred_lab_b)

    label_abc_data = label_a_data + label_b_data + label_c_data
    return label_abc_data, labels

def combined_method(train_data: list, train_label: list):
    combined = [(label, data) for label, data in zip(train_label, train_data)]
    return combined


def non_neural_knn_exp(
    compressor_name: str,
    test_data: list,
    test_label: list,
    train_data: list,
    train_label: list,
    agg_func: Callable,
    dis_func: Callable,
    k: int,
    para: bool = True,
):
    print("Start classification task")
    cp = DefaultCompressor(compressor_name)
    knn_exp_ins = KnnExpText(agg_func, cp, dis_func)
    start = time.time()

    combined = combined_method(train_data, train_label)
    compressor_provider = COMPRESSOR_PROVIDERS["ZSTD_CL9"]
    # compressor_level = self_adaption(args.dataset)
    compressor_level = 9
    classifier = CompressorClassifier(lambda: compressor_provider(-1), 2, num_compressors_per_class=compressor_level)
    classifier.fit(combined)
    obs_count = 0
    correct_obs_count = 0
    preds_k1 =[]
    preds_k2 =[]
    for label, data in zip(test_label, test_data):
        predicted = classifier.predict(data)
        obs_count += 1
        preds_k1.append(predicted[0])
        preds_k2.append(predicted[1])
        if label in predicted:
            # if predicted == label:
            correct_obs_count += 1

    # train_data, train_label = obtain_classes_data(train_data, train_label, num_classes)
    test_label = [int(label) for label in test_label]
    # print("-------------------------------------------")
    # print(train_data)
    # print(train_label)
    # print(test_data)
    # print(test_label)
    # print(preds_k1)
    # print(preds_k2)
    # print("-------------------------------------------")
    if para:
        pred, correct = knn_exp_ins.combine_dis_acc_single(k, train_data, train_label, preds_k1, preds_k2, test_data, test_label)

    return pred, correct
    #     with Pool(1) as p:
    #         pred_correct_pair = p.map(
    #             partial(knn_exp_ins.combine_dis_acc_single, k, train_data, train_label, preds_k1, preds_k2, test_data, test_label),
    #             # partial(knn_exp_ins.combine_dis_acc_single, k, train_data, train_label, preds_k1, preds_k2, preds_k3, test_data, test_label),
    #             test_data, test_label
    #         )
    #     print(
    #         "accuracy:{}".format(
    #             np.average(np.array(pred_correct_pair, dtype=np.int32)[:, 1])
    #         )
    #     )
    #     # print('accuracy:{}'.format(np.average(np.array(pred_correct_pair, dtype=np.object_)[:, 1])))
    # else:
    #     knn_exp_ins.calc_dis(test_data, train_data=train_data)
    #     knn_exp_ins.calc_acc(k, test_label, train_label=train_label)
    # print("spent: {}".format(time.time() - start))
    # return pred_correct_pair