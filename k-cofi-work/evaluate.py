import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time


# Global variables that are shared across processes
_model = None
_K = None
_test_dict = None
_train_dict = None

def evaluation(model, sess, top_k, test_unobserved_dict, test_dict, all_items):

    global _model
    global _test_dict
    global _test_unobserved_dict
    global _K
    global _all_items
    global _sess

    _model = model
    _K = top_k
    _test_unobserved_dict = test_unobserved_dict
    _test_dict = test_dict
    _all_items = all_items
    _sess = sess

    precisions, recalls, F1s, one_calls, ndcgs= [], [], [], [], []
    eval_losses = []

    for user in _test_unobserved_dict.keys():
        (precision, recall, F1, ndcg, one_call, eval_loss) = eval_one_user(user) # "eval_loss" calculated by eval_one_user() and others calculated by getRankingmetrics()
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
        ndcgs.append(ndcg)
        one_calls.append(one_call)
        eval_losses.append(eval_loss)
    return (precisions, recalls, F1s, ndcgs, one_calls, eval_losses)


def eval_one_user(u):

    # Get prediction scores
    itemlist = _test_dict[u]
    u_items_evaluate=_test_unobserved_dict[u]
    u_items_evaluate = list(u_items_evaluate)


    users = np.full(len(u_items_evaluate), u, dtype = 'int32')

    test_users = np.array(users)[:,None]
    test_items = np.array(u_items_evaluate)[:,None]
    labels = np.zeros(len(u_items_evaluate))[:,None]

    predictions, eval_loss = _sess.run([_model.output, _model.loss], feed_dict = {_model.user : test_users, _model.item : test_items, _model.rating : labels})

    map_item_score = {}
    for i in range(len(u_items_evaluate)):
        item = u_items_evaluate[i]
        map_item_score[item] = predictions[i]

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    precision, recall, F1, ndcg, one_call = getRankingmetrics(ranklist, itemlist)

    return (precision, recall, F1, ndcg, one_call, eval_loss)

def getRankingmetrics(ranklist, itemlist):
    precision, recall, F1, ndcg, one_call = [], [], [], [], []
    dcg, dcgbest, dcgbest2 = [], [], []
    dcg.append(0)
    dcgbest.append(0)
    dcgbest2.append(0)

    for k in range(1,_K+1):
        dcg_best_k = dcgbest[k-1]
        dcg_best_k += 1 / math.log(1+k)
        dcgbest.append(dcg_best_k)

    precision.append(_K)
    recall.append(_K)
    F1.append(_K)
    ndcg.append(_K)
    one_call.append(_K) # insure that the real index starts from 1, eg: ndcg[5] is the fifth ndcg value(ndcg@5)

    hit_sum = 0
    for k in range(1, _K+1):
        dcg.append(dcg[k-1])
        dcg_k = dcg[k-1]

        itemID = ranklist[k-1]
        if itemID in itemlist:
            hit_sum += 1
            dcg_k += 1 / math.log(k+1)
            dcg[k]=dcg_k

        prec = float(hit_sum) / k
        rec = float(hit_sum) / len(itemlist)
        precision.append(prec)
        recall.append(rec)
        f_1 = 0
        if prec+rec>0:
            f_1 =  2 * prec*rec / (prec+rec)
        F1.append(f_1)
        onecall = 1 if hit_sum > 0 else 0
        one_call.append(onecall)

        if len(itemlist) >= k:
            dcgbest2.append(dcgbest[k])
        else:
            dcgbest2.append(dcgbest2[k-1])

        ndcg_k = dcg[k] / dcgbest2[k]
        ndcg.append(ndcg_k)

    return precision, recall, F1, ndcg, one_call