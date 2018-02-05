import numpy as np

#predicted_order = [4, 4, 2, 3, 2, 4, 0, 1, 1, 4, 1, 3, 3, 2, 3, 4,2, 1, 0, 0]

def naive_dcg(predicted_order):
    i = 1
    cumulative_dcg = 0
    for x in predicted_order:
        cumulative_dcg += (x)/(np.log2(1+i))
        i += 1
    return cumulative_dcg

def dcg(predicted_order):
    index = np.array([float(i) for i in range(0,len(predicted_order))])
    discount = np.log2(index+2)
    discounted_result = predicted_order/discount
    return np.sum(discounted_result)

def ndcg(predicted_order, top_count=None):
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order[:top_count]) if top_count is not None else dcg(predicted_order)
    if our_dcg == 0:
      return 0
    #print('our_dcg:%f' %(our_dcg))
    max_dcg = dcg(sorted_list[:top_count]) if top_count is not None else dcg(sorted_list)
    if max_dcg >0:
        ndcg_output = our_dcg/max_dcg
    else:
        ndcg_output = 0
    return ndcg_output


def ndcg_lambdarank(predicted_order):
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order)
    max_dcg = dcg(sorted_list)
    ndcg_output = our_dcg/max_dcg
    return ndcg_output

def delta_ndcg(order1, pos1, pos2):
    ndcg1 = ndcg_lambdarank(order1)
    order1[[pos2, pos1]] = order1[[pos1,pos2]]
    ndcg2 = ndcg_lambdarank(order1)
    return np.absolute(ndcg1-ndcg2)

