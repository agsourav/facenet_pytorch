import torch
import torch.nn as nn

def evaluate(input_, true_label, model_inc, model_cls, topk):
    with torch.no_grad():
        feature_out = model_inc(input_)
        out = model_cls(feature_out)
    pred_topk = torch.topk(out, topk)
    pred = torch.argmax(out, dim = 1)
    true_positive_TOPK = 0
    true_positive = 0
    print('input shape: {0}\toutput shape: {1}'.format(input_.shape, out.shape))
    print(pred_topk.indices.shape)
    for i in range(input_.shape[0]):
        #print('true labels: {0}\tpred_topk: {1}'.format(true_label[i], pred_topk[i]))
        if true_label[i] in pred_topk.indices[i]:
            #print(true_label[i], pred_topk[i])
            true_positive_TOPK = true_positive_TOPK + 1
        if true_label[i] == pred[i]:
            true_positive = true_positive + 1
    print('top{0} accuracy score {1}'.format(topk, true_positive_TOPK/input_.shape[0]))
    print('top 1 accuracy score {0}'.format(true_positive/input_.shape[0]))