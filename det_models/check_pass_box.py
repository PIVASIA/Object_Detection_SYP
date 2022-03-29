import numpy as np

def check_box(true_box,pred_box):
    box = [max(true_box[0],pred_box[0]), max(true_box[1],pred_box[1]),
           min(true_box[2],pred_box[2]), min(true_box[3],pred_box[3])]
    square = max(box[2]-box[0],0)*max(box[3]-box[1],0)
    sq_true_box = (true_box[2]-true_box[0])*(true_box[3]-true_box[1])
    sq_pred_box = (pred_box[2]-pred_box[0])*(pred_box[3]-pred_box[1])
    
    iou = square/(sq_pred_box+sq_true_box-square)
    if (iou>0.7):
        return True
    return False