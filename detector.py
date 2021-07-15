import os
import torch
import numpy as np
from pprint import pprint
import glob
import pdb
import cv2
import torchvision
import time
import pickle

from darknet import Darknet


class BBox:
    def __init__(self, xmin, ymin, xmax, ymax, score, clsName):
        self.xmin= int(xmin)
        self.ymin= int(ymin)
        self.xmax= int(xmax)
        self.ymax= int(ymax)
        self.score= score
        self.clsName= clsName

        self.width= xmax-xmin
        self.height= ymax-ymin
    
    def __repr__(self):
        format_str= 'xmin={}, ymin={}, xmax={}, ymax={}, score={:.3f}, clsName={}'.format(self.xmin, self.ymin, self.xmax, self.ymax, self.score, self.clsName)
        return format_str
    
    @property
    def pt1(self):
        return (self.xmin, self.ymin)
    @property
    def pt2(self):
        return (self.xmax, self.ymax)
        

def letterbox_img(img, input_size):
    target_w, target_h= input_size
    
    img_h, img_w= img.shape[0], img.shape[1]
    min_aspect_ratio= min(target_w/img_w, target_h/img_h)
    
    new_h, new_w= int(img_h * min_aspect_ratio), int(img_w * min_aspect_ratio)
    resize_img= cv2.resize(img, (new_w, new_h))
    
    output_img= np.full((target_h, target_w, 3), 128,dtype=np.uint8)
    padding_h= (target_h - new_h)//2
    padding_w= (target_w - new_w)//2
    
    output_img[padding_h:padding_h+new_h, padding_w:padding_w+new_w]= resize_img
    return output_img
    

def preprocess(img, input_size):
    img= letterbox_img(img, input_size)    
    img_data= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    img_data= img_data.astype(np.float32)/255.0    
    img_data= torch.from_numpy(img_data).permute(2,0,1).unsqueeze(0)
    return img_data



def IoUs(bboxes):
    inter_xmin= torch.maximum(bboxes[0,0], bboxes[:,0])
    inter_ymin= torch.maximum(bboxes[0,1], bboxes[:,1])
    inter_xmax= torch.minimum(bboxes[0,2], bboxes[:,2])    
    inter_ymax= torch.minimum(bboxes[0,3], bboxes[:,3])

    inter_diffx= torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_diffy= torch.clamp(inter_ymax - inter_ymin, min=0)

    inter_area= inter_diffx * inter_diffy
    
    area_bboxes= (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
    
    ious= inter_area / (area_bboxes[0]+area_bboxes - inter_area)
    return ious
    


def nms(bboxes_tensor, iou_thresh):
    """
    bboxes_tensor:只有指定某一個類別的所有bboxes進行NMS
    """
    scores= bboxes_tensor[:, -2]
    sort_idx= torch.argsort(scores, descending=True)
    keep_idxs= []
    
    while sort_idx.numel() > 0:
        
        #Computer ious        
        bboxes= bboxes_tensor[sort_idx]
        keep_idxs.append(sort_idx[0])

        ious= IoUs(bboxes)
        sort_idx= sort_idx[ious<iou_thresh]
    
    return keep_idxs

    

def postprocess_bboxes(predictions, thresh= 0.5, iou_thresh= 0.4):
    """
    Args:
        predictions: (N, bboxes, bbox_attrs)
        thresh: objectness的門檻值
        iou_thresh: iou小於大於此數值代表重疊率過高，濾雕分數較低的框
    Return:
        detection_bboxes(torch.tensor): (bbox_count, 7)=(xmin, ymin, xmax, ymax, objectness, cls_score, cls_idx)
    """    
    #Convert predictions to BBox(leftTopX, leftTopY, rightDownX, rightDownY, objectness, class_scores, class_idx)
    num_classes= predictions.size(-1) - 5
    bbox_corner= predictions[:,:,:4].clone()
    predictions[:,:,0]= (bbox_corner[:,:,0] - bbox_corner[:,:,2]/2.0)#lt_x
    predictions[:,:,1]= (bbox_corner[:,:,1] - bbox_corner[:,:,3]/2.0)#lt_y
    predictions[:,:,2]= (bbox_corner[:,:,0] + bbox_corner[:,:,2]/2.0)#rd_x
    predictions[:,:,3]= (bbox_corner[:,:,1] + bbox_corner[:,:,3]/2.0)#rd_y
           
    batch_detection_bboxes= []
    for batch_idx in range(predictions.size(0)):
        pred_tensor= predictions[batch_idx]             
        pred_tensor= pred_tensor[pred_tensor[:,4] >thresh]#Filter by ojbectness
        
        pred_tensor[:,:2]= torch.clamp(pred_tensor[:,:2], min=0)
        max_score, max_score_idx= torch.max(pred_tensor[:,5:].view(-1, num_classes), dim=1)#values, idx                
        
        max_score= max_score.unsqueeze(-1)        
        max_score_idx= max_score_idx.unsqueeze(-1)   
        
        pred_tensor= torch.cat((pred_tensor[:,:5], max_score, max_score_idx), dim=-1)

        bbox_coord= pred_tensor[:,:4]
        scores= pred_tensor[:,-2]
        class_idxs= pred_tensor[:,-1]
        # keep_idxs= torchvision.ops.batched_nms(bbox_coord, scores, class_idxs, iou_threshold= iou_thresh)      
        # detection_bboxes= pred_tensor[keep_idxs]

        class_idxs= torch.unique(class_idxs)
        detection_bboxes= []
        for class_id in class_idxs:
            class_mask= (pred_tensor[:, -1]==class_id)
            class_pred_tensor= pred_tensor[class_mask]
            keep_idxs= nms(class_pred_tensor, iou_thresh)                   
            keep_idxs= torch.tensor(keep_idxs)#需轉成tensor才可使用torch.index_select
            if class_pred_tensor.is_cuda: 
                keep_idxs= keep_idxs.cuda()

            #直接使用會有問題class_pred_tensor[keep_idxs].view(-1, 7)
            #若keep_idxs >=2, 會取得class_pred_tensor[keep_idxs[0], keep_idxs[1]]
            #因此改用index_select
            nms_pred_tensor= torch.index_select(class_pred_tensor, dim=0, index=keep_idxs)
            
            detection_bboxes.append(nms_pred_tensor)
        if len(detection_bboxes)>0:            
            detection_bboxes= torch.cat(detection_bboxes)
        else:
            detection_bboxes= pred_tensor.clone()#tensor([], size=(0,85))
        batch_idxs= detection_bboxes.new(detection_bboxes.size(0), 1)        
        batch_idxs[:,0]= batch_idx
        batchId_detection_bboxes= torch.cat([batch_idxs, detection_bboxes], dim=-1)
        batch_detection_bboxes.append(batchId_detection_bboxes)
        
    return torch.cat(batch_detection_bboxes, dim=0)
            
        

def get_imgPaths(img_root):
    img_paths= glob.glob(os.path.join(img_root, '*.jpg'))
    return sorted(img_paths)


def restore_bbox_from_letterimgBBox(pred_bbox, imgHW, netHW):
    """
    預測出來的pred_bbox是在letterimg上, 我們要將座標還原回原圖
    pred_bbox(Tensor): 從yolo網路裡面預測出來的bbox(經過letterbox處理的圖片上)
    imgHW: 原圖的imgSize
    netHW: 輸入network的圖片大小
    """
    
    scale_factor= min(netHW[0]/imgHW[0], netHW[1]/imgHW[1])
    padH= (netHW[0] - imgHW[0] * scale_factor)/2
    padW= (netHW[1] - imgHW[1] * scale_factor)/2
    
    pred_bbox[:, 0]-=padW#leftTopX
    pred_bbox[:, 2]-=padW#rightDownX
    pred_bbox[:, 1]-=padH#leftTopY
    pred_bbox[:, 3]-=padH#rightDownY
    pred_bbox[:, :4]/=scale_factor

    pred_bbox[:, [0,2]]= torch.clamp(pred_bbox[:, [0,2]], min=0, max=imgHW[1])
    pred_bbox[:, [1,3]]= torch.clamp(pred_bbox[:, [1,3]], min=0, max=imgHW[0])

    return pred_bbox




def bboxNumpy_to_BBox(bboxes, class_names):
    """
    bboxes= (xmin, ymin, xmax, ymax, objectness, cls_score, cls_idx)
    """
    convert_bboxes= []
    for bbox in bboxes:
        class_idx= int(bbox[6])
        clsName= class_names[class_idx]
        convert_bbox= BBox(xmin= bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3], score=bbox[5], clsName=clsName)
        convert_bboxes.append(convert_bbox)
    return convert_bboxes


def load_className(class_txt):
    class_names= []
    with open(class_txt, 'r')as f:
        for line in f.readlines():
            line= line.strip()
            class_names.append(line)
    return class_names

def load_colors(class_names, color_file):
    color_dict= {}
    with open(color_file, 'rb')as f:
        colors= pickle.load(f)

    for cls_name, one_color in zip(class_names, colors):
        color_dict[cls_name]= one_color
    return color_dict

def draw_bboxes(bboxes, img, color_dict):    
    font_face= cv2.FONT_HERSHEY_COMPLEX
    font_scale= 0.8
    font_thickness= 1
        
    for bbox in bboxes:                
        color= color_dict[bbox.clsName]
        cv2.rectangle(img, bbox.pt1, bbox.pt2, color, 2)        
        (txt_w, txt_h), baseline= cv2.getTextSize(bbox.clsName, font_face, font_scale, font_thickness)
        txt_pt1= (bbox.xmin, bbox.ymin-txt_h//2)        
        cv2.putText(img, bbox.clsName, txt_pt1, font_face, font_scale, color, font_thickness)
    
    return img


def generator_batchImgPaths(img_paths, batch_size=1):
    batch_imgPath= []
    for img_path in img_paths:
        batch_imgPath.append(img_path)
        if len(batch_imgPath)==batch_size:
            yield batch_imgPath
            batch_imgPath.clear()
    
    if len(batch_imgPath)>0:
        yield batch_imgPath


def main():    
    img_root= './imgs'
    cfg_file= './cfg/yolov3.cfg'
    weight_path= 'yolov3.weights'
    class_txt_path= './data/coco.names'
    color_file_path= './pallete'
    output_root= './detect_result'
    batch_size=5
    os.makedirs(output_root, exist_ok=True)

    class_names= load_className(class_txt_path)
    color_dict= load_colors(class_names, color_file_path)
    
    network= Darknet(cfg_file)
    network.load_weights(weight_path)
    network.cuda()
    network.eval()
    
    img_paths= get_imgPaths(img_root)
    
    for batch_img_path in generator_batchImgPaths(img_paths, batch_size= batch_size):
        start_t_batch= time.time()
        imgs= [cv2.imread(img_path) for img_path in batch_img_path]
        img_data= [preprocess(img, input_size=(416,416)) for img in imgs]        
        img_data= torch.cat(img_data, dim=0)        
        
        with torch.no_grad():
            output= network(img_data.cuda())
        
        batch_bboxes_tensor= postprocess_bboxes(output, thresh=0.5, iou_thresh=0.4)
        for batch_id, (img, img_path) in enumerate(zip(imgs, batch_img_path)):            
            mask= (batch_bboxes_tensor[:,0]==batch_id)
            bboxes_tensor= batch_bboxes_tensor[mask][:,1:]#去掉開頭的batch_idx
            bboxes_tensor= restore_bbox_from_letterimgBBox(bboxes_tensor, imgHW= img.shape[:2], netHW=(416,416))

            bboxes= bboxNumpy_to_BBox(bboxes_tensor, class_names)                
            img= draw_bboxes(bboxes, img, color_dict)
        
            output_path= os.path.join(output_root, os.path.basename(img_path))
            cv2.imwrite(output_path, img)
            # cv2.imshow("Draw", img)
            # cv2.waitKey()
        
        torch.cuda.synchronize()
        fps= (time.time() - start_t_batch) / len(batch_img_path)
        print("Process batch imgs: {}".format(fps))
    

if __name__=='__main__':
    main()
