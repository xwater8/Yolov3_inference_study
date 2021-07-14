import os
import torch
from torch import nn
import numpy as np
from pprint import pprint
from collections import OrderedDict
import glob
import pdb
import cv2
import torchvision
import pickle


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
        


def parser_cfg(cfg_file):
    """
    拆解yolov3的config成為一個個module(Conv, upsample, yolo, route)
    """
    filter_lines= []
    with open(cfg_file, 'r')as f:
        for line in f.readlines():
            line= line.strip()
            if len(line)==0:
                continue
            if line.startswith('#'):
                continue
            filter_lines.append(line)
    
    block= OrderedDict()
    blocks= []
    for line in filter_lines:
        if line.startswith('['):
            if len(block) >0:
                blocks.append(block)
                block= OrderedDict()
            block['type']= line[1:-1]
            continue
        key, values= line.split('=')
        block[key.strip()]= values.strip()
    
    #last module
    blocks.append(block)
    return blocks
        

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YoloLayer(nn.Module):
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors= anchors#在416x416的圖片上面anchor的大小
    
    def forward(self, input_x, netHW, class_count):
        N,C,H,W= input_x.size()
        stride= netHW[0]//H, netHW[1]//W
        grid_size= netHW[0]//stride[0], netHW[1]//stride[1]
        
        bbox_attrs= (5+class_count)
        num_anchors= len(self.anchors)
        #(N, anchors*box_attrs, H, W)
        input_x= input_x.view(N, num_anchors*bbox_attrs, grid_size[0]*grid_size[1])
        input_x= torch.transpose(input_x, 1,2).contiguous()#(N, gridH*gridW, bbox_attrs*anchors)        
        input_x= input_x.view(N, grid_size[0]*grid_size[1]*num_anchors, bbox_attrs)#(N, gridH*gridW*anchors, bbox_attrs)

        anchors= [(a[0]/stride[0], a[1]/stride[1]) for a in self.anchors]#在這張featureMap上anchor的實際大小
        x= torch.arange(grid_size[1])
        y= torch.arange(grid_size[0])
        
        y_offset, x_offset= torch.meshgrid(x,y)

        if torch.cuda.is_available():
            x_offset= x_offset.cuda()
            y_offset= y_offset.cuda()
        
        x_offset, y_offset= x_offset.view(-1, 1), y_offset.view(-1,1)
        xy_offset= torch.cat((x_offset, y_offset), dim=1)#(gridH*gridW, 2)
        xy_offset= xy_offset.repeat(1, num_anchors).view(-1,2).unsqueeze(0)#(gridH*gridW, 2*3)-->(1, gridH*gridW*3, 2)
                        
        #(tx, ty, tw, th, objectness, p1...pn)
        input_x[:,:,0]= torch.sigmoid(input_x[:,:,0])
        input_x[:,:,1]= torch.sigmoid(input_x[:,:,1])
        
        input_x[:,:,:2]+= xy_offset

        input_x[:,:,4]= torch.sigmoid(input_x[:,:,4])

        #W,H
        if torch.cuda.is_available():
            anchors= torch.FloatTensor(anchors).view(-1, 2).cuda()#(num_anchors, 2)
        anchors= anchors.repeat(grid_size[0]*grid_size[1], 1).view(-1, 2).unsqueeze(0)#(1, gridH*gridW*3, 2)
        input_x[:,:,2:4]= torch.exp(input_x[:,:,2:4]) * anchors
                
        #classes score        
        input_x[:,:,5:]= torch.sigmoid(input_x[:,:,5:])

        input_x[:,:,0]*= stride[1]#x * strideW
        input_x[:,:,1]*= stride[0]#y * strideH
        input_x[:,:,2]*= stride[1]#w * strideW        
        input_x[:,:,3]*= stride[0]#h * strideH

        return input_x







        

    


def create_modules(net_modules_config):

    module_list= nn.ModuleList()
    in_channel= 3
    output_channels= []

    for layer_idx, module_config in enumerate(net_modules_config):
        module= nn.Sequential()
        if module_config['type']=='convolutional':
            use_batchNorm= int(module_config.get('batch_normalize', 0))
            use_pad= int(module_config.get('pad', 0))
            

            kernel_size= int(module_config['size'])
            stride= int(module_config['stride'])            
            out_channel= int(module_config['filters'])
            if use_pad:
                padding_size= kernel_size //2
            else:
                padding_size= 0
            
            #若使用batchNorm則沒有bias
            conv_layer= nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding_size, bias= not use_batchNorm)
            module.add_module('conv_{}'.format(layer_idx), conv_layer)
            

            if use_batchNorm:
                bn_layer= nn.BatchNorm2d(out_channel)
                module.add_module('batchNorm_{}'.format(layer_idx), bn_layer)
            
            if module_config['activation']=='leaky':
                #Linear代表什麼都不做
                actviation_layer= nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{}'.format(layer_idx), actviation_layer)
        elif module_config['type']=='shortcut':
            #residual shortcut-skip connection            
            out_channel= in_channel
            module.add_module('shortcut_{}'.format(layer_idx), EmptyLayer())
        elif module_config['type']=='route':
            layers= module_config['layers'].split(',')
            if len(layers) > 1:
                start, end= (int(layer) for layer in layers)
                out_channel= output_channels[start] + output_channels[end]#two layer concate
            else:
                start= int(layers[0])
                out_channel= output_channels[start] 

            module.add_module('route_{}'.format(layer_idx), EmptyLayer())

        elif module_config['type']=='upsample':
            scale_factor= int(module_config['stride'])
            upsample_layer= nn.Upsample(scale_factor= scale_factor, mode='nearest')
            module.add_module('upsample_{}'.format(layer_idx), upsample_layer)
            out_channel= in_channel
        elif module_config['type']=='yolo':
            masks= list(map(int, module_config['mask'].split(',')))
            anchors= list(map(int, module_config['anchors'].split(',')))
            layer_anchors= [(anchors[idx*2], anchors[idx*2+1]) for idx in masks]
            yolo_layer= YoloLayer(anchors= layer_anchors)
            # module.add_module('yolo_{}'.format(layer_idx), yolo_layer)
            module= yolo_layer
            out_channel= in_channel
            


        module_list.append(module)        
        output_channels.append(out_channel)
        in_channel= out_channel
    
    return module_list
    
            

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        net_modules_config= parser_cfg(cfg_file)
        self.net_info= net_modules_config[0]
        self.net_modules_config= net_modules_config[1:]
        self.net_modules= create_modules(self.net_modules_config)
        self.seens= None
    
    def load_weights(self, weight_file):

        def get_weight_move_ptr(weights, ptr, weight_param):
            weight_count= weight_param.numel()
            module_weight= weights[ptr: ptr+weight_count]
            module_weight= module_weight.view_as(weight_param)#reshape as it
            return module_weight, ptr+weight_count

        with open(weight_file, 'rb')as f:
            #1.Major version number
            #2.Minor version number
            #3.Subvision version number
            #4,5: Images seen by network(during training)            #
            headers= np.fromfile(f, dtype=np.int32, count=5)
            self.headers= torch.from_numpy(headers)
            self.seens= self.headers[3]

            #Start to load weights
            weights= np.fromfile(f, dtype=np.float32)
            weights= torch.from_numpy(weights)
            ptr= 0
            for module_cfg, net_module in zip(self.net_modules_config, self.net_modules):
                if module_cfg['type']!='convolutional':
                    continue
                #Conv module: Conv -> BN -> leaklyReLU
                conv= net_module[0]
                
                use_batchNorm= int(module_cfg.get('batch_normalize', 0))
                if use_batchNorm:
                    batchNorm= net_module[1]
                    #Read from files                    
                    bn_bias, ptr= get_weight_move_ptr(weights, ptr, batchNorm.bias)                                        
                    bn_weight, ptr= get_weight_move_ptr(weights, ptr, batchNorm.weight)                    
                    bn_running_mean, ptr= get_weight_move_ptr(weights, ptr, batchNorm.running_mean)                    
                    bn_running_var, ptr= get_weight_move_ptr(weights, ptr, batchNorm.running_var)
                    
                    #Load to module
                    batchNorm.bias.data.copy_(bn_bias)
                    batchNorm.weight.data.copy_(bn_weight)
                    batchNorm.running_mean.data.copy_(bn_running_mean)
                    batchNorm.running_var.data.copy_(bn_running_var)
                    
                else:                    
                    conv_bias, ptr= get_weight_move_ptr(weights, ptr, conv.bias)
                    conv.bias.data.copy_(conv_bias)

                conv_weight, ptr= get_weight_move_ptr(weights, ptr, conv.weight)
                conv.weight.data.copy_(conv_weight)
                
        return    



    def forward(self, input_x):
        N,C,H,W= input_x.size()

        output_features= []
        detections= []
        for layer_idx, (module_cfg, net_module) in enumerate(zip(self.net_modules_config, self.net_modules)):
                        
            if module_cfg['type']=='convolutional' or module_cfg['type']=='upsample':
                input_x= net_module(input_x)            
            elif module_cfg['type']=='shortcut':
                from_layer_idx= int(module_cfg['from'])
                input_x= output_features[-1] + output_features[from_layer_idx]                
            elif module_cfg['type']=='route':
                layers= module_cfg['layers'].split(',')
                if len(layers) > 1:
                    start, end= (int(layer) for layer in layers)                    
                    input_x= torch.cat((output_features[start], output_features[end]), dim=1)  
                else:
                    start= int(layers[0])
                    input_x= output_features[start]
            elif module_cfg['type']=='yolo':
                #transforms data-->BBoxes
                netHW= int(self.net_info['height']), int(self.net_info['width'])
                class_count= int(module_cfg['classes'])         
                
                input_x= net_module(input_x, netHW, class_count)
                detections.append(input_x)
                                

            output_features.append(input_x)            
        
        
        return torch.cat(detections, 1)


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
        keep_idxs= torchvision.ops.batched_nms(bbox_coord, scores, class_idxs, iou_threshold= iou_thresh)      
        detection_bboxes= pred_tensor[keep_idxs]
            
    return detection_bboxes
            
        

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



def main():    
    img_root= './imgs'
    cfg_file= './cfg/yolov3.cfg'
    weight_path= 'yolov3.weights'
    class_txt_path= './data/coco.names'
    color_file_path= './pallete'
    output_root= './detect_result'
    os.makedirs(output_root, exist_ok=True)

    class_names= load_className(class_txt_path)
    color_dict= load_colors(class_names, color_file_path)
    
    network= Darknet(cfg_file)
    network.load_weights(weight_path)
    network.cuda()
    network.eval()
    
    img_paths= get_imgPaths(img_root)
    
    for img_path in img_paths:
        img= cv2.imread(img_path)
        img_data= preprocess(img, input_size=(416,416))
        
        with torch.no_grad():
            output= network(img_data.cuda())
                    
        bboxes_tensor= postprocess_bboxes(output, thresh=0.5, iou_thresh=0.4)
        bboxes_tensor= restore_bbox_from_letterimgBBox(bboxes_tensor, imgHW= img.shape[:2], netHW=(416,416))

        bboxes= bboxNumpy_to_BBox(bboxes_tensor, class_names)                
        img= draw_bboxes(bboxes, img, color_dict)
        
        output_path= os.path.join(output_root, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
        cv2.imshow("Draw", img)
        cv2.waitKey()
    

if __name__=='__main__':
    main()
