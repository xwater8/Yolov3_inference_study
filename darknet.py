
import torch
from torch import nn
import numpy as np
from collections import OrderedDict

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


def main():    
    cfg_file= './cfg/yolov3.cfg'
    weight_path= 'yolov3.weights'
            
    network= Darknet(cfg_file)
    network.load_weights(weight_path)
    network.cuda()
    network.eval()
    
    data= torch.randn(1,3,416,416).cuda()
    with torch.no_grad():
        output= network(data)
    print(output)
    print(output.size())


if __name__=='__main__':
    main()