import torch
from torch import nn

from pprint import pprint

from collections import OrderedDict

import pdb

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
        self.anchors= anchors
    


def create_modules(net_modules_config):

    module_list= nn.ModuleList()
    in_channel= 3
    output_channels= []

    for layer_idx, module_config in enumerate(net_modules_config[1:]):
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
                actviation_layer= nn.LeakyReLU(inplace=True)
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
            upsample_layer= nn.Upsample(scale_factor, mode='bilinear')
            module.add_module('upsample_{}'.format(layer_idx), upsample_layer)
            out_channel= in_channel
        elif module_config['type']=='yolo':
            masks= list(map(int, module_config['mask'].split(',')))
            anchors= list(map(int, module_config['anchors'].split(',')))
            layer_anchors= [(anchors[idx*2], anchors[idx*2+1]) for idx in masks]
            yolo_layer= YoloLayer(anchors= layer_anchors)
            module.add_module('yolo_{}'.format(layer_idx), yolo_layer)
            out_channel= in_channel
            # pdb.set_trace()


        module_list.append(module)        
        output_channels.append(out_channel)
        in_channel= out_channel
    
    return module_list
    
            
            



            




def main():
    cfg_file= './cfg/yolov3.cfg'
    net_modules_config= parser_cfg(cfg_file)
    # pprint(net_modules_config)
    # print(len(net_modules_config))

    module_list= create_modules(net_modules_config)
    print(module_list)

if __name__=='__main__':
    main()
