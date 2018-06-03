import os
import caffe_pb2
import sys
import argparse
import collections
from google.protobuf import text_format

class Caffe2CodeGenerator:
    def __init__(self):
        fs_

class CaffeUtil:
    def __init__(self):
        self.net_parameter = caffe_pb2.NetParameter()

    def format_caffe_name(self, name):
        return '_'.join(('_'.join(name.split('/')).split('-')))
    
    ## load the network from caffe prototxt file.    
    def loadNetworkFromCaffePrototxt(self, filename):
        text_format.Merge(open(filename, 'r').read(), self.net_parameter)
        print ("OK: caffe prototxt file read successful.")

        if len(self.net_parameter.layer) == 0:
            print ("ERROR: unsupported prototxt file given, please upgrade the prototxt file and re-run the script")
            sys.exit(1)

        return self.net_parameter
    
    ## extract the input name from network extracted.
    def extractInput(self, input_dims):
        input_info = {}
        layers = self.net_parameter.layer
        first_layer_param = layers[0]
        first_layer_param_type = first_layer_param.type
        input_name = ""

        if len(self.net_parameter.input) != 0:
            input_name = self.format_caffe_name(self.net_parameter.input[0])
        
        elif (first_layer_param_type == "Data" or first_layer_param_type == "Input" or first_layer_param_type == "ImageData"):
            top_list = first_layer_param.top
            if (len(top_list) == 0):
                input_name = self.format_caffe_name(first_layer_param.name)
            else:
                input_name = self.format_caffe_name(top_list[0])

        else:
            bottom_list = first_layer_param.bottom
            if (len(bottom_list)) == 0:
                top_list = first_layer_param.top
                input_name = self.format_caffe_name(top_list[0])
            else:
                input_name = self.format_caffe_name(bottom_list[0])

        input_info[str(input_name)] = input_dims
        return input_info

        # extract layer attribute information from caffe layers.
    def extractCaffeAttrInfo(self, layer_param):
        layer_type = layer_param.type
        attribute_map = {}
        if (layer_type == "Convolution" or layer_type == "Deconvolution"):
            conv = layer_param.convolution_param
            pad_h = conv.pad_h if (conv.HasField('pad_h')) else (int(conv.pad[0]) if (len(conv.pad) > 0) else 0)
            pad_w = conv.pad_w if (conv.HasField('pad_w')) else (int(conv.pad[1]) if (len(conv.pad) > 1) else pad_h)
            stride_h = conv.stride_h if (conv.HasField('stride_h')) else (int(conv.stride[0]) if (len(conv.stride) > 0) else 1)
            stride_w = conv.stride_w if (conv.HasField('stride_w')) else (int(conv.stride[1]) if (len(conv.stride) > 1) else stride_h)
            kernel_h = conv.kernel_h if (conv.HasField('kernel_h')) else (int(conv.kernel_size[0]) if (len(conv.kernel_size) > 0) else 0)
            kernel_w = conv.kernel_w if (conv.HasField('kernel_w')) else (int(conv.kernel_size[1]) if (len(conv.kernel_size) > 1) else kernel_h)
            num_out = conv.num_output
            dilation_h = conv.dilation[0] if (len(conv.dilation) > 0) else 1
            dilation_w = conv.dilation[1] if (len(conv.dilation) > 1) else dilation_h
            bias_term = conv.bias_term
            groups = conv.group if (conv.HasField('group')) else 1

            attribute_map["strides"] = [stride_w, stride_h]
            attribute_map["kernel_shape"] = [kernel_w, kernel_h]
            attribute_map["group"] = groups
            attribute_map["pads"] = [pad_w, pad_h, pad_w, pad_h]
            attribute_map["dilations"] = [dilation_w, dilation_h]

        elif (layer_type == "Pooling"):
            pooling = layer_param.pooling_param
            pad_h = int(pooling.pad_h) if (pooling.HasField('pad_h')) else int(pooling.pad)
            pad_w = int(pooling.pad_w) if (pooling.HasField('pad_w')) else int(pooling.pad)
            stride_h = int(pooling.stride_h) if (pooling.HasField('stride_h')) else int(pooling.stride)
            stride_w = int(pooling.stride_w) if (pooling.HasField('stride_w')) else int(pooling.stride)
            kernel_h = int(pooling.kernel_h) if (pooling.HasField('kernel_h')) else int(pooling.kernel_size)
            kernel_w = int(pooling.kernel_w) if (pooling.HasField('kernel_w')) else int(pooling.kernel_size)

            attribute_map["strides"] = [stride_w, stride_h]
            attribute_map["kernel_shape"] = [kernel_w, kernel_h]
            attribute_map["pads"] = [pad_w, pad_h, pad_w, pad_h]
            attribute_map["dim_round_mode"] = "ceil"
            #attribute_map["dilations"] = [1,1]

        elif (layer_type == "LRN"):
            lrn = layer_param.lrn_param
            local_size = int(lrn.local_size)
            alpha = float(lrn.alpha)
            beta = float(lrn.beta)
            k = float(lrn.k)
            norm_region = lrn.norm_region

            attribute_map["alpha"] = alpha
            attribute_map["beta"] = beta
            attribute_map["size"] = local_size
            attribute_map["bias"] = k
            if (norm_region == caffe_pb2.LRNParameter.ACROSS_CHANNELS):
                attribute_map["mode"] = 1
            elif (norm_region == caffe_pb2.LRNParameter.WITHIN_CHANNEL):
                attribute_map["mode"] = 0

        elif (layer_type == "BatchNorm"):
            attribute_map["epsilon"] = float(layer_param.batch_norm_param.eps)

        elif (layer_type == "InnerProduct"):
            attribute_map["broadcast"] = 1
            attribute_map["transB"] = 1
        elif (layer_type == "ReLU"):
            relu = layer_param.relu_param
            slope = relu.negative_slope
            attribute_map["alpha"] = slope
                
        return attribute_map

    def extractNetworkInfo(self, network_input_info):
        input_output_map = collections.OrderedDict()
        inputsMap = {}
        outputsMap = {}
        outputNameAliasMap = {}
        layers = self.net_parameter.layer
        
        count = 0
        for i in range(len(layers)):
            layer_param = layers[i]
            layer_name = self.format_caffe_name(str(layer_param.name))
            layer_type = str(layer_param.type)
            inputs = layer_param.bottom
            outputs = layer_param.top

            layer_info = {}
            if (layer_type == "Data" or layer_type == "ImageData" or layer_type == "Input"):
                continue

            if (count == 0):
                input_keys = network_input_info.keys()
                in_name = self.format_caffe_name(input_keys[0])
                if (in_name in outputNameAliasMap):
                    in_name = outputNameAliasMap[in_name]
                layer_info["input"] = [in_name]
            else:
                if (layer_type == "Concat" or layer_type == "Eltwise"):
                    input_names = []
                    for j in range(len(inputs)):
                        in_name = self.format_caffe_name(str(inputs[j]))
                        if (in_name in outputNameAliasMap):
                            in_name = outputNameAliasMap[in_name]
                        input_names.append(in_name)
                    layer_info["input"] = input_names
                else:
                    in_name = self.format_caffe_name(str(inputs[0]))
                    if (in_name in outputNameAliasMap):
                        in_name = outputNameAliasMap[in_name]
                    layer_info["input"] = [in_name]


            if (layer_name != self.format_caffe_name(str(outputs[0]))):
                outputNameAliasMap[self.format_caffe_name(str(outputs[0]))] = layer_name

            attribute_map = self.extractCaffeAttrInfo(layer_param)
            layer_info["attributes"] = attribute_map
            layer_info["layer_type"] = layer_type
            layer_info["output"] = layer_name

            input_output_map[count] = layer_info

            print ("============= " + str(count) + "============")
            print (layer_info)
            count += 1
        
        return input_output_map        
        
def generate_caffe2_code(filename, output_folder, input_dims):
    caffe_util = CaffeUtil()
    caffe_util.loadNetworkFromCaffePrototxt(filename)
    input_info = caffe_util.extractInput(input_dims)
    print (input_info)
    input_output_map = caffe_util.extractNetworkInfo(input_info)

def main():
    
    if len(sys.argv) < 4:
        print ("Usage : python caffe2_code_generator.py <caffe prototxt file> <output folder> --input-dims n,c,h,w")
        sys.exit(1)

    filename = sys.argv[1]
    output_folder = sys.argv[2]
    input_dims = sys.argv[4].split(",")

    if not os.path.isfile(filename):
        print ("ERROR: unable to open the file.")
        sys.exit(1)

    if (".prototxt" not in filename):
        print ("ERROR: Improper file given. please give a deploy.prototxt file as input")
        sys.exit(1)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    generate_caffe2_code(filename, output_folder, input_dims)
        
if __name__ == '__main__':
    main()
