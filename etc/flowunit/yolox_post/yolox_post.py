# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _flowunit as modelbox
import json
import numpy as np
from yolox_utils import postprocess


class yolox_postFlowUnit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        # 获取功能单元的配置参数
        self.params = {}
        self.params['net_h'] = config.get_int('net_h')
        self.params['net_w'] = config.get_int('net_w')
        self.params['num_classes'] = config.get_int('num_classes')
        self.params['strides'] = config.get_int_list('strides')
        self.params['conf_thre'] = config.get_float('conf_threshold')
        self.params['nms_thre'] = config.get_float('iou_threshold')
        self.num_classes = config.get_int('num_classes')

        self.index = 0  # frame计数
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        # 从DataContext中获取输入输出BufferList对象
        in_feat = data_context.input("in_feat")
        out_data = data_context.output("out_data")

        # 循环处理每一个输入Buffer数据
        for buffer_feat in in_feat:
            # 将输入Buffer转换为numpy对象
            feat_data = np.array(buffer_feat.as_object(), copy=False)
            feat_data = feat_data.reshape((-1, self.num_classes + 5))

            # 业务处理：解码yolox模型的输出数据，得到检测框，转化为json数据
            bboxes = postprocess(feat_data, self.params)
            result = {"det_result": str(bboxes)}
            modelbox.debug(f'result for {self.index}-th image is {result}')
            self.index += 1

            # 将业务处理返回的结果数据转换为Buffer
            result_str = json.dumps(result)
            out_buffer = modelbox.Buffer(self.get_bind_device(), result_str)

            # 将输出Buffer放入输出BufferList中
            out_data.push_back(out_buffer)

        # 返回成功标志，ModelBox框架会将数据发送到后续的功能单元
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()