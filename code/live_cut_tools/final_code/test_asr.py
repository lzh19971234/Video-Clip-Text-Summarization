#!/usr/bin/env python
import datetime
import yaml
from infra.kafka import KafkaProducers
from proto.get_cutting_pb2 import SentItemRes
from proto.get_cutting_pb2 import LiveClipSentSelectResponse
from proto.get_cutting_pb2 import LiveClipSentSelectRequest
from get_sents_algorithm.v2_sent import get_summary_v2
import logging
import numpy as np

import os
from infra.kafka import (
    ConsumerParameter,
    KsKafkaConsumer,
    MessageContext,
    FinishConsumeException
)

logger = logging.getLogger("mainlogger")

def get_word_embeddings():
    with open('./word/sgns.sogou.char', encoding='utf-8') as f:
        lines = f.readlines()
        word_embeddings = {}
        for _, line in enumerate(lines):
            if _ != 0:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs
    return word_embeddings

def get_time():
    time = str(datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    return time

# 解析config文件
def get_yaml_data(yaml_file):
    # 打开yaml文件
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    # 将字符串转化为字典或列表
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data

def setup_logger():
    fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
               '[%(thread)d][%(process)d] %(message)s')
    fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
    handler = logging.FileHandler('./log/main.log', encoding='utf-8')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class GetCut(object):
    def __init__(self, times, word_embeddings):
        self.times = times
        self.word_embeddings = word_embeddings

    def cutting(self, asr_req):

        # stage 2: 开始剪辑
        logger.info('start v2')
        regions, asr, score = get_summary_v2(asr_req, self.times, self.word_embeddings)
        logger.info('v2 finish')
       # key = str(live_id) + typee + get_time()
       # source_key = self.store(key, sent_result)
        msg = LiveClipSentSelectResponse()
        for i in range(len(regions)):
            res = SentItemRes()
            res.start_time = regions[i][0]
            res.end_time = regions[i][1]
            res.sent_asr = asr[i]
            res.score = score[i]
            msg.res_items.append(res)
        logger.info(f'msg: {msg}')
        return msg

class TestNewConsumer(KsKafkaConsumer):
    def __init__(self, parameter, times, word_embeddings):
        super().__init__(parameter)
        self.times = times
        self.word_embeddings = word_embeddings

    def consume(self, message: bytes, context: MessageContext):

        logger.info(f'message: {message}, topic_name: {context.topic_name}, '
                    f'partition: {context.partition}, '
                    f'offset: {context.offset}')
        logger.info('---------')

        ####
        # 获得request，调用tackle返回response
        request = LiveClipSentSelectRequest()
        request.ParseFromString(message)
        tpc = "ad_material_industry_live_clip_algo_response"
        response = self.tackle(request)
        KafkaProducers.send(tpc, response)
        ####
        # raise FinishConsumeException()

        if message == bytes('produce over', 'utf8'):
            logger.info("consume over")
            # 在消费逻辑中可以使用FinishConsumeException来终止消费，
            # 该场景适用于group中只有一个consumer的情况
            raise FinishConsumeException()

    # 处理request剪辑, 并返回response
    def tackle(self, request):

        logger.info(f'request: {request}')
        req = request.req_items
        asr_req = []
        for x in req:
            tmp = [(x.start_time, x.end_time), x.sent_asr]
            asr_req.append(tmp)
        parameters = request.parameters

        try:
            cut = GetCut(self.times, self.word_embeddings)
            msg = cut.cutting(asr_req)
        except Exception as e:
            logger.exception(e)
            msg = LiveClipSentSelectResponse()
        msg.parameters = parameters
        logger.info(f'response: {msg}')
        return msg.SerializeToString()



if __name__ == "__main__":
    current_path = os.path.abspath("./config/")
    yaml_path = os.path.join(current_path, "config.yaml")
    data = get_yaml_data(yaml_path)
    times = data['total_time']

    setup_logger()
    topic = 'ad_material_industry_live_clip_algo_request'
    ##
    group_id = 'ad_material_industry_live_clip'
    ##
    word_embeddings = get_word_embeddings()
    # 构建消费者
    parameter = ConsumerParameter(topic, group_id, True)
    consumer = TestNewConsumer(parameter, times, word_embeddings)
    consumer.start()