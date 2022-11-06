#!/usr/bin/env python
import cv2
import datetime
import proto.get_cutting_pb2
import proto.get_cutting_pb2_grpc
import yaml
from infra.kafka import KafkaProducers
from proto.get_cutting_pb2 import LiveClipAlgRequest
from proto.get_cutting_pb2 import AlgClipType

from cutting_algorithm.v1 import getstarted
from cutting_algorithm.v2 import get_summary
from cutting_algorithm.v3 import get_v3_summary
from cutting_algorithm.v4 import get_v4_summary
from get_asr import getsrt
import wget
import logging
import numpy as np
from storage import (
    BlobStore,
    BlobStoreError,
    MARK_FAKE_DELETE,
    MARK_NORMAL,
    close_bloblstore_resource,
)
import os
import subprocess
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
# 获取视频时长
def get_video_duration(video_path: str):
    ext = os.path.splitext(video_path)[-1]
    if ext != '.mp4' and ext != '.avi' and ext != '.flv':
        raise Exception('format not support')
    ffprobe_cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'
    p = subprocess.Popen(
        ffprobe_cmd.format(video_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    out, err = p.communicate()
    duration_info = float(str(out, 'utf-8').strip())
    return duration_info
# 获取视频长宽
def get_lw(video_name):
    vcap = cv2.VideoCapture(video_name) # 0=camera
    if vcap.isOpened():
        width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        #print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) # 3, 4

        # or
        width  = int(vcap.get(3)) # float
        height = int(vcap.get(4)) # float
        return [height, width]
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
    def __init__(self, times, aasr, merge, interval, retain, word_embeddings):
        self.times = times
        self.aasr = aasr
        self.merge = merge
        self.interval = interval
        self.retain = retain
        self.word_embeddings = word_embeddings

    def cutting(self, live_id, start_time, end_time, typ):
        # stage 1: 先拿视频和ASR结果
        url = "https://js-ad.a.yximgs.com/bs2/ad-material-opt-tmp/d9853f36-849c-4ed4-b337-add2cdbfea85.mp4"
        text_name = "9873615864_7"
        #text_name, url = getasr(live_id, start_time, end_time, self.merge, self.aasr, self.interval)
        #getsrt(text_name)
        live_id = 9873615864
        url2 = "https://js-ad.a.yximgs.com/bs2/ad-material-opt-tmp/9873615864_7.txt"
        url3 ="https://js-ad.a.yximgs.com/bs2/ad-material-opt-tmp/9873615864_7.srt"
        wget.download(url, out=str(live_id) + ".mp4")
        wget.download(url2, out=text_name + ".txt")
        wget.download(url3, out=text_name + ".srt")

        # stage 2: 开始剪辑
        if typ == AlgClipType.WORD_COUNTS_CLIP:
            # get v1 result
            logger.info('start v1')
            getsrt(text_name)
            video_name = getstarted(str(live_id) + ".mp4", text_name + ".srt", self.times)
            logger.info('v1 finish')
        elif typ == AlgClipType.TEXTRANK_CLIP:
            # get v2 result
            logger.info('start v2')
            video_name = get_summary(str(live_id) + ".mp4", text_name + ".txt", self.retain, self.times, self.word_embeddings)
            logger.info('v2 finish')
        elif typ == AlgClipType.BAGGING_CLIP:
            # get v3 result
            logger.info('start v3')
            video_name = get_v3_summary(str(live_id) + ".mp4", text_name + ".txt", self.retain, self.times)
            logger.info('v3 finish')
        elif typ == AlgClipType.BERT_GRAPH_CLIP:
            # get v4 result
            logger.info('start v4')
            video_name = get_v4_summary(str(live_id) + ".mp4", text_name + ".txt", self.retain, self.times)
            logger.info("v4 finish")
        else:
            return
        logger.info(get_time())
        os.remove(str(live_id) + ".mp4")
        os.remove(text_name + ".srt")
        os.remove(text_name + ".txt")
        # 存入blob
        logger.info("store in blob")
        key = self.store(video_name)
        logger.info(get_time())
        # 获取其他参数
        logger.info("get other parameters")
        duration = int(get_video_duration(video_name))
        [height, width] = get_lw(video_name)
        logger.info(get_time())
        #os.remove(video_name)
        return [key, duration, height, width]

    def store(self, video_name):
        # 构建 BlobStore 资源对象
        upload_test = BlobStore('ad', 'material-opt-tmp')
        key = video_name
        with open(video_name, 'rb') as f:
            # save
            try:
                upload_test.save(key=key, value=f)
                logger.info(f'key: {key}')
            except BlobStoreError as e:
                logger.info(e)

        # get
        # try:
        #    value = upload_test.get(key = "ffasafafsasfasf.mp4")
        # except BlobStoreError as e:
        #    logger.error(e)
        # 程序退出前，请显示调用，释放 blobstore 资源
        # close_bloblstore_resource()
        return "ad" + "_" + "material-opt-tmp" + '_' + key

    # 继承infra组提供的抽象类KsKafkaConsumer，并实现consume函数即可。
    # 业务线具体的消费逻辑在consume中完成，如TestNewConsumer:


# 继承infra组提供的抽象类KsKafkaConsumer，并实现consume函数即可。
# 业务线具体的消费逻辑在consume中完成，如TestNewConsumer:
class TestNewConsumer(KsKafkaConsumer):
    def __init__(self, parameter, times, aasr, merge, interval, retain, word_embeddings):
        super().__init__(parameter)
        self.times = times
        self.aasr = aasr
        self.merge = merge
        self.interval = interval
        self.retain = retain
        self.word_embeddings = word_embeddings

    def consume(self, message: bytes, context: MessageContext):

        logger.info(f'message: {message}, topic_name: {context.topic_name}, '
                    f'partition: {context.partition}, '
                    f'offset: {context.offset}')
        logger.info('---------')

        ####
        # 获得request，调用tackle返回response
        request = LiveClipAlgRequest()
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
        try:
            logger.info(f'request: {request}')
            logger.info(get_time())
            live_id = request.live_stream_id
            start_time = request.start_time
            end_time = request.start_time
            typ = request.clip_type
            parameters = request.parameters

            cut = GetCut(self.times, self.aasr, self.merge, self.interval, self.retain, self.word_embeddings)
            [key, duration, height, width] = cut.cutting(live_id, start_time, end_time, typ)

            msg = proto.get_cutting_pb2.LiveClipAlgResponse(resource_key=key, video_height=height, video_width=width,
                                                             video_duration=duration, parameters=parameters)
            logger.info(f'response: {msg}')
            return msg.SerializeToString()
        except Exception as e:
            logger.exception(e)
            msg = proto.get_cutting_pb2.LiveClipAlgResponse(resource_key="", video_height=0, video_width=0,
                                                         video_duration=0, parameters=parameters)
            logger.info(f'response: {msg}')
            response = msg.SerializeToString()
            return response


if __name__ == "__main__":
    current_path = os.path.abspath("./config/")
    yaml_path = os.path.join(current_path, "config.yaml")
    data = get_yaml_data(yaml_path)
    aasr = data['is_asr']
    times = data['total_time']
    merge = data['merge_second']
    interval = data['interval_time']
    retain = data['retain_ratio']

    setup_logger()
    topic = 'ad_material_industry_live_clip_algo_request'
    ##
    group_id = 'ad_material_industry_live_clip'
    ##
    word_embeddings = get_word_embeddings()

    # 构建消费者
    parameter = ConsumerParameter(topic, group_id, True)
    consumer = TestNewConsumer(parameter, times, aasr, merge, interval, retain, word_embeddings)


    #def close_consume():
    #    time.sleep(18)
    #    consumer.close()


    # 开启停止消费的线程
    #threading.Thread(target=close_consume).start()

    # 开始消费，注意：如果是python3.9+环境，请将block设置为True，否则将会报错。
    # Python3.9之后对GC进行了优化：主线程退出之后，主线程内的资源将被回收。
    # https://docs.python.org/3/whatsnew/3.9.html#gc
    # consumer.start(block=True)
    consumer.start()