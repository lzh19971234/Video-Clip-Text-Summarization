import os
from ast import literal_eval
from proto.get_cutting_pb2 import VideoAsrRequest
from proto.get_cutting_pb2_grpc import GetVideoAsrServiceStub
from kess.framework import ClientOption, GrpcClient
import logging
logger = logging.getLogger("mainlogger")

def setup_logger():
    fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
               '[%(thread)d][%(process)d] %(message)s')
    fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
    handler = logging.FileHandler('./log/main.log', encoding='utf-8')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def getres(ids, start, end, text):

    def sync(grpc_client: GrpcClient, ids, start, end, text):
        try:
            req = VideoAsrRequest(live_stream_id=ids, start_time=start, end_time=end, get_text=text)
            resp = grpc_client.GetVideoAsr(req)
            return resp
        except Exception as e:
            logger.error('发生异常, err: %s', e)

    client_option = ClientOption(
        biz_def='infra',
        grpc_service_name='grpc_getVideoAsrService',
        grpc_stub_class=GetVideoAsrServiceStub,
    )
    client = GrpcClient(client_option)
    return sync(client, ids, start, end, text)

import math
def caltime(time):
    haomiao = ','+str(int(1000 * (time - math.floor(time))))
    second = math.floor(time)
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    res = "%02d:%02d:%02d" % (h, m, s)
    return res+haomiao
def caltime1(st1,st2):
    return int(st2.split(':')[-1]) - int(st1.split(':')[-1]) if int(st2.split(':')[-1]) - int(st1.split(':')[-1]) >= 0 else int(st2.split(':')[-1]) - int(st1.split(':')[-1]) + 60
def merge(mess1,mess2):
    return [mess1[0],mess2[1],mess1[2] + mess2[2],str(int(mess1[3])+int(mess2[3]))]

"""有道翻译API的调用函数（封装为一个函数使用）"""
import json
import requests
def translator(str):
    """
    input : str 需要翻译的字符串
    output：translation 翻译后的字符串
    """
    # API
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    # 传输的参数， i为要翻译的内容
    key = {
        'type': "AUTO",
        'i': str,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    # key 这个字典为发送给有道词典服务器的内容
    response = requests.post(url, data=key)
    # 判断服务器是否相应成功
    if response.status_code == 200:
        # 通过 json.loads 把返回的结果加载成 json 格式
        result = json.loads(response.text)
#         print ("输入的词为：%s" % result['translateResult'][0][0]['src'])
#         print ("翻译结果为：%s" % result['translateResult'][0][0]['tgt'])
        translation = result['translateResult'][0][0]['tgt']
        return translation
    else:
        print("有道词典调用失败")
        # 相应失败就返回空
        return None

def getasr(ids, start_time, end_time, merges, get__asr, interval):
    file = open('./' + str(ids) + '.txt', 'w')
    file_name = str(ids) + '.txt'
    result = str(getres(ids, start_time, end_time, get__asr)).split("text: ")
    logger.info(f'asr_result: {result}')
    #  print(result)
    url = result[0].split('@')
    file.write("live_id: " + str(ids) + '  ')
    file.write("start_time: " + str(start_time) + '  ')
    file.write("end_time: " + str(end_time) + '\n')
    urls = url[0][9:] + '_' + url[1] + '_' + url[2][:-2]
    file.write(urls + '\n')
    file.write('\n')
    for i in range(1, len(result)):
        ans = result[i].split(' ')
        # 英文
        # text = translator(literal_eval("b'{}'".format(ans[0][1:])).decode('utf-8'))
        # 中文
        # print(ans)
        text = literal_eval("b'{}'".format(ans[0][1:])).decode('utf-8')
        start = caltime(float(ans[-2]))
        end = caltime(float(ans[-1][:-2]))
        file.write(str(i) + '\n')
        file.write(start + ' ')
        file.write("--> ")
        file.write(end + '\n')
        file.write(text + '\n')
        file.write('\n')
    file.close()
    with open('./' + file_name, "r", encoding='UTF-8') as f:  # 此处根据文件格式使用'UTF-8'/'GBK'
        all_line_contents: list = f.readlines()  # 所有行的内容 -> all_line_contents
        content = [all_line_contents[0][:-1], all_line_contents[1][:-1]]
        for i in range(3, len(all_line_contents), 4):
            content.append([all_line_contents[i + 1][:-1].split(' ')[0], all_line_contents[i + 1][:-1].split(' ')[-1],
                            all_line_contents[i + 2][:-1], int(caltime1(all_line_contents[i + 1].split(' ')[0][:8],
                                                                        all_line_contents[i + 1].split(' ')[-1][:8]))])
    newcontent = [content[0], content[1]]
    index = 1
    temp = []
    time = merges
    for i in range(2, len(content)):
        if temp == []:
            if int(content[i][-1]) > time:
                newcontent.append([str(index)] + [str(x) for x in content[i]])
                index += 1
            else:
                temp = content[i]
        else:
            if int(content[i][-1]) > time or float(content[i][0].split(':')[-1].split(',')[0]) - float(temp[1].split(':')[-1].split(',')[0]) > interval or int(content[i][-1]) + int(temp[-1]) > time:
                newcontent.append([str(index)] + temp)
                newcontent.append([str(index + 1)] + content[i])
                index += 2
                temp = []
            else:
                temp = merge(temp, content[i])
    if temp:
        newcontent.append([str(index)] + temp)
    file1 = open('./' + str(ids) + '_' + str(time) + '.txt', 'w')
    file1.write(newcontent[0] + '\n')
    file1.write(newcontent[1] + '\n')
    file1.write('\n')
    for i in range(2, len(newcontent)):
        file1.write(newcontent[i][0] + '\n')
        file1.write(newcontent[i][1] + " --> " + newcontent[i][2] + '\n')
        file1.write(newcontent[i][3] + '\n')
        file1.write('\n')
    file1.close()
    os.remove(file_name)
    return str(ids) + '_' + str(time), content[1]

def getsrt(file_name):
    file = open('./' + file_name + '.srt', 'w')
    fp = file_name + ".txt"
    with open(fp, "r", encoding='UTF-8') as f:  # 此处根据文件格式使用'UTF-8'/'GBK'
        all_line_contents: list = f.readlines()  # 所有行的内容 -> all_line_contents
        # 2.分行打印
        index = 0
        for i in all_line_contents:
            if index <= 2:
                index += 1
                continue
            if index - 5 >= 0 and (index - 5) % 4 == 0:
                i = translator(i)
                file.write(i)
                file.write('\n')
            else:
                file.write(i)
            index += 1
    file.close()

