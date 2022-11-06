from __future__ import unicode_literals
import networkx
import chardet
import numpy as np
import random
import datetime
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from itertools import starmap
import nltk
import pysrt
import logging
logger = logging.getLogger("mainlogger")
def get_time():
    time = str(datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    return time
def setup_logger():
    fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
               '[%(thread)d][%(process)d] %(message)s')
    fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
    handler = logging.FileHandler('./log/main.log', encoding='utf-8')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
if not os.path.exists("/root/nltk_data"):
    nltk.download('punkt')

from moviepy.editor import VideoFileClip, concatenate_videoclips
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
import torch

"""
调用bert得到句子embedding
"""
def get_sentence_embedding(sentence, tokenizer, model):
    #tokenizer = BertTokenizer.from_pretrained('bert')
    #model = BertModel.from_pretrained('bert')
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    sequence_output = outputs[0]
    pooled_output = outputs[1].detach().numpy()[0]
    return pooled_output

class ExtractableAutomaticSummary:
    def __init__(self, article, tokenize, model):
        """
        抽取式自动摘要
        :param article: 文章内容，列表，列表元素为字符串，包含了文章内容，形如['完整文章']
        :param num_sentences: 生成摘要的句子数
        """
        self.article = article
        self.stopwords = None
        self.word_embeddings = {}
        self.sentences_vectors = []
        self.ranked_sentences = None
        self.similarity_matrix = None
        self.tokenize = tokenize
        self.model = model

    def __get_sentences(self, sentences):
        """
        断句函数
        :param sentences:字符串，完整文章，在本例中，即为article[0]
        :return:列表，每个元素是一个字符串，字符串为一个句子
        """
        sentences = re.sub('([（），。！？\?])([^”’])', r"\1\n\2", sentences)  # 单字符断句符
        sentences = re.sub('(\.{6})([^”’])', r"\1\n\2", sentences)  # 英文省略号
        sentences = re.sub('(\…{2})([^”’])', r"\1\n\2", sentences)  # 中文省略号
        sentences = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', sentences)
        sentences = sentences.rstrip()  # 段尾如果有多余的\n就去掉它
        return sentences.split("\n")

    def __get_simlarity_matrix(self):
        # 计算相似度矩阵，基于余弦相似度
        self.similarity_matrix = np.zeros((len(self.sentences_vectors), len(self.sentences_vectors)))
        for i in range(len(self.sentences_vectors)):
            for j in range(len(self.sentences_vectors)):
                if i != j:
                    self.similarity_matrix[i][j] = cosine_similarity(
                        self.sentences_vectors[i].reshape(1, -1), self.sentences_vectors[j].reshape(1, -1)
                    )

    def calculate(self):
        sentences = self.__get_sentences(self.article[0])
        # 将文章分割为句子
        for x in sentences:
            self.sentences_vectors.append(get_sentence_embedding(x, self.tokenize, self.model))
        self.__get_simlarity_matrix()
        # 获取句向量
        self.__get_simlarity_matrix()
        # 获取相似度矩阵
        nx_graph = networkx.from_numpy_array(self.similarity_matrix)
        scores = networkx.pagerank(nx_graph)
        # 将相似度矩阵转为图结构
        self.ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
        )
        # 排序

    def get_abstract(self, num_abstract_sentences):
        ans = []
        for i in range(num_abstract_sentences):
            ans.append(self.ranked_sentences[i][1])
        return ans


def caltime1(times):
    s = times.split(',')[0].split(':')
    x = times.split(',')[1]
    x = "0" * max(0, (3-len(x))) + x
    ans = str(int(s[0]) * 3600 + int(s[1]) * 60 + int(s[2])) + "." + x
    return ans

def get_text(srt_filename, retain, times, language="english"):
    srt_file = pysrt.open(srt_filename)
    enc = chardet.detect(open(srt_filename, "rb").read())['encoding']
    srt_file = pysrt.open(srt_filename, encoding=enc)
    # generate average subtitle duration
    dur = time_regions(map(srt_segment_to_range, srt_file))
    subtitle_duration = dur / len(srt_file)
    # compute number of sentences in the summary file
    n_sentences = int(dur * retain) / subtitle_duration
    summary = summarize(srt_file, n_sentences, language)
    sub_dur = dur / len(srt_file)
    # compute number of sentences in the summary file
    sentence = times / sub_dur
    return [summary, sentence]

def getstart_v4(fp, retain, subtitles, times, tokenize, model):
    dic = {}
    dicx = {}
    with open(fp, "r", encoding='UTF-8') as f:  # 此处根据文件格式使用'UTF-8'/'GBK'
        all_line_contents: list = f.readlines()  # 所有行的内容 -> all_line_contents
        #all_line_contents: list = f.read().splitlines()
        # 2.分行打印
        index = 0
       # article = ""
        for i in all_line_contents:
            if index <= 2:
                index += 1
                continue
            if index - 4 >= 0 and (index - 4) % 4 == 0:
                start = format(float(caltime1(i.split(' ')[0])), '.3f')
                end = format(float(caltime1(i.split(' ')[2][:-1])), '.3f')
            if index - 5 >= 0 and (index - 5) % 4 == 0:
                dic[i[:-1]+"。"] = (start, end)
                dicx[(float(start),float(end))] = i[:-1]
           #     article = article + i + "。"
            index += 1
    contents = get_text(subtitles, retain, times, "english")
    content = contents[0]
    nums = int(contents[1])
    article = ""
    for i in range(len(content)):
        key = (float(format(content[i][0], '.3f')), float(format(content[i][1], '.3f')))
        article = article + dicx[key] + "。"
    article = [article]
    demo = ExtractableAutomaticSummary(article, tokenize, model)
    demo.calculate()
    ans = demo.get_abstract(nums)
    temp = []
    for i in range(len(ans)):
        temp.append(dic[ans[i]])
    dics = {}
    for x in temp:
        k = int(float(x[0]))
        dics[k] = x
    res = sorted(dics.items(), key = lambda x:x[0])
    final = []
    for i in range(len(res)):
        final.append((float(res[i][1][0]),float(res[i][1][1])))
    return final

#imageio.plugins.ffmpeg.download()

def summarize(srt_file, n_sentences, language="english"):
    """ Generate segmented summary

    Args:
        srt_file(str) : The name of the SRT FILE
        n_sentences(int): No of sentences
        language(str) : Language of subtitles (default to English)

    Returns:
        list: segment of subtitles

    """
    parser = PlaintextParser.from_string(
        srt_to_txt(srt_file), Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    segment = []
    for sentence in summarizer(parser.document, n_sentences):
        index = int(re.findall("\(([0-9]+)\)", str(sentence))[0])
        item = srt_file[index]
        segment.append(srt_segment_to_range(item))
    return segment


def srt_to_txt(srt_file):
    """ Extract text from subtitles file

    Args:
        srt_file(str): The name of the SRT FILE

    Returns:
        str: extracted text from subtitles file

    """
    text = ''
    for index, item in enumerate(srt_file):
        if item.text.startswith("["):
            continue
        text += "(%d) " % index
        text += item.text.replace("\n", "").strip("...").replace(
                                     ".", "").replace("?", "").replace("!", "")
        text += ". "
    return text


def srt_segment_to_range(item):
    """ Handling of srt segments to time range

    Args:
        item():

    Returns:
        int: starting segment
        int: ending segment of srt

    """
    start_segment = item.start.hours * 60 * 60 + item.start.minutes * \
        60 + item.start.seconds + item.start.milliseconds / 1000.0
    end_segment = item.end.hours * 60 * 60 + item.end.minutes * \
        60 + item.end.seconds + item.end.milliseconds / 1000.0
    return start_segment, end_segment


def time_regions(regions):
    """ Duration of segments

    Args:
        regions():

    Returns:
        float: duration of segments

    """
    return sum(starmap(lambda start, end: end - start, regions))



def create_summary(filename, regions):
    """ Join segments

    Args:
        filename(str): filename
        regions():
    Returns:
        VideoFileClip: joined subclips in segment

    """
    subclips = []
    input_video = VideoFileClip(filename)
    last_end = 0
    for (start, end) in regions:
        subclip = input_video.subclip(start, end)
        subclips.append(subclip)
        last_end = end
    return concatenate_videoclips(subclips)



def get_v4_summary(filename,text_name, retain, times, tokenize, model):
    """ Abstract function

    Args:
        filename(str): Name of the Video file (defaults to "1.mp4")
        subtitles(str): Name of the subtitle file (defaults to "1.srt")

    Returns:
        True

    """
    logger.info('summary start')
    logger.info(get_time())
    subtitles = text_name.split('.')[0] + '.srt'
    regions = getstart_v4(text_name, retain, subtitles, times, tokenize, model)
    logger.info('summary end')
    logger.info(get_time())
    logger.info('splicing start')
    summary = create_summary(filename, regions)
    base, ext = os.path.splitext(filename)
    video_name = "{0}v4".format(base) + str(random.randint(1, 100000)) + str(datetime.datetime.now().strftime('%y%m%d%H%M%S'))
    output = video_name + '.mp4'
    summary.to_videofile(
                output,
                codec="libx264",
                temp_audiofile="temp.m4a", remove_temp=True, audio_codec="aac")
    logger.info('splicing end')
    logger.info(get_time())
    return output