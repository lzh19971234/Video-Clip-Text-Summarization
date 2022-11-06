from __future__ import unicode_literals
import networkx
import chardet
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import os
import re
from itertools import starmap
import datetime
import nltk
import pysrt
import logging
logger = logging.getLogger("mainlogger")
def get_time():
    time = str(datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    return time
#imageio.plugins.ffmpeg.download()
def setup_logger():
    fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
               '[%(thread)d][%(process)d] %(message)s')
    fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
    handler = logging.FileHandler('./log/main.log', encoding='utf-8')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
#imageio.plugins.ffmpeg.download()
if not os.path.exists("/root/nltk_data"):
    nltk.download('punkt')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer


class ExtractableAutomaticSummary:
    def __init__(self, article, word_embeddings):
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
        self.word_embeddings = word_embeddings

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

    def __get_stopwords(self):
        # 加载停用词
        self.stopwords = [line.strip() for line in open('./word/cn_stopwords.txt', encoding='utf-8').readlines()]

    def __remove_stopwords_from_sentence(self, sentence):
        # 去除停用词
        sentence = [i for i in sentence if i not in self.stopwords]
        return sentence

    #def __get_word_embeddings(self):
    #    with open('./word/sgns.sogou.char', encoding='utf-8') as f:
    #        lines = f.readlines()
    #        for _, line in enumerate(lines):
    #            if _ != 0:
    #                values = line.split()
    #                word = values[0]
    #                coefs = np.asarray(values[1:], dtype='float32')
    #                self.word_embeddings[word] = coefs

    def __get_sentence_vectors(self, cutted_clean_sentences):
        # 获取句向量，将句子中的每个词向量相加，再取均值，所得即为句向量
        for i in cutted_clean_sentences:
            if len(i) != 0:
                v = sum(
                    [self.word_embeddings.get(w.strip(), np.zeros((300,))) for w in i]
                ) / (len(i) + 1e-2)
            else:
                v = np.zeros((300,))
                # 因为预训练的词向量维度是300
            self.sentences_vectors.append(v)

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
        logger.info(get_time())
        logger.info('get_word_embeddings()')

        #self.__get_word_embeddings()
        # 获取词向量
        logger.info(get_time())
        logger.info('get_stopwords()')

        self.__get_stopwords()
        # 获取停用词
        logger.info(get_time())
        logger.info('get_sentences')

        sentences = self.__get_sentences(self.article[0])
        # 将文章分割为句子
        logger.info(get_time())
        logger.info('cutted_sentences')
        cutted_sentences = [jieba.lcut(s) for s in sentences]
        # 对每个句子分词
        logger.info(get_time())
        logger.info('cutted_clean_sentences')
        cutted_clean_sentences = [self.__remove_stopwords_from_sentence(sentence) for sentence in cutted_sentences]
        # 句子分词后去停用词
        # 先分词，再去停用词，直接去停用词会把每个字分开，比如变成‘直 接 去 停 用 词 会 把 每 个 字 分 开’
        logger.info(get_time())
        logger.info('get_sentence_vectors')
        self.__get_sentence_vectors(cutted_clean_sentences)
        # 获取句向量
        logger.info(get_time())
        logger.info('get_simlarity_matrix')
        self.__get_simlarity_matrix()
        # 获取相似度矩阵
        logger.info(get_time())
        logger.info('networkx')
        nx_graph = networkx.from_numpy_array(self.similarity_matrix)
        self.scores = networkx.pagerank(nx_graph)
        # 将相似度矩阵转为图结构
        logger.info(get_time())
        logger.info('rank')
        self.ranked_sentences = sorted(
            ((self.scores[i], s) for i, s in enumerate(sentences)), reverse=True
        )
        # 排序
        self.scores = sorted(self.scores, reverse=True)
    def get_abstract(self, num_abstract_sentences):
        ans = []
        for i in range(num_abstract_sentences):
            ans.append(self.ranked_sentences[i][1])
        return ans, self.scores[:num_abstract_sentences]


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
    return [summary, sentence - 3]

#def getstart_v2(fp, retain, subtitles, times, word_embeddings):
def getstart_v2(asr_req, times, word_embeddings):
    dic = {}
    article = ""
    for x in asr_req:
        start = format(x[0][0], '.3f')
        end = format(x[0][1], '.3f')
        dic[x[1]+"。"] = (start, end)
        article = article + x[1]+"。"
           #     article = article + i + "。"
    nums = len(asr_req)
    article = [article]
    demo = ExtractableAutomaticSummary(article, word_embeddings)
    demo.calculate()
    ans, scores = demo.get_abstract(nums)
    total, index = 0, 0
    while total < times:
        total += (dic[ans[index]][1] - dic[ans[index]][0])
        index += 1
    ans = ans[:index]
    scores = scores[:index]
    temp = []
    dicc = {}
    for i in range(len(ans)):
        temp.append(dic[ans[i]])
        dicc[dic[ans[i]]] = ans[i]
    dics = {}
    for x in temp:
        k = int(float(x[0]))
        dics[k] = x
    res = sorted(dics.items(), key = lambda x:x[0])
    final = []
    sentences = []
    for i in range(len(res)):
        final.append((float(res[i][1][0]), float(res[i][1][1])))
        sentences.append(dicc[(res[i][1][0], res[i][1][1])].encode(encoding = 'utf-8').decode(encoding = 'utf-8'))
    return final, sentences, scores

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

def get_summary_v2(asr_req, times, word_embeddings):
    """ Abstract function

    Args:
        filename(str): Name of the Video file (defaults to "1.mp4")
        subtitles(str): Name of the subtitle file (defaults to "1.srt")

    Returns:
        True

    """
    final, sentences, scores = getstart_v2(asr_req, times, word_embeddings)
    return final, sentences, scores
