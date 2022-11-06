from __future__ import unicode_literals
import chardet
import os
import datetime
import re
from itertools import starmap
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

from moviepy.editor import VideoFileClip, concatenate_videoclips
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
import macropodus
import collections

def norms(num):
    sums = sum([x[0] for x in num])
    ans = []
    for x in num:
        ans.append([float(x[0])/sums, x[1]])
    return ans

def getsummary(summary, n):
# 文本摘要(summarization, 可定义方法, 提供9种文本摘要方法, 'lda', 'mmr', 'textrank', 'text_teaser')
    sents1 = macropodus.summarization(text=summary, type_summarize="lda")
    sents2 = macropodus.summarization(text=summary, type_summarize="mmr")
    sents3 = macropodus.summarization(text=summary, type_summarize="textrank")
    sents4 = macropodus.summarization(text=summary, type_summarize="text_teaser")
    sents5 = macropodus.summarization(text=summary, type_summarize="text_pronouns")
    sents6 = macropodus.summarization(text=summary, type_summarize="word_sign")
    sents7 = macropodus.summarization(text=summary, type_summarize="lead3")
    sents8 = macropodus.summarization(text=summary, type_summarize="lsi")
    sents9 = macropodus.summarization(text=summary, type_summarize="nmf")
    sents = []
    num = 9
    for i in range(num):
        sents = sents + norms(eval("sents" + str(i+1)))
    #sents = sents + norms(eval("sents" + str(3)))
    dic = collections.defaultdict(float)
    for i in range(len(sents)):
        dic[sents[i][1]] += sents[i][0]
    for key in dic:
        dic[key] = dic[key] / num
    final = sorted(dic.items(), key=lambda x: x[0], reverse=True)
    ans = []
    scores = []
    for i in range(n):
        ans.append(final[i][0])
        scores.append(final[i][1])
    return ans, scores

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

def getstart_v3(fp, retain, subtitles, times):
    dic = {}
    dicx = {}
    with open(fp, "r", encoding='UTF-8') as f:  # 此处根据文件格式使用'UTF-8'/'GBK'
        #all_line_contents: list = f.readlines()  # 所有行的内容 -> all_line_contents
        all_line_contents: list = f.read().splitlines()
        # 2.分行打印
        index = 0
        article = ""
        for i in all_line_contents:
            if index <= 2:
                index += 1
                continue
            if index - 4 >= 0 and (index - 4) % 4 == 0:
                start = format(float(caltime1(i.split(' ')[0])), '.3f')
                end = format(float(caltime1(i.split(' ')[2])), '.3f')
            if index - 5 >= 0 and (index - 5) % 4 == 0:
                dic[i+"。"] = (start, end)
                dicx[(float(start),float(end))] = i
                #article = article + i + "。"
            index += 1
    contents = get_text(subtitles, retain, times,  "english")
    content = contents[0]
    nums = int(contents[1])
    article = ""
    for i in range(len(content)):
        try:
            key = (float(format(content[i][0], '.3f')), float(format(content[i][1], '.3f')))
            article = article + dicx[key] + "。"
        except:
            continue
    article = str(article)
    ans, scores = getsummary(article, nums)
    temp = []
    dicc = {}
    for i in range(len(ans)):
        temp.append(dic[ans[i] + '。'])
        dicc[dic[ans[i]+'。']] = ans[i]+"。"
    dics = {}
    for x in temp:
        k = int(float(x[0]))
        dics[k] = x
    res = sorted(dics.items(), key = lambda x:x[0])
    final = []
    sentences = []
    for i in range(len(res)):
        final.append((float(res[i][1][0]), float(res[i][1][1])))
        sentences.append(dicc[(res[i][1][0], res[i][1][1])].encode(encoding='utf-8').decode(encoding='utf-8'))
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

def get_v3_summary(text_name, retain, times):
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
    regions = getstart_v3(text_name, retain, subtitles, times)
    return regions
