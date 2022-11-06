#!/usr/bin/env python
# v1剪辑算法
from __future__ import unicode_literals
import os
import re
from itertools import starmap
import pysrt
import chardet
import nltk
import random
import datetime
if not os.path.exists("/root/nltk_data"):
    nltk.download('punkt')
from moviepy.editor import VideoFileClip, concatenate_videoclips
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
import logging
logger = logging.getLogger("mainlogger")
# 读取当前时间
def get_time():
    time = str(datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    return time
# 设置log
def setup_logger():
    fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
               '[%(thread)d][%(process)d] %(message)s')
    fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
    handler = logging.FileHandler('./log/main.log', encoding='utf-8')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 输入srt文件与句子数，生成summary
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

# 从srt中读取text
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

# 读取句子对应的时间片段与时长
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


def find_summary_regions(srt_filename, duration, language="english"):
    """ Find important sections

    Args:
        srt_filename(str): Name of the SRT FILE
        duration(int): Time duration
        language(str): Language of subtitles (default to English)

    Returns:
        list: segment of subtitles as "summary"

    """
    srt_file = pysrt.open(srt_filename)

    enc = chardet.detect(open(srt_filename, "rb").read())['encoding']
    srt_file = pysrt.open(srt_filename, encoding=enc)

    # generate average subtitle duration
    subtitle_duration = time_regions(
        map(srt_segment_to_range, srt_file)) / len(srt_file)
    # compute number of sentences in the summary file
    n_sentences = duration / subtitle_duration
    summary = summarize(srt_file, n_sentences, language)
    total_time = time_regions(summary)
    too_short = total_time < duration
    if too_short:
        while total_time < duration:
            n_sentences += 1
            summary = summarize(srt_file, n_sentences, language)
            total_time = time_regions(summary)
    else:
        while total_time > duration:
            n_sentences -= 1
            summary = summarize(srt_file, n_sentences, language)
            total_time = time_regions(summary)
    return summary


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
    for (start, end) in regions:
        subclip = input_video.subclip(start, end)
        subclips.append(subclip)
    return concatenate_videoclips(subclips)


def get_summary(filename, subtitles, times):
    """ Abstract function

    Args:
        filename(str): Name of the Video file (defaults to "1.mp4")
        subtitles(str): Name of the subtitle file (defaults to "1.srt")

    Returns:
        True

    """
    logger.info('summary start')
    regions = find_summary_regions(subtitles, times, "english")
    logger.info('summary end')
    logger.info(get_time())
    logger.info('splicing start')
    summary = create_summary(filename, regions)
    base, ext = os.path.splitext(filename)
    video_name = "{0}v1".format(base) + str(random.randint(1, 100000)) + str(datetime.datetime.now().strftime('%y%m%d%H%M%S'))
    output = video_name + '.mp4'
    summary.to_videofile(
                output,
                codec="libx264",
                temp_audiofile="temp.m4a", remove_temp=True, audio_codec="aac")
    logger.info('splicing end')
    logger.info(get_time())
    return output

# 主函数，返回剪辑后的video_name
def getstarted(video_file, subtitles_file, times):
    return get_summary(video_file, subtitles_file, times)
