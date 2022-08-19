"""
Utils for handling/parsing data from vtt files
"""

from datetime import datetime, timedelta
import webvtt
from os import listdir
import re
from seaborn import histplot
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from winsound import PlaySound, SND_FILENAME
from numpy import ndarray

SAMPLING_RATE = 22050
WORD_DURATION_THRESHOLD = 1500  # 1.5s

def timestamp_to_millisec(t):
    dt = datetime.strptime(t, "%H:%M:%S.%f")
    td = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
    return int(td / timedelta(milliseconds=1))

def duration_of_line(line):
    return timestamp_to_millisec(line.end) - timestamp_to_millisec(line.start)

music_count = 0
def prune_lines(vtt_file):
    """
    Prune out lines too short, empty lines, stuff like [music].
    Also removes outliers wrt word density per line.
    Returns list of pruned lines.
    """
    # TODO: there will be sentences with words at the end that are dragged behind
    # go word by word, if we find a word that has extended duration, cut off there
    pruned_lines = []
    for line in vtt_file:
        if duration_of_line(line) < 20:
            continue
        line_text = line.text.strip().split('\n')[-1]
        if line_text.isspace() or re.match('.*\[[A-Za-z]+\]', line_text):
            # print("found empty or [music] stuff", line_text)
            global music_count
            music_count += 1
            continue
        # # TODO: remove outliers
        # if False:
        #     continue
        pruned_lines.append(line)
    # print(music_count)
    return pruned_lines

def ms_to_sample(t):
    return int(SAMPLING_RATE / 1000 * t)

def prepare_training_audio(audio_file_names):
    """
    Takes a list of audio files, find the matching vtt file for each, and
    cut the audio accordingly, with some cleaning/checks
    """
    # TODO: walk through each line, and cutoff sentence when reaching a word with extended duration (1500ms?)
    wav_segment_list = []
    for audio_file in audio_file_names:
        sr, audio = read(audio_file)
        assert sr == SAMPLING_RATE
        vtt_file = audio_file.replace('audio', 'vtt').replace('wav','vtt')
        vtt = prune_lines(webvtt.read(vtt_file))
        print(vtt_file)
        cut_count = 0

        for line in vtt:
            line_start, line_end = timestamp_to_millisec(line.start), timestamp_to_millisec(line.end)
            line_text = line.raw_text.replace('<c>', '').replace('</c>', '')
            splits = re.split(" *<(.+?)> +", line_text)
            splits.insert(0, line.start)
            splits.append(line.end)
            cutoff = False
            for i in range(0, len(splits), 2):
                if i == len(splits) - 1:
                    break
                word_start, word_end = timestamp_to_millisec(splits[i]), timestamp_to_millisec(splits[i+2])
                if word_end - word_start >= WORD_DURATION_THRESHOLD:
                    if i == 0:
                        # it's the first word, sth gotta be wrong with the line, skip
                        line_end = line_start  # hacky, just to skip it
                        break
                    # cutoff line here
                    # print("cutting off at word", splits[i+1])
                    line_end = word_start
                    cutoff = True
                    cut_count += 1
                    break
            if line_start != line_end:
                # cut the wav segment, use copy so as to later free the whole audio data
                wav_segment_list.append(
                    audio[ms_to_sample(line_start) : ms_to_sample(line_end)].copy())
                # print("added wav seg to list")
                # if cutoff:
                #     print("writing file to play")
                #     write('test.wav', sr, audio[ms_to_sample(line_start) : ms_to_sample(timestamp_to_millisec(line.end))])
                #     PlaySound('test.wav', SND_FILENAME)
                #     write('test.wav', sr, audio[ms_to_sample(line_start) : ms_to_sample(line_end)])
                #     PlaySound('test.wav', SND_FILENAME)
                # else:
                #     print("didn't cutoff")
        print(cut_count)
    del audio  # free memory, only keep the segments which are copied (not a view of `audio`)
    return wav_segment_list


# vs = listdir('./files/vtt')
# ratios = []
# line_total = 0
# long_lines = 0
# for file_name in vs:
#     file = webvtt.read('./files/vtt/' + file_name)
#     pruned = prune_lines(file)
#     for line in pruned:
#         line_total += 1
#         line_dur = duration_of_line(line)/1000
#         if line_dur >= 5.0:
#             long_lines += 1
#         if line_dur > 8.0:
#             ratios.append(len(line.text.strip().split(' ')) / line_dur)
# print("lines:", line_total, long_lines, long_lines / line_total)
# print("music total", music_count)
# histplot(ratios)
# plt.show()
# import numpy as np
# ra=np.array(ratios)
# print(ra.mean(), np.median(ra), np.std(ra), np.percentile(ra,[1,3,5,90,95]))

