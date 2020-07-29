# Author: Eric Cao
# Created: 2020/07/20

import librosa.display
import librosa
import matplotlib.pyplot as plt
import matplotlib
import numpy
from madmom.features.chords import CNNChordFeatureProcessor
from madmom.features.chords import CRFChordRecognitionProcessor

classification = ["美妆护理", "男装", "女装", "日用百货", "食品生鲜", "手机数码"]
id_map = [(1, 208), (209, 400), (401, 577), (578, 895), (896, 1154), (1155, 1426)]
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 频率、功率、节拍/节奏、音调、音量


# 波形图
def get_wave_plot(x):
    librosa.display.waveplot(x)
    plt.show()


# 频谱图
def get_spectrogram(x):
    # 短时傅里叶变换
    S = numpy.abs(librosa.stft(x))
    # 转分贝
    Sdb = librosa.power_to_db(S)
    return Sdb


def show_spectrogram():
    plt.figure(figsize=(12, 16))
    for i in range(6):
        x = numpy.loadtxt(open('sptg_pro/' + str(i) + '.csv', 'rb'), delimiter=',', skiprows=0)
        plt.subplot(3, 2, i+1)
        plt.subplots_adjust(hspace=0.5)
        librosa.display.specshow(x, x_axis='time', y_axis='hz')
        plt.xlabel('时间/s')
        plt.ylabel('频率/Hz')
        plt.title(classification[i] + ' 频谱图（Spectrogram）')
        cb = plt.colorbar()
    plt.show()


# 过零率
def get_zero_crossing_rate(x):
    zero_crossings = librosa.zero_crossings(x)
    count = sum(zero_crossings)
    zero_crossings_rate = count / zero_crossings.shape[0]
    return zero_crossings_rate


def show_zero_crossing_rate():

    def get_color(y):
        color = []
        for i in range(len(y)):
            if y[i] < 0.08:
                color.append("lightgreen")
            elif y[i] < 0.09:
                color.append("skyblue")
            else:
                color.append("orange")
        return color

    plt.figure(figsize=(10, 6))
    # 平均过零率
    zcr = [0.0869, 0.0927, 0.0853, 0.0847, 0.0868, 0.0959]
    plt.bar(x=classification, height=zcr, width=0.6, color=get_color(zcr))
    plt.title("平均过零率（Average Zero Crossing Rate）")
    for a, b in zip(classification, zcr):
        plt.text(a, b + 0.0005, b, ha='center', va='bottom')
    plt.show()


# 节拍
def get_tempo(x):
    tp, beat_frames = librosa.beat.beat_track(x)
    return tp


def show_tempo():

    def get_color(y):
        color = []
        for i in range(len(y)):
            if y[i] < 110:
                color.append("lightgreen")
            elif y[i] < 125:
                color.append("skyblue")
            else:
                color.append("orange")
        return color

    plt.figure(figsize=(10, 6))
    # 平均节奏
    tempo = [118.38, 117.26, 117.80, 117.59, 120.97, 122.64]
    plt.bar(x=classification, height=tempo, width=0.6, color=get_color(tempo))
    plt.title("平均拍速（Average Tempo）")
    plt.ylabel("每分钟拍数")

    for a, b in zip(classification, tempo):
        plt.text(a, b + 0.5, b, ha='center', va='bottom')
    plt.show()


# 调谐偏差
def get_tune(x):
    t = librosa.estimate_tuning(x)
    return t


def show_tune():

    def get_color(y):
        color = []
        for i in range(len(y)):
            if abs(y[i]) < 0.0025:
                color.append("lightgreen")
            elif abs(y[i]) < 0.005:
                color.append("skyblue")
            else:
                color.append("orange")
        return color

    plt.figure(figsize=(10, 6))
    # 平均调谐偏差
    tuning_deviation = [0.0075, -0.0085, 0.0013, 0.0035, 0.0066, -0.0048]
    plt.bar(x=classification, height=tuning_deviation, width=0.6, color=get_color(tuning_deviation))
    plt.title("平均调谐偏差（Average Tuning Deviation）")
    for a, b in zip(classification, tuning_deviation):
        if b > 0:
            plt.text(a, b + 0.0001, b, ha='center', va='bottom')
        else:
            plt.text(a, b - 0.0001, b, ha='center', va='top')
    plt.show()


# MFCC(梅尔频率倒谱系数)
def get_mfcc(x):
    m = librosa.feature.mfcc(x)
    # 20 * 345
    return m


def show_mfcc():
    plt.figure(figsize=(12, 16))
    for i in range(6):
        x = numpy.loadtxt(open('mfcc_pro/' + str(i) + '.csv', 'rb'), delimiter=',', skiprows=0)
        plt.subplot(3, 2, i+1)
        plt.subplots_adjust(hspace=0.5)
        librosa.display.specshow(x, x_axis='time')
        plt.xlabel('时间/s')
        plt.colorbar()
        plt.title(classification[i] + " 梅尔频率倒谱系数（MFCC）")
    plt.show()


# 和弦
def get_chord(music):
    featproc = CNNChordFeatureProcessor()
    decode = CRFChordRecognitionProcessor()
    feats = featproc(music)
    chords = decode(feats)
    chord_dict = {}
    for chord in chords:
        if chord[2] not in chord_dict:
            chord_dict[chord[2]] = chord[1] - chord[0]
    return chord_dict


def show_chords():

    # 此处记录已经跑出来的结果，方便直接展示
    chords = [{'N': 183.9, 'F#:maj': 49.3, 'F:min': 39.6, 'A#:min': 30.0, 'G#:min': 30.3, 'C#:maj': 58.9, 'E:maj': 56.4, 'D#:min': 12.9, 'C#:min': 32.7, 'A:maj': 76.8, 'B:maj': 71.8, 'F#:min': 42.9, 'D:maj': 87.0, 'D:min': 23.8, 'F:maj': 68.3, 'E:min': 58.5, 'G:maj': 105.3, 'C:maj': 88.7, 'B:min': 34.1, 'C:min': 30.7, 'D#:maj': 58.9, 'G#:maj': 56.4, 'A#:maj': 53.7, 'A:min': 65.5, 'G:min': 19.2},
              {'N': 137.6, 'B:maj': 72.1, 'A:maj': 96.0, 'G#:min': 35.7, 'C#:min': 45.0, 'A:min': 54.6, 'C:maj': 55.6, 'G:maj': 76.7, 'D:maj': 52.1, 'B:min': 78.5, 'F#:min': 28.9, 'E:min': 155.3, 'E:maj': 61.1, 'F:maj': 42.5, 'F#:maj': 57.4, 'G#:maj': 43.9, 'C#:maj': 47.2, 'D#:maj': 33.5, 'D:min': 31.0, 'D#:min': 36.4, 'C:min': 16.4, 'F:min': 9.6, 'G:min': 17.0, 'A#:maj': 30.4, 'A#:min': 8.4},
              {'N': 88.8, 'A#:maj': 84.8, 'F:maj': 75.2, 'G:min': 58.7, 'D:maj': 77.7, 'D#:maj': 45.0, 'C#:maj': 58.3, 'C:maj': 63.1, 'E:maj': 67.1, 'F:min': 29.0, 'G#:maj': 60.9, 'A:min': 58.7, 'E:min': 51.1, 'D:min': 48.5, 'C:min': 29.3, 'F#:maj': 63.3, 'D#:min': 49.5, 'A#:min': 48.0, 'B:maj': 61.4, 'A:maj': 65.0, 'G#:min': 28.8, 'C#:min': 42.1, 'B:min': 75.9, 'F#:min': 44.2, 'G:maj': 68.1},
              {'N': 218.7, 'D#:maj': 88.1, 'B:maj': 98.2, 'C#:maj': 74.0, 'C:min': 55.0, 'C:maj': 121.2, 'F:min': 39.7, 'G:min': 66.3, 'A:min': 73.9, 'F#:min': 60.2, 'G:maj': 126.9, 'D:maj': 137.1, 'B:min': 70.2, 'A:maj': 137.1, 'E:maj': 82.8, 'E:min': 55.3, 'D:min': 57.9, 'G#:maj': 106.5, 'D#:min': 52.3, 'F#:maj': 84.3, 'A#:maj': 89.1, 'F:maj': 93.8, 'C#:min': 57.2, 'A#:min': 32.9, 'G#:min': 46.3},
              {'N': 255.0, 'D#:maj': 76.3, 'A#:maj': 94.0, 'G#:maj': 97.9, 'C:maj': 93.8, 'G:maj': 105.7, 'A:maj': 110.0, 'E:maj': 63.6, 'F#:min': 57.6, 'B:maj': 66.9, 'F:maj': 121.8, 'E:min': 54.1, 'D:min': 50.0, 'A:min': 51.5, 'F#:maj': 54.1, 'D:maj': 68.7, 'C#:min': 53.1, 'G#:min': 37.5, 'G:min': 22.7, 'C:min': 41.7, 'B:min': 60.7, 'C#:maj': 50.2, 'A#:min': 27.5, 'F:min': 17.4, 'D#:min': 13.1},
              {'N': 151.4, 'D#:min': 51.7, 'G#:min': 42.9, 'C#:maj': 65.0, 'F#:maj': 75.7, 'A#:min': 51.7, 'A:maj': 92.8, 'B:min': 61.8, 'G:maj': 103.3, 'F#:min': 36.8, 'B:maj': 73.3, 'G#:maj': 113.2, 'C#:min': 54.4, 'A:min': 63.7, 'E:min': 70.2, 'D:min': 29.5, 'D:maj': 73.5, 'G:min': 50.7, 'A#:maj': 96.2, 'F:maj': 75.0, 'C:maj': 59.8, 'F:min': 72.9, 'D#:maj': 93.4, 'E:maj': 97.5, 'C:min': 56.1}]

    def get_color(y, lm, mh):
        color = []
        for i in range(len(y)):
            if y[i] < lm:
                color.append("lightgreen")
            elif y[i] < mh:
                color.append("skyblue")
            else:
                color.append("orange")
        return color

    for k in range(6):
        plt.figure(figsize=(18, 5))
        chord = chords[k]
        del chord['N']
        dct = sorted(chord.items(), key=lambda x: x[0])
        x = []
        y = []
        for item in dct:
            x.append(item[0])
            y.append(item[1])
        if k != 3:
            plt.bar(x, y, color=get_color(y, 40, 80))
        else:
            plt.bar(x, y, color=get_color(y, 50, 100))
        plt.xlabel('和弦类别')
        plt.ylabel('累计时间/s')
        plt.title(classification[k] + ' 和弦分布')
        plt.show()


def extract_features():
    sptg = [numpy.array([[0.0 for i in range(345)] for j in range(1025)]) for k in range(6)]
    sptg[2] = numpy.array([[0.0 for i in range(431)] for j in range(1025)])
    mfcc = [numpy.array([[0.0 for i in range(345)] for j in range(20)]) for k in range(6)]
    mfcc[2] = numpy.array([[0.0 for i in range(431)] for j in range(20)])
    zcr = [0 for i in range(6)]
    tempo = [0 for i in range(6)]
    tune = [0 for i in range(6)]
    for i in range(6):
        begin, end = id_map[i]
        dct = {}
        for j in range(begin, end + 1):
            print(j)
            file = 'D:/抖音数据/二代数据/音频/全部/音频 (' + str(j) + ').wav'
            y, sr = librosa.load(file)
            if i == 2 and get_spectrogram(y).shape[1] != 431:
                continue
            sptg[i] += numpy.array(get_spectrogram(y))
            mfcc[i] += numpy.array(get_mfcc(y))
            zcr[i] += get_zero_crossing_rate(y)
            tempo[i] += get_tempo(y)
            tune[i] += get_tune(y)
            temp = get_chord(file)
            for chord in temp:
                if chord not in dct:
                    dct[chord] = temp[chord]
                else:
                    dct[chord] += temp[chord]
        sptg[i] /= end - begin + 1
        mfcc[i] /= end - begin + 1
        zcr[i] /= end - begin + 1
        tempo[i] /= end - begin + 1
        tune[i] /= end - begin + 1
        numpy.savetxt('sptg_pro/' + str(i) + '.csv', sptg[i], delimiter=',')
        numpy.savetxt('mfcc_pro/' + str(i) + '.csv', mfcc[i], delimiter=',')
        print("zcr: ", zcr[i])
        print("tempo: ", tempo[i])
        print("tune: ", tune[i])
        print("chord: ", dct)


if __name__ == '__main__':
    show_mfcc()