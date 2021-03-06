import numpy as np
import soundfile
import librosa
import os
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):

    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1

    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def calculate_scalar(x):

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """

    samples_num = len(target)
    
    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):
        
        total[target[n]] += 1
        
        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy
        
    elif average == 'macro':
        return np.mean(accuracy)

    elif average == 'binary': # labels = ['normal', 'crackle', 'wheeze', 'both']
        se = np.sum(correctness[1:]) / np.sum(total[1:])
        sp = correctness[0]/total[0]
        as_score = (se + sp) / 2
        hs_score = (2 * se * sp) / (se + sp)
        return se, sp, as_score, hs_score

    else:
        raise Exception('Incorrect average!')


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes

    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num, classes_num), dtype=int)
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def print_accuracy(class_wise_accuracy, labels):

#    print('{:<30}{}'.format('Scene label', 'accuracy'))
#    print('------------------------------------------------')
#    for (n, label) in enumerate(labels):
#        print('{:<30}{:.3f}'.format(label, class_wise_accuracy[n]))
#    print('------------------------------------------------')
#    print('{:<30}{:.3f}'.format('Average', np.mean(class_wise_accuracy)))
    logging.info('{:<30}{}'.format('Label', 'Accuracy'))
    logging.info('------------------------------------------------')
    for (n, label) in enumerate(labels):
        logging.info('{:<30}{:.4f}'.format(label, class_wise_accuracy[n]))
    logging.info('------------------------------------------------')
    logging.info('{:<30}{:.4f}'.format('Average', np.mean(class_wise_accuracy)))

def print_accuracy_binary(se, sp, as_score, hs_score, labels):
    logging.info('{:<30}{}'.format('Label_binary', 'Accuracy'))
    logging.info('------------------------------------------------')
    logging.info('{:<4}{:<30}{:.4f}'.format('Se: ', ','.join(labels[1:]), se))
    logging.info('{:<4}{:<30}{:.4f}'.format('Sp: ', labels[0], sp))
    logging.info('------------------------------------------------')
    logging.info('{:<34}{:.4f}'.format('AS: ', as_score))
    logging.info('{:<34}{:.4f}'.format('HS: ', hs_score))

def print_confusion_matrix(confusion_matrix, labels):
    logging.info('Confusion matrix:')
    logging.info('{}'.format('\t'.join(labels)))
    for i in range(0, len(labels)):
        logging.info('{}'.format('\t'.join(map(str, confusion_matrix[i]))))


def plot_confusion_matrix(confusion_matrix, title, labels, values, path):
    """Plot confusion matrix.

    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal

    Ouputs:
      None
    """

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')
#    plt.show()


     
def write_evaluation_submission(submission_path, audio_names, predictions):
    
    ix_to_lb = config.ix_to_lb
    
    f = open(submission_path, 'w')	
    
    for n in range(len(audio_names)):
        f.write('audio/{}'.format(audio_names[n]))
        f.write('\t')
        f.write(ix_to_lb[predictions[n]])
        f.write('\n')
        
    f.close()
    
    logging.info('Write result to {}'.format(submission_path))

def projection(audio_names, y, x_logmel, outputs, distance, pre_audio_names, pre_x_logmel, pre_distance):
    '''
    Project the prototypes to their closest log mel spectrograms on each class.
    Args:
        audio_names: audio names
        y: labels
        x_logmel: logmel spectrograms, delta, and delta-delta
        outputs: predictions
        distance: distance between feature maps and prototypes
        pre_audio_names: previously calculated best audio names
        pre_x_logmel: previously calculated best logmel

    Returns:
        best_audio_names
        best_x_logmel
        best_distance
    '''
    pred = np.argmax(outputs, axis=-1)

    best_audio_names = [[] for i in range(0, outputs.shape[1])]
    best_x_logmel = [[] for i in range(0, outputs.shape[1])]
    best_distance = [[] for i in range(0, outputs.shape[1])]
    for i in range(0, len(y)):
        if pred[i] == y[i]:
            label = y[i]
            if pre_audio_names[label] == []:
                best_audio_names[label] = audio_names[i]
                best_x_logmel[label] = x_logmel[i][0]
                best_distance[label] = distance[i][label]
            else:
                if distance[i][label] < pre_distance[label]:
                    best_audio_names[label] = audio_names[i]
                    best_x_logmel[label] = x_logmel[i][0]
                    best_distance[label] = distance[i][label]
                else:
                    best_audio_names[label] = pre_audio_names[label]
                    best_x_logmel[label] = pre_x_logmel[label]
                    best_distance[label] = pre_distance[label]
    return best_audio_names, best_x_logmel, best_distance

def plot_prototypes(audio_names, x_logmel, distance, path_folder):
    for i in range(0, len(audio_names)):
        if len(x_logmel[i]) != 0:
            plt.matshow(x_logmel[i].T, origin='lower', aspect='auto', cmap='jet')
            plt.savefig(os.path.join(path_folder, audio_names[i] + '_y=' + str(i) + '_dist_' + str(distance[i]) +'.pdf'))


import torchlibrosa as tl
import torch
def extract_spectrograms(audio_input):
    sample_rate=config.sample_rate
    window_size = config.win_length
    hop_size = config.hop_length
    mel_bins = config.mel_bins
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    spectrogram_extractor = tl.Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)
    logmel_extractor = tl.LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=20, fmax=2000, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)
    x = spectrogram_extractor(audio_input)   # (batch_size, 1, time_steps, freq_bins)
    x = logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
    batch_size, channel_num, _, mel_bins = x.shape
    x_diff1 = torch.diff(x, n=1, dim=2, append=x[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
    x_diff2 = torch.diff(x_diff1, n=1, dim=2, append=x_diff1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
    # for the caanet, only the log mel spectrograms
    # x = torch.cat((x, x_diff1, x_diff2), dim=1)
    return x


def plot_att(atts_label, atts_org, atts):
    for key, value in atts.items():
        att = value[0] # (num_class, 126 * 128)
        label = atts_label[key][0] # (num_class, )
        # print(label.shape, label, np.argmax(label))
        att = att[np.argmax(label)]  # (126*128, )
        att = att.reshape((126, 128))
        # att = np.squeeze(att)
        im = plt.matshow(att.T, origin='lower', aspect='auto', cmap='jet')
        plt.colorbar(im)
        plt.savefig('{}_att.jpg'.format(key))
        plt.close()
 
        median_val = np.median(att)
        att_org = atts_org[key][0][0] # (1, 126, 128)
        att_org = np.squeeze(att_org) # (126, 128)
        for iy, ix in np.ndindex(att.shape):
            if att[iy, ix] < median_val:
                att_org[iy, ix] = None
        
        im = plt.matshow(att_org.T, origin='lower', aspect='auto', cmap='jet', vmax=-1)
        # plt.colorbar(im)
        plt.savefig('{}_att_org.jpg'.format(key))
        plt.close()


# for tsne plotting
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_embedding_2D(data, label, pros, cris, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        if i in pros:
            plt.text(data[i, 0], data[i, 1], str('*'),
                     color=plt.cm.Set1(label[i]),
                     fontdict={'weight': 'bold', 'size': 15})
        elif i in cris:
            plt.text(data[i, 0], data[i, 1], str('+'),
                     color=plt.cm.Set1(label[i]),
                     fontdict={'weight': 'bold', 'size': 15})
        else:
            plt.text(data[i, 0], data[i, 1], str('.'),
                     color=plt.cm.Set1(label[i]),
                     fontdict={'weight': 'light', 'size': 5})

    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    patch_0 = mpatches.Patch(color=plt.cm.Set1(0), label='Normal')
    patch_1 = mpatches.Patch(color=plt.cm.Set1(1), label='Crackle')
    patch_2 = mpatches.Patch(color=plt.cm.Set1(2), label='Wheeze')
    patch_3 = mpatches.Patch(color=plt.cm.Set1(3), label='Both')
    plt.legend(handles=[patch_0, patch_1, patch_2, patch_3])
    plt.savefig('{}_new.pdf'.format(title))
    plt.close()

def plot_embedding_3D(data,label, pros, cris, title):
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0)
    data = (data- x_min) / (x_max - x_min)
    # ax = plt.figure().add_subplot(111,projection='3d')
    fig = plt.figure()
    ax =fig.add_subplot(111,projection='3d')
    for i in range(data.shape[0]):
        if i in pros:
            ax.text(data[i, 0], data[i, 1], data[i,2],str('*'), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 12})
        elif i in cris:
            ax.text(data[i, 0], data[i, 1], data[i,2],str('+'), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 12})
        else:
            ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'light', 'size': 5})
    plt.savefig('{}.pdf'.format(title))
    plt.close()

