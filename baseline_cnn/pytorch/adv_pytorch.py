import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       print_confusion_matrix, print_accuracy, print_accuracy_binary,
                       plot_embedding_2D, plot_embedding_3D, plot_att)
from attackers import BIM
from models_org import move_data_to_gpu, DecisionLevelMaxPooling
from models_att import DecisionLevelMaxPooling_Att
from models_dia_att import move_data_to_gpu, DecisionLevelMaxPooling_Dia_Att
import config


# Model = DecisionLevelMaxPooling
# Model = DecisionLevelMaxPooling_Att
Model = DecisionLevelMaxPooling_Dia_Att
batch_size = 1
CLIP_MAX = 0.5
CLIP_MIN = -0.5
# For the log mel
# prototypes = ['213_2p2_Tc_mc_AKGC417L_6.wav', '158_2p2_Ar_mc_AKGC417L_6.wav', '193_1b2_Ar_mc_AKGC417L_10.wav', '158_2p2_Ar_mc_AKGC417L_4.wav']
# criticisms = ['157_1b1_Ar_sc_Meditron_6.wav', '138_1p3_Pl_mc_AKGC417L_3.wav', '158_1p3_Ll_mc_AKGC417L_1.wav', '107_3p2_Tc_mc_AKGC417L_4.wav']

# For TSNE:
prototypes = ['166_1p1_Al_sc_Meditron_5.wav', '213_2p2_Tc_mc_AKGC417L_6.wav', '158_2p2_Ar_mc_AKGC417L_6.wav', '193_1b2_Ar_mc_AKGC417L_10.wav', '158_2p2_Ar_mc_AKGC417L_4.wav', '130_1p4_Tc_mc_AKGC417L_0.wav', '158_2p3_Tc_mc_AKGC417L_0.wav', '158_2p3_Tc_mc_AKGC417L_8.wav']
criticisms = ['107_2b5_Pr_mc_AKGC417L_0.wav', '130_1p3_Pr_mc_AKGC417L_5.wav', '157_1b1_Ar_sc_Meditron_6.wav', '193_7b3_Pr_mc_AKGC417L_4.wav', '130_2p5_Ar_mc_AKGC417L_1.wav', '138_1p2_Pl_mc_AKGC417L_2.wav', '138_1p3_Pl_mc_AKGC417L_3.wav', '158_1p3_Al_mc_AKGC417L_3.wav', '200_2p2_Tc_mc_AKGC417L_1.wav', '218_1b1_Lr_sc_Meditron_10.wav', '158_1p3_Ll_mc_AKGC417L_1.wav', '107_3p2_Tc_mc_AKGC417L_0.wav', '107_3p2_Tc_mc_AKGC417L_4.wav', '158_2p3_Tc_mc_AKGC417L_5.wav']


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def forward(model, generate_func, cuda, attacker):
    """Forward data to a model.
    
    Args:
      model: object
      generate_func: generate function
      cuda: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    targets = []
    tsnes = []
    adv_failures = {0:[], 1:[], 2:[], 3:[]}
    # For the attention matrix visualisation
    atts = {}
    atts_org = {}
    atts_label = {}

    # attacker = BIM(eps=0.25, eps_iter=0.001, n_iter=4, clip_max=CLIP_MAX, clip_min=CLIP_MIN)
    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y, batch_audio_names) = data
        targets.append(batch_y)
        audio_names.append(batch_audio_names)
        # Prepare for the attack
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        batch_x_adv = torch.zeros(batch_x.shape)
        batch_x_adv = move_data_to_gpu(batch_x_adv, cuda)
        # Attack
        # for i in range(len(batch_x)):
        #    batch_x_adv[i] = attacker.generate(model, batch_x[i], batch_y[i])
        
        model.eval()
        # batch_size = 1 just for plotting the log Mel spectrograms
        
        ifplot = batch_audio_names[0] in prototypes or batch_audio_names[0] in criticisms
        ifplot = False
        # batch_tsne, batch_output = model(batch_x, False, batch_audio_names[0])# (audios_num, classes_num)
        logmel_x, att, batch_output = model(batch_x, ifplot, batch_audio_names[0])# (audios_num, classes_num)
        outputs.append(batch_output.data.cpu().numpy())
        # print(activation['cnn_encoder'].shape)
        tsnes.append(activation['cnn_encoder'].data.cpu().numpy())
        # For the attention matrix visualisation
        if ifplot:
            atts_label[batch_audio_names[0]] = batch_output.data.cpu().numpy()
            atts[batch_audio_names[0]] = att.data.cpu().numpy()
            atts_org[batch_audio_names[0]] = logmel_x.data.cpu().numpy()
        
        # tsnes.append(batch_tsne.data.cpu().numpy())
        # _, _, batch_output_adv = model(batch_x_adv, False, batch_audio_names[0])# (audios_num, classes_num)
        
        batch_output = batch_output.data.cpu().numpy()
        # batch_output_adv = batch_output_adv.data.cpu().numpy()
        batch_y = batch_y.data.cpu().numpy()
        
        batch_output = np.argmax(batch_output, axis=-1)
        # batch_output_adv = np.argmax(batch_output_adv, axis=-1)
        # Search for Prototypes
        # for i in range(len(batch_output)):
        #    if batch_output[i] == batch_y[i] and batch_output[i] != batch_output_adv[i]:
        #        adv_failures[batch_y[i]].append(batch_audio_names[i])
        
    dict = {}

    tsnes = np.concatenate(tsnes, axis=0)
    dict['tsne'] = tsnes

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets

    # adv_failures = np.concatenate(adv_failures, axis=0)
    dict['adv_failures'] = adv_failures
    
    # atts = np.concatenate(atts, axis=0)
    dict['att'] = atts

    # atts_org = np.concatenate(atts_org, axis=0)
    dict['att_org'] = atts_org

    dict['att_label'] = atts_label
    
    return dict


def inference_validation_data(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    validate = args.validate
    iteration_max = args.iteration_max
    filename = args.filename
    cuda = args.cuda
    isres = args.isres
    
    # for the BIM
    eps = args.eps
    steps = args.steps
    attacker = BIM(eps=0.01, eps_iter=eps, n_iter=steps, clip_max=CLIP_MAX, clip_min=CLIP_MIN)
    print('eps:{}, steps:{}'.format(eps, steps))
    
    labels = config.labels
    classes_num = len(labels)

    # Paths
    if validate:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_train.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_dev.csv')
    else:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_traindev.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_test.csv')

    if isres:
        model_path = os.path.join(workspace, 'dia_att_res_models', subdir, 'md_{}_iters.tar'.format(iteration_max))
    else:
        model_path = os.path.join(workspace, 'dia_att_models', subdir, 'md_{}_iters.tar'.format(iteration_max))

    # Load model
    model = Model(classes_num, isres)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('param number:')
    # print(param_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()
    
    model.cnn_encoder.register_forward_hook(get_activation('cnn_encoder'))
    # Predict & evaluate
    # Data generator
    generator = DataGenerator(dataset_dir = dataset_dir,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)
    # Use the training data for adv
    generate_func = generator.generate_validate(data_type='train',
                                                shuffle=False)

    # Inference
    start_time = time.time()
    dict = forward(model=model,
                   generate_func=generate_func,
                   cuda=cuda,
                   attacker=attacker)
    print("--- %s seconds in forwarding ---" % (time.time() - start_time))
    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    # adv_failures = dict['adv_failures']
    
    # Display the adv_failures
    #for key, value in adv_failures.items():
    #    print("There are {} number of samples in class {}".format(len(value), key))
    #    print(adv_failures[key])
    # 
    # For the attention matrix visualisation
    atts = dict['att']
    atts_org = dict['att_org']
    atts_label = dict['att_label']
    # plot_att(atts_label, atts_org, atts)

    predictions = np.argmax(outputs, axis=-1)
    classes_num = outputs.shape[-1]

    
    # tsne
    tsnes = dict['tsne']
    audio_names = dict['audio_name']
    print(len(tsnes), len(targets))
    # Faltten the inputs for tsne
    tsnes = tsnes.reshape(tsnes.shape[0], -1)
    tsne_2D = TSNE(n_components=2, init='pca', learning_rate='auto')
    result_2D = tsne_2D.fit_transform(tsnes)
    # tsne_3D = TSNE(n_components=3, init='pca', learning_rate='auto', perplexity=50, n_iter=5000)
    # result_3D = tsne_3D.fit_transform(tsnes)
    pros = [audio_names.tolist().index(i) for i in prototypes]
    cris = [audio_names.tolist().index(i) for i in criticisms]
    fig1 = plot_embedding_2D(result_2D, targets, pros, cris, 't-SNE-2D')
    # fig2 = plot_embedding_3D(result_3D, targets, pros, cris, 't-SNE-3D')
    

    # Evaluate
    confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num)
            
    class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)
    se, sp, as_score, hs_score = calculate_accuracy(targets, predictions, classes_num, average='binary')

    # Print
    print_accuracy(class_wise_accuracy, labels)
    print_confusion_matrix(confusion_matrix, labels)
    #print('confusion_matrix: \n', confusion_matrix)
    print_accuracy_binary(se, sp, as_score, hs_score, labels)



if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Example of parser. ')
    
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_dir', type=str, default='../../../data_experiment/')
    parser.add_argument('--subdir', type=str, default='models_dev')
    parser.add_argument('--workspace', type=str, default='../../../experiment_workspace/baseline_cnn/')
    parser.add_argument('--validate', action='store_true', default=True)
    parser.add_argument('--iteration_max', type=int, default=15000)
    parser.add_argument('--cuda', action='store_true', default=False)
    '''
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--iteration_max', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--validate', action='store_true', default=False)
    parser_inference_validation_data.add_argument('--iteration_max', type=int, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation_data.add_argument('--isres', action='store_true', default=False)

    # For the BIM
    parser_inference_validation_data.add_argument('--steps', type=int, required=True)
    parser_inference_validation_data.add_argument('--eps', type=float, required=True)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    else:
        raise Exception('Error argument!')
