import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator
# from data_generator_caanet import DataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       print_confusion_matrix, print_accuracy, print_accuracy_binary,
                       extract_spectrograms, plot_att)
from models_pytorch import move_data_to_gpu, DecisionLevelMaxPooling
from models_caanet import CnnPooling_Attention
import config


Model = CnnPooling_Attention
batch_size = 1
prototypes = ['193_1b2_Ar_mc_AKGC417L_0.wav', '186_2b3_Lr_mc_AKGC417L_4.wav', '193_1b2_Al_mc_AKGC417L_10.wav', '114_1b4_Pl_mc_AKGC417L_2.wav']
criticisms = ['141_1b1_Pr_mc_LittC2SE_0.wav', '199_2b1_Ll_mc_LittC2SE_1.wav', '112_1p1_Ll_sc_Litt3200_4.wav', '141_1b2_Ar_mc_LittC2SE_3.wav']

def evaluate(model, generator, data_type, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    
    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]

    loss = F.nll_loss(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy()
    loss = float(loss)
    
    accuracy = calculate_accuracy(targets, predictions, classes_num, average='macro')

    return accuracy, loss

def forward(model, generate_func, cuda):
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
    atts = {}
    atts_org = {}
    atts_label = {}
    
    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y, batch_audio_names) = data
        batch_x = move_data_to_gpu(batch_x, cuda)
        # Predict
        model.eval()
        logmel_x, att, batch_output = model(batch_x)
        if batch_audio_names[0] in prototypes or batch_audio_names in criticisms:
            atts_label[batch_audio_names[0]] = batch_output.data.cpu().numpy()
            atts[batch_audio_names[0]] = att.data.cpu().numpy()
            atts_org[batch_audio_names[0]] = logmel_x.data.cpu().numpy()
        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets

    # atts = np.concatenate(atts, axis=0)
    dict['att'] = atts

    # atts_org = np.concatenate(atts_org, axis=0)
    dict['att_org'] = atts_org

    dict['att_label'] = atts_label
        
    return dict

def train(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    validate = args.validate
    iteration_max = args.iteration_max
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    # Paths
    if validate:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_train.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_dev.csv')
    else:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_traindev.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_test.csv')
        
    models_dir = os.path.join(workspace, 'caanet_models', subdir)
    create_folder(models_dir)

    # Model
    model = Model(classes_num)

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(dataset_dir=dataset_dir,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)
    class_weight = generator.calculate_class_weight()
    class_weight = move_data_to_gpu(class_weight, cuda)

    # Optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    train_bgn_time = time.time()

    # Train on mini batches
    for (iteration, (batch_x, batch_y, _)) in enumerate(generator.generate_train()):
        # Evaluate
        if iteration % 100 == 0:
            train_fin_time = time.time()

            (tr_acc, tr_loss) = evaluate(model=model,
                                         generator=generator,
                                         data_type='train',
                                         max_iteration=None,
                                         cuda=cuda)

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f}'.format(tr_acc, tr_loss))

            (va_acc, va_loss) = evaluate(model=model,
                                         generator=generator,
                                         data_type='evaluate',
                                         max_iteration=None,
                                         cuda=cuda)
                                
            logging.info('va_acc: {:.3f}, va_loss: {:.3f}'.format(va_acc, va_loss))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        # Train
        model.train()
        batch_output = model(batch_x)

        loss = F.nll_loss(batch_output, batch_y, weight=class_weight)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Stop learning
        if iteration == iteration_max:
            break


def inference_validation_data(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    validate = args.validate
    iteration_max = args.iteration_max
    filename = args.filename
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    # Paths
    if validate:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_train.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_dev.csv')
    else:
        dev_train_csv = os.path.join(dataset_dir, 'meta_data', 'meta_traindev.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'meta_data', 'meta_test.csv')

    model_path = os.path.join(workspace, 'caanet_models', subdir, 'md_{}_iters.tar'.format(iteration_max))

    # Load model
    model = Model(classes_num)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param number:')
    print(param_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Predict & evaluate
    # Data generator
    generator = DataGenerator(dataset_dir = dataset_dir,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)

    generate_func = generator.generate_validate(data_type='train',
                                                shuffle=False)

    # Inference
    dict = forward(model=model,
                   generate_func=generate_func,
                   cuda=cuda)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    atts = dict['att']
    atts_org = dict['att_org']
    atts_label = dict['att_label']
    plot_att(atts_label, atts_org, atts)
    predictions = np.argmax(outputs, axis=-1)
    classes_num = outputs.shape[-1]

    # Evaluate
    # confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num)
            
    # class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)
    # se, sp, as_score, hs_score = calculate_accuracy(targets, predictions, classes_num, average='binary')

    # Print
    # print_accuracy(class_wise_accuracy, labels)
    # print_confusion_matrix(confusion_matrix, labels)
    #print('confusion_matrix: \n', confusion_matrix)
    # print_accuracy_binary(se, sp, as_score, hs_score, labels)



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