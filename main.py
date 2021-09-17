import time
import argparse

from train_faster_rcnn import faster_rcnn_training
# from test_faster_rcnn import faster_rcnn_testing

def main(args):
    # Time setting
    total_start_time = time.time()

    if args.model == 'faster_rcnn':
        if args.training:
            faster_rcnn_training(args)
        # else:
        #     faster_rcnn_testing(args)

    elif args.model == 'retinanet':
        pass

    elif args.model == 'iterdet':
        pass

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    
    # Task setting
    parser.add_argument('--model', default='faster_rcnn' ,type=str, choices=['faster_rcnn', 'retinanet'],
                        help="Choose model in 'Faster R-CNN'")
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')

    # Path setting
    parser.add_argument('--data_path', default='.', type=str,
                        help='CrowdHuman data path')
    parser.add_argument('--save_path', default='.', type=str,
                        help='Checkpoint save path')

    # Data setting
    parser.add_argument('--img_height', default=64, type=int,
                        help='Image resize size; Default is 256')
    parser.add_argument('--img_width', default=64, type=int,
                        help='Image resize size; Default is 256')

    # Optimizer & LR_Scheduler setting
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD'; Default is AdamW")
    parser.add_argument('--scheduler', default='constant', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is constant")
    parser.add_argument('--n_warmup_epochs', default=2, type=float, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")

    # Training setting
    parser.add_argument('--num_epochs', default=10, type=int, 
                        help='Training epochs; Default is 10')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=2, type=int, 
                        help='Batch size; Default is 16')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    # Testing setting
    parser.add_argument('--test_batch_size', default=32, type=int, 
                        help='Test batch size; Default is 32')
    # Print frequency
    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')
    args = parser.parse_args()

    main(args)