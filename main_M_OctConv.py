import os
import argparse
from files.solver_M_OctConv import Solver
from files.data_loader import get_loader
from torch.backends import cudnn
import pickle
from torch.cuda import max_memory_reserved
from torch.cuda import max_memory_allocated



def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        if config.CelebA_data_loader_load_dir == "":
            celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                       config.celeba_crop_size, config.image_size, config.batch_size,
                                       'CelebA', config.mode, config.num_workers)
        else:
            with open(config.CelebA_data_loader_load_dir, "rb") as f:
                celeba_loader = pickle.load(f)

    if config.dataset in ['RaFD', 'Both']:
        if config.RaFD_data_loader_load_dir == "":
            rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                     config.rafd_crop_size, config.image_size, config.batch_size,
                                     'RaFD', config.mode, config.num_workers)
        else:
            with open(config.RaFD_data_loader_load_dir, "rb") as f:
                rafd_loader = pickle.load(f)

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, config)

    if config.CelebA_data_loader_save_dir !="":
        with open(config.CelebA_data_loader_save_dir, "wb") as f:
            pickle.dump(celeba_loader, f)

    if config.RaFD_data_loader_save_dir != "":
        with open(config.RaFD_data_loader_save_dir, "wb") as f:
            pickle.dump(rafd_loader, f)
            
    print("mem-reserved:",max_memory_reserved())
    print("mem-allocated:",max_memory_allocated())
    if config.mode == 'train':
        print("mem-reserved:",max_memory_reserved())
        print("mem-allocated:",max_memory_allocated())
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train() 
        elif config.dataset in ['Both']:
            solver.train_multi()
        print("mem-reserved:",max_memory_reserved())
        print("mem-allocated:",max_memory_allocated())
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    #append

    parser.add_argument('--CelebA_data_loader_load_dir', type=str, default="", help='dir to pre-loaded CelebA data_loader')
    parser.add_argument('--CelebA_data_loader_save_dir', type=str, default="", help='dir save the new CelebA data_loader (not saved if not path is given)')
    parser.add_argument('--RaFD_data_loader_load_dir', type=str, default="", help='dir to pre-loaded RaFD data_loader')
    parser.add_argument('--RaFD_data_loader_save_dir', type=str, default="", help='dir save the new RaFD data_loader (not saved if not path is given)')

    parser.add_argument('--alpha1', type=float, default=0.5)
    parser.add_argument('--alpha2', type=float, default=0)


    config = parser.parse_args()
    main(config)
