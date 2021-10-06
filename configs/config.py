import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_dataset_dir', type=str, default=r'/home/fss/dataset/cloud/',
                    help='train dataset path')
parser.add_argument('--log_dir', type=str, default='./logs', help='log path')
parser.add_argument('--saved_model_dir', type=str, default='./weights/', help='saved pb model path')

parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--val_ratio', type=float, default=0.2)

parser.add_argument('--batch_size', type=int, default=12, help='total batch size for all GPUs')
parser.add_argument('--multi_gpus', type=bool, default=False)
parser.add_argument('--init_learning_rate', type=float, default=3e-4)
parser.add_argument('--warmup_alpha', type=float, default=1e-2)
parser.add_argument('--warmup_epochs', type=int, default=2)
parser.add_argument('--img_size', type=int, default=192, help='image target size')

args = parser.parse_args()
params = vars(args)
