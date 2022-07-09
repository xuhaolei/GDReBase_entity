import argparse
import sys

parser = argparse.ArgumentParser()
dataset = "bacterium"
# 以后再统一吧
# parser.add_argument("-datatype", type=str, default="disease",help='disease or bacterium')
parser.add_argument("-train_path", type=str, default="train_%s.csv"%dataset,help='train_path')
parser.add_argument("-dev_path", type=str, default="dev_%s.csv"%dataset,help='dev_path')
parser.add_argument("-test_path", type=str, default="test_%s.csv"%dataset,help='test_path')
parser.add_argument("-vocab_path", type=str, default="vocab_%s.pkl"%dataset,help='vocab_path')
parser.add_argument("-testonly", action="store_const", const=True, default=False)

# model
parser.add_argument("-filter_sizes", type=str, default="2,3,4")
parser.add_argument("-num_filters", type=int, default=256)
parser.add_argument("-num_classes", type=int, default=1)
parser.add_argument("-embed_file", type=str, default='processed-%s.embed'%dataset)
parser.add_argument("-embed_size", type=int, default=100)
parser.add_argument("-test_model", type=str, default=None)

# training
parser.add_argument("-num_epochs", type=int, default=500)
parser.add_argument("-pad_size", type=int, default=512)
parser.add_argument("-dropout", type=float, default=0.5)
parser.add_argument("-patience", type=int, default=5)
parser.add_argument("-batch_size", type=int, default=128)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-gpu", type=int, default=-1, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("-save_path", type=str, default='TextCNN-%s.pth'%dataset, help='path for save model')

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command
