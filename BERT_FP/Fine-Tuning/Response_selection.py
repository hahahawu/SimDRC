import time
import argparse
import pickle
import os
from BERT_finetuning import NeuralNetwork
from setproctitle import setproctitle

setproctitle('BERT_FP')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Dataset path.
FT_data = {
    'ubuntu': '../ubuntu_data/ubuntu_dataset_1M.pkl',
    'douban': '../douban_data/douban_dataset_1M.pkl',
    'e_commerce': '../e_commerce_data/e_commerce_dataset_1M.pkl'
}
print(os.getcwd())
## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--batch_size",
                    default=2,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adamw.")
parser.add_argument("--epochs",
                    default=2,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--model_name",
                    required=True,
                    type=str,
                    help="The path to save model.")
# parser.add_argument("--score_file_path",
#                     default="score_file.txt",
#                     type=str,
#                     help="The path to save model.")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--locality", action="store_true", default=False)
parser.add_argument("--cross", action="store_true", default=False)
parser.add_argument("--margin", type=float, default=None)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--simctg", action="store_true", default=False)

args = parser.parse_args()

model_dir = os.path.join("checkpoints", args.model_name)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

args.save_path = os.path.join(model_dir, args.task + '.' + "0.pt")
args.score_file_path = os.path.join("score_files", args.model_name)
# load bert

print(args)
print("Task: ", args.task)


def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    if args.is_training:
        train_model(train, dev)
        test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")
