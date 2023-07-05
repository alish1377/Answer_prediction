from dataset import *
from data_generation import *
from model import Net
import torch
import os


def test_procedure(test_set, test_loader, model):
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            data = data.to(torch.float32)
            labels = labels.to(torch.float32)
            target = model(data)
            target = torch.flatten(target)
            target_label = torch.where(target > 0.5, 1, 0)
            test_correct += (target_label == labels).sum().item()


    test_acc = test_correct / len(test_set)
    print(f"test accuracy ===> {test_acc:.4f}")

if __name__ == "__main__":
    question_meta = pd.read_csv('../data/question_meta.csv')
    test = pd.read_csv('../data/test_data.csv')
    n_users, n_questions, subjects = 542, 1774, 388
    features = n_users + n_questions + subjects
    batch_size, shuffle = 32, True
    model = Net(features)
    model_path = '../part_b/model_path/model'
    if not os.path.exists(model_path):
        raise Exception("The specified path {} does not exist.".format(model_path))
    model.load_state_dict(torch.load(model_path))
    test_set, test_loader = data_generation(
        n_users, n_questions, subjects,
        question_meta, test,
        batch_size, shuffle)
    print("Test accuracy..")
    test_procedure(test_set, test_loader, model)

