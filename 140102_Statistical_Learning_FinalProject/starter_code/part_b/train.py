#Train Process
from data_generation import data_generation
from model import Net
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn





def train_procedure(train_set, val_set, train_loader, val_loader,
                    epoch=10,
                    lr=0.0001,
                    model = None):

    ce_loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # learning_rate scheduler that multiply lr with 0.5 in each 10 steps
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # define lists to save accuracies of each epoch
    train_acc_list, val_acc_list = [], []
    # define these 2 variable to save the best model based on the best validation accuracy
    best_val_acc, best_model = 0,0
    for step in range(epoch):

        train_correct = 0
        for data , labels in train_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            data = data.to(torch.float32)
            labels = labels.to(torch.float32)
            optimizer.zero_grad()
            target = model(data)
            target = torch.flatten(target)
            # convert target to one if bigger than 0.5
            target_label = torch.where(target > 0.5, 1, 0)
            # apply binary cross entropy
            loss = ce_loss(target, labels)
            loss.backward()
            optimizer.step()
            # add correct data to train_correct variables
            train_correct += (target_label == labels).sum().item()

        scheduler.step()
        val_correct =0
        model.eval()
        with torch.no_grad():
            for data, labels in val_loader:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                data = data.to(torch.float32)
                labels = labels.to(torch.float32)
                target = model(data)
                target = torch.flatten(target)
                target_label = torch.where(target > 0.5, 1, 0)
                val_correct += (target_label == labels).sum().item()

        train_acc = train_correct / len(train_set)
        train_acc_list.append(train_acc)
        val_acc = val_correct / len(val_set)
        val_acc_list.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
        print(f"Epoch {step+1} :")
        print(f"train accuracy ===> {train_acc:.4f}")
        print(f"val accuracy ===> {val_acc:.4f}")

    print("The best accuracy of validation set is: ",best_val_acc)
    return best_model, train_acc_list, val_acc_list

if __name__ == "__main__":
    # read train, validation and question_meta data
    train = pd.read_csv('../data/train_data.csv')
    val = pd.read_csv('../data/valid_data.csv')
    question_meta = pd.read_csv('../data/question_meta.csv')
    # define number of users, questions and, subjects
    n_users, n_questions, subjects = 542, 1774, 388
    batch_size, shuffle = 64, True
    features = n_users + n_questions + subjects
    # call Net model from model.py
    model = Net(features)
    print("Model is created")
    # call data_generation for both train and validation dataset
    train_set, train_loader = data_generation(
        n_users, n_questions, subjects,
        question_meta, train,
        batch_size, shuffle)
    val_set, val_loader = data_generation(
        n_users, n_questions, subjects,
        question_meta, val,
        batch_size, shuffle)

    epoch, lr = 30, 0.01
    train_params = {
    'train_set' : train_set,
    'val_set' : val_set,
    'train_loader' : train_loader,
    'val_loader' : val_loader,
    'epoch' : epoch,
    'lr' : lr,
    'model' : model
    }
    print("Start train procedure...")
    model, train_acc_list, val_acc_list = train_procedure(**train_params)
    print("End train procedure...")

    # plot train and validation accuracies based on the iteration
    epoch_list = [i for i in range(epoch)]
    plt.plot(epoch_list, train_acc_list)
    plt.plot(epoch_list, val_acc_list)
    plt.legend(['train_acc', 'val_acc'])
    plt.title('train and validation accuracies for part2')
    plt.savefig('../plots/Part_B/part_B.png')

    # save the model
    torch.save(model.state_dict(), '../part_b/model_path/model')



