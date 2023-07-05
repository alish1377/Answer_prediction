# Generators
from dataset import Dataset
from torch.utils.data import DataLoader

def data_generation(n_users, n_questions, subjects, meta, dataset,  batch_size, shuffle):
    params = {'batch_size': batch_size,
              'shuffle': shuffle
              }
    #
    # n_users, n_questions = 542, 1774
    # subjects = 388
    # train_set = Dataset(train, question_meta, n_users, n_questions, subjects)
    # train_generator = torch.utils.data.DataLoader(train_set, **params)

    data_set = Dataset(dataset, meta, n_users, n_questions, subjects)
    generator = DataLoader(data_set, **params)

    return data_set, generator