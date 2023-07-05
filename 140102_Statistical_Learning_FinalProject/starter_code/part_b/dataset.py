import numpy as np
import pandas as pd
import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataFrame, meta, n_users, n_questions, subjects):
        self.dataFrame = dataFrame
        self.meta = meta
        self.users = n_users
        self.questions = n_questions
        self.subjects = subjects
    def __len__(self):
        return len(self.dataFrame.index)
    def __getitem__(self, idx):

        y = self.dataFrame.iloc[idx]['is_correct']

        x_temp1 = np.zeros((1, self.users))
        x_temp1[:,self.dataFrame.iloc[idx]['user_id']] +=1
        x_temp2 = np.zeros((1, self.questions))
        x_temp2[:,self.dataFrame.iloc[idx]['question_id']] +=1

        subjects = eval(self.meta.loc[self.meta['question_id'] == self.dataFrame.iloc[idx]['question_id']]['subject_id'].item())
        x_temp3 = np.zeros((1, self.subjects))
        for i in subjects:
            x_temp3[:,i]+=1
        X = np.concatenate([x_temp1, x_temp2, x_temp3], axis=1)
        return X, y





