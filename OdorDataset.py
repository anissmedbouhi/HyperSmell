#create a custom pytorch dataset for the odor dataset
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import euclidean_distances


from utils.helpers import *
class OdorMonoDataset(Dataset):
    def __init__(self, embeddings,labels=None, transform=None):

        self.transform = transform
        self.embeddings = embeddings
        if labels is not None:
            self.labels = labels
        else:
            self.labels = embeddings



    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        sample = self.embeddings[idx]

        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = self.embeddings[idx]

        return idx, sample,label





class OdorMixDataset(Dataset):
    def __init__(self,base_dir, perception_csv, embeddings_csv, transform=None, grand_avg=False, descriptors=None):
        df_mols, df_mols_embeddings, df_mols_embeddings_zscored = prepare_similarity_mols_mix_on_representations(base_dir, perception_csv,
                                                       embeddings_csv,
                                                       mixing_type='average', sep=';',
                                                       start='0', end='255')

        self.base_dir = base_dir
        if grand_avg:
            self.ds = self.grand_average(self.ds, descriptors)
        else:
            pass
        self.labels = self.ds['y']
        self.data = df_mols['embeddings']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label





