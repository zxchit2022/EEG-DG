import torch
import torch.nn as nn
from sklearn.metrics.pairwise import euclidean_distances

class Dist_Loss(nn.Module):
    def __init__(self, classes=4):
        super(Dist_Loss, self).__init__()
        self.classes = classes


    def intraclass_compactness(self, data, labels):
        unique_labels = torch.unique(labels)
        compactness = 0

        for label in unique_labels:
            class_data = data[labels == label]

            class_data = class_data.cpu().detach()

            class_data_np = class_data.numpy()

            distances = euclidean_distances(class_data_np)
            compactness += torch.sum(torch.from_numpy(distances)) / 2

        return compactness / data.shape[0]


    def interclass_separability(self, data, labels):
        unique_labels = torch.unique(labels)
        separability = 0

        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                if i != j:
                    class_data_1 = data[labels == unique_labels[i]]
                    class_data_2 = data[labels == unique_labels[j]]

                    class_data_1 = class_data_1.cpu().detach()
                    class_data_2 = class_data_2.cpu().detach()

                    class_data_1_np = class_data_1.numpy()
                    class_data_2_np = class_data_2.numpy()

                    distances = euclidean_distances(class_data_1_np, class_data_2_np)
                    separability += torch.sum(torch.from_numpy(distances))

        return separability / data.shape[0]

    def compute_class_centers(self, data, labels):
        unique_labels = torch.unique(labels)
        class_centers = []

        for label in unique_labels:
            class_data = data[labels == label]
            class_center = torch.mean(class_data, dim=0)
            class_centers.append(class_center)

        return class_centers


    def forward(self, source_data, source_labels, target_data, target_labels, alpha):
        dist_1 = self.intraclass_compactness(source_data, source_labels) - alpha * self.interclass_separability(
            source_data, source_labels)
        dist_2 = self.intraclass_compactness(target_data, target_labels) - alpha * self.interclass_separability(
            target_data, target_labels)

        source_centers = self.compute_class_centers(source_data, source_labels)
        target_centers = self.compute_class_centers(target_data, target_labels)

        source_centers_np = torch.stack(source_centers).detach().cpu().numpy()
        target_centers_np = torch.stack(target_centers).detach().cpu().numpy()

        matrix = torch.cdist(torch.from_numpy(source_centers_np), torch.from_numpy(target_centers_np), p=2)

        diagonal = torch.diag(matrix)
        dist_31 = torch.mean(diagonal)

        dist = dist_1 + dist_2 + dist_31

        return dist

