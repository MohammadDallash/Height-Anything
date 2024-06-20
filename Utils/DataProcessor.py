import itertools
import torch

class DataProcessor:
    @staticmethod
    def get_data_element(p1, p2):
        """Calculate the depth difference and scale change between two points."""
        depth_diff = p2[0] - p1[0]

        if depth_diff == 0:
            return None

        return p1[0], depth_diff, p2[1] / p1[1]

    @staticmethod
    def generate_and_prepare_tensors(tracking_data, device):
        """Generate dataset using permutations of tracking data elements and prepare tensors on the specified device."""
        dataset = []
        for id_data in tracking_data.values():
            # Generate permutations of two elements
            permutations = list(itertools.permutations(id_data, 2))
            for perm in permutations:
                p1, p2 = perm
                scale_change = DataProcessor.get_data_element(p1, p2)
                if scale_change is not None:
                    dataset.append(scale_change)

        X = [[elem[0], elem[1]] for elem in dataset]
        y = [elem[2] for elem in dataset]

        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

        return X, y
