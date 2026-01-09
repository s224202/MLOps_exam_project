from mlops_exam_project.model import Model
from mlops_exam_project.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here
    print(len(dataset))
    print(model.__getstate__())
if __name__ == "__main__":
    train()
