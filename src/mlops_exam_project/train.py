from mlops_exam_project.model import Model
from mlops_exam_project.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    print(len(dataset))
    model = Model()
    print(len(model))
    # add rest of your training code here

if __name__ == "__main__":
    train()
