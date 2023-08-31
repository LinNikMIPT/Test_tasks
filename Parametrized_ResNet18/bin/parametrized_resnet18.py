import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from datetime import datetime
from time import time
from pathlib import Path
import os
import random
import json

from utils.logging_config import get_logger


def init_logging(log_rotation: int = 7) -> get_logger:
    file = Path(__file__).absolute()

    logdir = file.parent.parent / 'logs'
    logdir.mkdir(exist_ok=True)

    startup = datetime.fromtimestamp(time()).strftime('%d:%m:%y')
    log_file_name = str(logdir / f'{startup}_{file.stem}.log')

    logger = get_logger(name=os.path.basename(__file__),
                        log_file_name=log_file_name,
                        datefmt='%d-%m-%y %H:%M:%S',
                        midnight=True,
                        log_rotation=log_rotation)
    logger.info(f'Initialization logger for {os.path.basename(__file__)}')
    return logger


logger = init_logging()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH2DATA = './data'
NUM_EPOCHS = 5
NUM_EXPERIMENTS = 10
logger.info(f"Device: {DEVICE}")


class ParametrizedConv(nn.Module):
    def __init__(self, conv, mlp):
        super(ParametrizedConv, self).__init__()
        self.conv = conv
        self.mlp = mlp

    def forward(self, x):
        n_1, n_2 = self.conv.weight.shape[0], self.conv.weight.shape[1]
        x_vals = torch.linspace(-1, 1, n_1).to(x.device)
        y_vals = torch.linspace(-1, 1, n_2).to(x.device)
        x_vals, y_vals = torch.meshgrid(x_vals, y_vals)
        mlp_output = self.mlp(x_vals.flatten(), y_vals.flatten())
        mlp_output = mlp_output.view(n_1, n_2, *self.conv.weight.shape[2:])
        self.conv.weight.data = mlp_output
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, y):
        out = torch.cat([x.view(-1, 1), y.view(-1, 1)], dim=1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        return out


def parametrized_model(model, hidden_size=128):
    parametrized = copy.deepcopy(model)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            mlp = MLP(2, hidden_size, module.weight.shape[2] * module.weight.shape[3])
            param_conv = ParametrizedConv(module, mlp)
            setattr(parametrized, name, param_conv)
    return parametrized


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, logger,
                epochs=NUM_EPOCHS, model_name=None):
    try:
        train_loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []

        if model_name is None:
            logger.info(f"Start training {model.__class__.__name__}")
        else:
            logger.info(f"Start training {model_name}")
        model.to(DEVICE)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    logger.info(f"[{epoch + 1}, {i + 1:5d}] train loss: {running_loss / 100:.3f}")
                    train_loss_history.append(running_loss)
                    running_loss = 0.0

                    test_accuracy = evaluate_accuracy(model, test_dataloader)
                    test_accuracy_history.append(test_accuracy)
                    train_accuracy = evaluate_accuracy(model, train_dataloader)
                    train_accuracy_history.append(train_accuracy)
                    logger.info(f"test accuracy: {round(test_accuracy, 2)}%")
                    logger.info(f"train accuracy: {round(train_accuracy, 2)}%")

        if model_name is None:
            logger.info(f"Finished training {model.__class__.__name__} \n")
        else:
            logger.info(f"Finished training {model_name} \n")

        return train_loss_history, train_accuracy_history, test_accuracy_history
    except Exception:
        logger.exception(f"Critical error in {train_model.__name__}!")


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def compare_model_with_parametrized_model(num_experiments, criterion, train_dataloader, test_dataloader,
                                          start_num_epochs, stop_num_epochs, logger):
    history = []
    history_parametrized = []

    for i in range(num_experiments):
        logger.info(50*"-" + f"experiment num {i}" + 50*"-")

        epoch = random.randint(start_num_epochs, stop_num_epochs)
        logger.info(f"Num epochs: {epoch}")

        resnet = models.resnet18(pretrained=False)
        parametrized_resnet = parametrized_model(resnet)

        resnet_optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
        parametrized_resnet_optimizer = optim.SGD(parametrized_resnet.parameters(), lr=0.001, momentum=0.9)

        train_loss_param, train_accuracy_history_param, test_accuracy_history_param = train_model(model=parametrized_resnet,
                                                                                                  model_name="Parametrized_ResNet",
                                                                                                  train_dataloader=train_dataloader,
                                                                                                  test_dataloader=test_dataloader,
                                                                                                  criterion=criterion,
                                                                                                  optimizer=parametrized_resnet_optimizer,
                                                                                                  epochs=epoch,
                                                                                                  logger=logger)

        train_loss, train_accuracy, test_accuracy = train_model(model=resnet,
                                                                model_name="ResNet",
                                                                train_dataloader=train_dataloader,
                                                                test_dataloader=test_dataloader,
                                                                criterion=criterion,
                                                                optimizer=resnet_optimizer,
                                                                epochs=epoch,
                                                                logger=logger)

        history.append((train_loss, train_accuracy, test_accuracy))
        history_parametrized.append((train_loss_param, train_accuracy_history_param, test_accuracy_history_param))

        logger.info("\n \n")

    return history, history_parametrized


def init_dataloaders(batch_size: int = 64, num_workers: int = 2, train_shuffle: bool = True,
                     test_shuffle: bool = False):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root=PATH2DATA, train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=train_shuffle, num_workers=num_workers)
    test_dataset = torchvision.datasets.CIFAR10(root=PATH2DATA, train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=test_shuffle, num_workers=num_workers)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    import argparse
    from utils.config_reader_json import Config

    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description=os.path.basename(__file__),
                                     formatter_class=fmt)
    parser.add_argument('config_path', type=str, help='Path to the configuration file',
                        default="../configurations/input/configurations-1.json")
    parser.add_argument('constrain_path', type=str, help='Path to the default configuration file',
                        default="../constraints/constraint.json")
    args = parser.parse_args()

    conf = Config(configurations_file=args.config_path, constraints_file=args.constrain_path)
    parameters = conf.cfg

    num_experiments = parameters["num_experiments"]
    start_num_epochs = parameters["start_num_epochs"]
    stop_num_epochs = parameters["stop_num_epochs"]
    path2history_json = parameters["path2history_json"]

    train_dataloader, test_dataloader = init_dataloaders()

    criterion = nn.CrossEntropyLoss()
    history, history_parametrized = compare_model_with_parametrized_model(num_experiments=num_experiments,
                                                                          criterion=criterion,
                                                                          train_dataloader=train_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          start_num_epochs=start_num_epochs,
                                                                          stop_num_epochs=stop_num_epochs,
                                                                          logger=logger)

    history_json = {"history": history,
                    "history_parametrized": history_parametrized}

    with open(path2history_json, 'w') as f:
        json.dump(history_json, f, indent=4)
