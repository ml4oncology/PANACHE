import argparse
import torch.nn as nn
import os
import torch
import torch.utils.data as data
from tqdm import tqdm
import random
import numpy as np
import time
import psutil
import collections
from sksurv.metrics import concordance_index_censored as concordance_index
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cox_ph_loss(log_h, durations, events, eps=1e-7):
    # sort descending by days until event
    durations, idx = durations.sort(descending=True)
    events = events[idx]
    log_h = log_h[idx]

    # indices of uncensored observations
    uncensored_idx = (events == 1).nonzero()

    # compute loss
    loss = torch.tensor([0]).to(device)
    for i in uncensored_idx:
        h_i = log_h[i]
        j = i
        # exclude samples with same duration from risk set
        while j > 0 and durations[j - 1] == durations[i]:
            j = j - 1
        # skip case where there is no risk set (all other patients experience event before patient)
        # factor out max risk from riskset before exponentiating to avoid numeric instability
        if j > 0:
            risk_set = log_h[:j]
            gamma = risk_set.max()
            log_cusum = torch.log(torch.sum(torch.exp(risk_set.sub(gamma)))).add(gamma)
            # case where all events occur at same time, loss will be 0, add eps
            loss = loss.add(torch.sub(h_i, log_cusum)).add(eps)
    return -torch.div(loss, torch.sum(events))


class MicroscopeData(data.Dataset):
    """
    A Pytorch dataset that returns a tensor of concatenated HIPT 4k features and associated targets

    Inputs:

        libraryfile: The dictionary with the following keys: --> comes from slide_features.py
            slide_path: a list to openslide paths, len is number of WSI
            time: a list of time to death for each WSI, len is number of WSI
            vital: a list of 0, 1 whether event has occured for each patient, len is number of WSI
            cohort: a list of cohorts each WSI belongs to, len is number of WSI
            magnification: a list of each WSI's original magnification, len is number of WSI
            grid: a list containing sub-list of tile coords (x,y) each sublist belongs to a WSI, len is number of WSI
            grade: a list the tumour grade of each WSI, len is number of WSI
        ftr_path: path to folder containing slide_id subfolders, each subfolder contains saved tile features

    """

    def __init__(self, libraryfile="", ftr_path=""):
        lib = torch.load(libraryfile)

        print("Slides: {}".format(len(lib["slide_id"])))

        self.slide_id = lib["slide_id"]
        self.coords = lib["grid"]
        self.os_time = lib["time"]
        self.event = lib["vital"]
        self.cohort = lib["cohort"]
        self.magnification = lib["magnification"]
        self.slide_path = lib["slide_path"]
        self.grade = lib["grade"]
        self.ftr_path = ftr_path

        # concatenate all 4k ftrs for each WSI
        self.ftrs_4k_cat = []
        for i, slide_id in enumerate(self.slide_id):
            ftrs_4k = []
            for coord in self.coords[i]:
                ftr_name = slide_id + "_" + str(coord[0]) + "_" + str(coord[1]) + ".pt"
                ftrs_4k.append(
                    torch.load(os.path.join(self.ftr_path, slide_id, ftr_name))
                )
            self.ftrs_4k_cat.append(torch.cat(ftrs_4k))

    def __getitem__(self, index):
        slide_id = self.slide_id[index]
        ftrs_4k_cat = self.ftrs_4k_cat[index]
        # mean pooling of ftrs, [1 x #4096tiles x 192] -> [1 x 192]
        ftrs_mean = torch.mean(ftrs_4k_cat, 0)

        return ftrs_mean, self.os_time[index], self.event[index]

    def __len__(self):
        return len(self.slide_id)


def get_dataloader(dataset, batch_size, pin_memory=False, num_workers=0):
    g = torch.Generator()
    g.manual_seed(416)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        generator=g,
    )

    print(f"\nNumber of samples: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    return dataloader


def get_val_loss_concordance(model, dset, criterion, batch_size):

    loader = get_dataloader(dset, batch_size=batch_size, num_workers=args.workers)
    # VALIDATION: COMPUTE LOSS
    print("\nRetreving validation loss and concordance index")

    mem = psutil.virtual_memory()
    print(
        "Memory used: {}, Free: {}, Available: {}".format(
            round(mem.used / 1024**3, 2),
            round(mem.free / 1024**3, 2),
            round(mem.available / 1024**3, 2),
        ),
        flush=True,
    )

    preds = []

    model.eval()
    with torch.no_grad():
        batches = tqdm(loader, total=len(loader), desc="Tuning loss")
        for i, batch in enumerate(batches):
            ftrs, _, _ = batch
            pred = model(ftrs)
            preds.append(pred.detach()[:, 0].clone())

    # split predictions and targets by cohort
    preds = torch.cat(preds, 0)
    os_times = collections.defaultdict(list)
    events = collections.defaultdict(list)
    risks = collections.defaultdict(list)
    for i, cohort in enumerate(dset.cohort):
        risks[cohort].append(preds[i])
        os_times[cohort].append(torch.tensor(dset.os_time[i]))
        events[cohort].append(torch.tensor(dset.event[i]))

    # stratified concordance and loss
    # stratified concordance is calculated by total num of concordant pairs / total num of comparable pairs -> pairs are not comparable btwn cohorts
    strat_loss = 0.0
    concordances = collections.defaultdict(list)
    cohort_concordances = []
    concordant_pairs = 0
    total_pairs = 0

    for cohort in risks.keys():
        strat_loss += criterion(
            torch.stack(risks[cohort]).to(device),
            torch.stack(os_times[cohort]).to(device),
            torch.stack(events[cohort]).to(device),
        ).item()
        concord_stats = concordance_index(
            np.array(events[cohort]).astype(np.bool),
            np.array(os_times[cohort]),
            np.array(risks[cohort]),
        )
        concordances[cohort] = concord_stats[0]
        print(f"{cohort}: {concordances[cohort]}, concord_stats: {concord_stats}")
        cohort_concordances.append(
            str(cohort) + "-" + str(round(concordances[cohort], 4))
        )
        # keep track of total concordant pairs and total pairs for strat concord
        concordant_pairs += (
            concord_stats[1] + 0.5 * concord_stats[3]
        )  # add 0.5 for tied pairs
        total_pairs += concord_stats[1] + concord_stats[2] + concord_stats[3]

    cohort_concordances = "_".join(cohort_concordances)
    strat_concord = concordant_pairs / total_pairs

    torch.cuda.empty_cache()
    return strat_loss, strat_concord, cohort_concordances


class Model1Layer(nn.Module):
    def __init__(self):
        super(Model1Layer, self).__init__()
        self.regression = nn.Linear(192, 1).to(device)

    def forward(self, ftrs):
        return self.regression(ftrs)


class Model2Layers(nn.Module):
    def __init__(self, dropout_prob):
        super(Model2Layers, self).__init__()
        self.hidden = nn.Linear(192, 96).to(device)
        self.regression = nn.Linear(96, 1).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)

    def forward(self, ftrs):
        x = self.hidden(ftrs)
        x = self.dropout(x)
        return self.regression(x)


class Model3Layers(nn.Module):
    def __init__(self, dropout_prob):
        super(Model3Layers, self).__init__()
        self.hidden1 = nn.Linear(192, 96).to(device)
        self.hidden2 = nn.Linear(96, 48).to(device)
        self.regression = nn.Linear(48, 1).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)

    def forward(self, ftrs):
        x = self.hidden1(ftrs)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        return self.regression(x)


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_val_loss = np.inf

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0  # reset
        elif val_loss > (self.min_val_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_net(parameters):
    if args.hyper_sweep:
        print(f"Running hyperparam sweep {args.sweep_n+1}")

    lr = float(parameters.get("lr", 0.01))
    weight_decay = float(parameters.get("weight_decay", 0.1))
    batch_size = int(parameters.get("batch_size", 64))
    n_layers = int(parameters.get("n_layers", 1))
    dropout_prob = float(parameters.get("dropout_prob", 0.0))
    nepochs = args.nepochs

    # removed number of epochs from hyperparams
    hyperparam_str = (
        str(batch_size)
        + "_"
        + str(n_layers)
        + "_"
        + str(lr)
        + "_"
        + str(weight_decay)
        + "_"
        + str(dropout_prob)
        + "_"
        + f"fold{args.fold_n}"
    )
    output_csv = args.runs_csv_location + "/" + hyperparam_str + ".csv"
    writer = open(output_csv, "a+")

    print("\nLoading ftrs 4k train...")
    train_dset = MicroscopeData(args.train_lib, args.ftr_directory)
    print("\nLoading ftrs 4k tune...")
    tune_dset = MicroscopeData(args.tune_lib, args.ftr_directory)

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # instantiate the model
    if n_layers == 1:
        model = Model1Layer()
    if n_layers == 2:
        model = Model2Layers(dropout_prob)
    if n_layers == 3:
        model = Model3Layers(dropout_prob)

    model.to(device)

    # loss and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = cox_ph_loss

    # Load from checkpoint
    if (
        args.use_checkpoint
        and args.checkpoint_file
        and os.path.exists(args.checkpoint_file)
    ):
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"]
        best_auc = checkpoint["best_auc"]
        print(f"Checkpoint loaded: {args.checkpoint_file}")
    else:
        epoch_start = 0
        best_auc = 0.5
        print("No checkpoint provided")

    # sanity check, ensure random predictions are around c=0.5
    if epoch_start == 0 and args.sanity_check:
        print("\nConducting sanity check run on tune dataset")
        t0 = time.time()
        loss, concord, cohort_concord = get_val_loss_concordance(
            model, tune_dset, criterion, batch_size
        )
        t1 = time.time()
        print(
            f"Loss {loss}, concordance overall: {concord}, cohort concordance {cohort_concord}"
        )
        writer = open(output_csv, "a+")
        writer.write(
            "{},{},{},{},{},{},tune,{},{},loss,{},{}\n".format(
                args.fold_n,
                args.sweep_n + 1,
                lr,
                batch_size,
                weight_decay,
                dropout_prob,
                0,
                (t1 - t0) / 60,
                loss,
                "n/a",
            )
        )
        writer.write(
            "{},{},{},{},{},{},tune,{},{},concordance,{},{}\n".format(
                args.fold_n,
                args.sweep_n + 1,
                lr,
                batch_size,
                weight_decay,
                dropout_prob,
                0,
                (t1 - t0) / 60,
                concord,
                cohort_concord,
            )
        )
        writer.close()

    # early stopping criteria
    early_stopping = EarlyStopping(patience=args.max_epochs_no_improvement, delta=1)

    for epoch in range(epoch_start, nepochs):
        start_epoch_time = time.time()

        print("\nEpoch: ", epoch + 1)
        train_loader = get_dataloader(train_dset, batch_size, num_workers=args.workers)

        # TRAINING
        print(f"Examples: {len(train_loader.dataset)}, beginning training")
        mem = psutil.virtual_memory()
        print(
            "Memory used: {}, Free: {}, Available: {}".format(
                round(mem.used / 1024**3, 2),
                round(mem.free / 1024**3, 2),
                round(mem.available / 1024**3, 2),
            ),
            flush=True,
        )

        model.train()
        running_loss = 0
        skipped_batches = 0

        batches = tqdm(train_loader, total=len(train_loader), desc="Training model")
        for i, batch in enumerate(batches):
            ftrs, os_time, event = batch
            pred = model(ftrs)
            os_time = os_time.to(device)
            event = event.to(device)

            if torch.sum(event) > 0:
                loss = criterion(pred, os_time, event)
                if torch.isnan(loss):
                    print("\n WARNING NAN LOSS")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                skipped_batches += 1
                print("Skip, no events in batch")

        writer = open(output_csv, "a+")
        writer.write(
            "{},{},{},{},{},{},train,{},{},loss,{},{}\n".format(
                args.fold_n,
                args.sweep_n + 1,
                lr,
                batch_size,
                weight_decay,
                dropout_prob,
                epoch + 1,
                (t1 - t0) / 60,
                running_loss,
                "n/a",
            )
        )
        writer.close()
        print(f"Running loss: {running_loss}")
        print(f"Skipped batches: {skipped_batches}")

        if args.save_model_location is not None:
            checkpoint_model = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),  # Saves the weights
                "loss": running_loss,
                "optimizer": optimizer.state_dict(),  # Saves the parameters of the optimizer
                "best_auc": best_auc,
            }
            torch.save(
                checkpoint_model,
                args.save_model_location + "/" + hyperparam_str + "_checkpoint.pt",
            )

        #! Delete old loader
        torch.cuda.empty_cache()
        del train_loader

        # Validation loss + concordance
        t0 = time.time()
        loss, concord, cohort_concord = get_val_loss_concordance(
            model, tune_dset, criterion, batch_size
        )
        t1 = time.time()

        writer = open(output_csv, "a+")
        writer.write(
            "{},{},{},{},{},{},tune,{},{},loss,{},{}\n".format(
                args.fold_n,
                args.sweep_n + 1,
                lr,
                batch_size,
                weight_decay,
                dropout_prob,
                epoch + 1,
                (t1 - t0) / 60,
                loss,
                "n/a",
            )
        )
        writer.write(
            "{},{},{},{},{},{},tune,{},{},concordance,{},{}\n".format(
                args.fold_n,
                args.sweep_n + 1,
                lr,
                batch_size,
                weight_decay,
                dropout_prob,
                epoch + 1,
                (t1 - t0) / 60,
                concord,
                cohort_concord,
            )
        )
        writer.close()
        print(
            f"Loss {loss}, concordance overall: {concord}, cohort concordance {cohort_concord}"
        )

        if epoch + 1 > 5 and concord > best_auc and args.save_model_location:
            best_model = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "best_auc": concord,
            }
            torch.save(
                best_model,
                args.save_model_location + "/" + hyperparam_str + "_best_model.pt",
            )
            best_auc = concord

        end_epoch_time = time.time()
        print(
            f"Completed epoch {epoch+1} in time: {(end_epoch_time - start_epoch_time)/60}"
        )
        torch.cuda.empty_cache()

        # check if need early stopping
        if early_stopping.early_stop(loss):
            break

    return best_auc


def evaluate_trained_net(parameterization):
    # train model
    best_auc = train_net(parameterization)
    return best_auc


def main():

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "lr",
                "type": "choice",
                "values": [0.001, 0.0001, 0.00001, 0.000001],
            },
            {
                "name": "weight_decay",
                "type": "choice",
                "values": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
            },
            {
                "name": "batch_size",
                "type": "choice",
                "values": [64, 128, 256, 512, 1024],
            },
            {"name": "n_layers", "type": "choice", "values": [1, 2, 3]},
            {
                "name": "dropout_prob",
                "type": "choice",
                "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            },
        ],
        evaluation_function=evaluate_trained_net,
        objective_name="concordance",
        total_trials=args.bayes_sweep_n,
    )

    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)

    best_objectives = np.array(
        [[trial.objective_mean * 100 for trial in experiment.trials.values()]]
    )

    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )
    obj_plot = render(best_objective_plot)
    torch.save(obj_plot, os.path.join(args.runs_csv_location, "obj_plot.html"))
    render(best_objective_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN for histology")
    parser.add_argument(
        "--train_lib", type=str, default="", help="the training library dictionary"
    )
    parser.add_argument(
        "--tune_lib", type=str, default="", help="the tuning library dictionary."
    )
    parser.add_argument(
        "--ftr_directory",
        type=str,
        default="",
        help="path to the tile ftrs from HIPT model",
    )
    parser.add_argument("--seed", default=416, type=int, help="random seed")
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--use_checkpoint", action="store_true", help="Load from checkpoint"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="",
        help="model checkpoint location, for vremote",
    )
    parser.add_argument("--nepochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--runs_csv_location",
        type=str,
        default="",
        help="directory to save the run information in csv",
    )
    parser.add_argument(
        "--save_model_location",
        type=str,
        default="",
        help="directory to save the checkpoint and best model",
    )
    parser.add_argument(
        "--sweep_n",
        type=int,
        default=0,
        help="If hyperparameter sweep, iteration number",
    )
    parser.add_argument("--hyper_sweep", action="store_true")
    parser.add_argument(
        "--max_epochs_no_improvement",
        type=int,
        default=10,
        help="Number of consecutive epochs with no tuning loss decrease before early stopping",
    )
    parser.add_argument(
        "--bayes_sweep_n", type=int, default=100, help="Number of bayes optim trials"
    )

    global args
    args = parser.parse_args()

    main()
