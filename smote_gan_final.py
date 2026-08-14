from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from pandas import DataFrame
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline


def two_classes_Abalone(Abalone_df):

    class_category = np.repeat("empty000", Abalone_df.shape[0])


    for i in range(0, Abalone_df["Class_number_of_rings"].size):

        if Abalone_df["Class_number_of_rings"][i] <= 7:

            class_category[i] = 1

        elif Abalone_df["Class_number_of_rings"][i] > 7:

            class_category[i] = 0


    Abalone_df = Abalone_df.drop(["Class_number_of_rings"], axis=1)

    Abalone_df["Class"] = class_category

    return Abalone_df


def four_classes_Abalone(Abalone_df):

    class_category = np.repeat("empty000", Abalone_df.shape[0])

    for i in range(0, Abalone_df["Class_number_of_rings"].size):

        if Abalone_df["Class_number_of_rings"][i] <= 7:

            class_category[i] = int(0)

        elif (

            Abalone_df["Class_number_of_rings"][i] > 7

            and Abalone_df["Class_number_of_rings"][i] <= 10

        ):

            class_category[i] = int(1)

        elif (

            Abalone_df["Class_number_of_rings"][i] > 10

            and Abalone_df["Class_number_of_rings"][i] <= 15

        ):

            class_category[i] = int(2)

        else:

            class_category[i] = int(3)


    Abalone_df = Abalone_df.drop(["Class_number_of_rings"], axis=1)

    Abalone_df["Class"] = class_category

    return Abalone_df


def get_features(Abalone_df, Sex_onehotencoded, test_size):

    features = Abalone_df.iloc[:, np.r_[0:7]]

    X_train, X_test, X_gender, X_gender_test = train_test_split(

        features, Sex_onehotencoded, random_state=10, test_size=test_size

    )

    X_train = np.concatenate((X_train.values, X_gender), axis=1)

    X_test = np.concatenate((X_test.values, X_gender_test), axis=1)

    return X_train, X_test


def get_labels(Abalone_df, test_size):

    labels = Abalone_df.iloc[:, 7]

    y_train, y_test = train_test_split(labels, random_state=10, test_size=test_size)

    train_list = [int(i) for i in y_train.ravel()]

    y_train = np.array(train_list)

    test_list = [int(i) for i in y_test.ravel()]  # Flattening the matrix

    y_test = np.array(test_list)


    return y_train, y_test


def GANs_two_class_real_data(

    X_train: DataFrame, y_train, minority

):  # Defining the real data for GANs

    X_real = []

    y_train = y_train.ravel()

    print(type(X_train))

    X_train = X_train.reset_index(drop=True)

    for i in range(len(y_train)):

        if int(y_train[i]) == minority:

            X_real.append(X_train.loc[i])

    X_real = np.array(X_real)

    y_real = np.ones((X_real.shape[0],))

    return X_real, y_real


def GANs_four_class_real_data(X_train, y_train):

    X_real_0 = []

    X_real_2 = []

    X_real_3 = []

    for i in range(len(y_train)):

        if int(y_train[i]) == 0:

            X_real_0.append(X_train[i])

        if int(y_train[i]) == 2:

            X_real_2.append(X_train[i])

        if int(y_train[i]) == 3:

            X_real_3.append(X_train[i])

    X_real_0 = np.array(X_real_0)

    X_real_2 = np.array(X_real_2)

    X_real_3 = np.array(X_real_3)


    y_real_0 = np.full((X_real_0.shape[0],), 0)

    y_real_2 = np.full((X_real_2.shape[0],), 2)

    y_real_3 = np.full((X_real_3.shape[0],), 3)


    return X_real_0, X_real_2, X_real_3, y_real_0, y_real_2, y_real_3


######### GETTING THE GPU ###########

def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device("cuda")

    else:

        return torch.device("cpu")


def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list, tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)


class DeviceDataLoader:

    """Wrap a dataloader to move data to a device"""


    def __init__(self, dl, device):

        self.dl = dl

        self.device = device


    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl:

            yield to_device(b, self.device)


    def __len__(self):

        """Number of batches"""

        return len(self.dl)


class train_discriminator:

    def __init__(

        self,

        real_data,

        latent_data,

        opt_d,

        generator,

        discriminator,

        device,

        minority_class,

        majority_class,

    ):

        self.real_data = real_data

        self.latent_data = latent_data

        self.opt_d = opt_d

        self.discriminator = discriminator

        self.generator = generator

        self.device = device

        self.minority_class = minority_class

        self.majority_class = majority_class


    def __call__(self):

        self.opt_d.zero_grad()

        #print("real data")

        #print(self.real_data)

        #input()

        # Pass real data through discriminator

        real_preds = self.discriminator(self.real_data)

        if self.minority_class == 0:

            real_targets = torch.zeros_like(real_preds, device=self.device)

        elif self.minority_class == 1:

            real_targets = torch.ones_like(real_preds, device=self.device)

        elif self.minority_class == 2:

            real_targets = torch.full_like(real_preds, 2.0, device=self.device)

        if self.minority_class == 3:

            real_targets = torch.full_like(real_preds, 3.0, device=self.device)

        # real_targets = torch.ones_like(real_preds, device=self.device)

        # real_loss = F.binary_cross_entropy(real_preds, real_targets)

        criterion = nn.BCEWithLogitsLoss()

        real_loss = criterion(real_preds, real_targets)

        real_score = torch.mean(real_preds).item()


        # Generate fake data

        # latent = torch.randn(batch_size, latent_size, 1, 1, device=device)

        fake_data = self.generator(self.latent_data)

        # fake = gen(X_oversampled.float().to(device))


        # Pass fake data through discriminator

        #print("fake data")

        #print(fake_data)

        #input()

        fake_preds = self.discriminator(fake_data)

        if self.majority_class == 0:

            fake_targets = torch.zeros_like(fake_preds, device=self.device)

        elif self.majority_class == 1:

            fake_targets = torch.ones_like(fake_preds, device=self.device)


        # fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)


        fake_loss = criterion(fake_preds, fake_targets)

        fake_score = torch.mean(fake_preds).item()


        # Update discriminator weights

        loss = (real_loss + fake_loss) / 2

        loss.backward()

        self.opt_d.step()

        return loss.item(), real_score, fake_score


class train_generator:

    def __init__(

        self, latent_data, opt_g, generator, discriminator, device, minority_class

    ):

        self.latent_data = latent_data

        self.opt_g = opt_g

        self.generator = generator

        self.discriminator = discriminator

        self.device = device

        self.minority_class = minority_class


    def __call__(self):

        # Clear generator gradients

        self.opt_g.zero_grad()


        # Generate fake images

        fake_data = self.generator(self.latent_data)


        # Try to fool the discriminator

        preds = self.discriminator(

            fake_data

        )  # We put the fake data generated by generator into the discriminator

        if self.minority_class == 0:

            targets = torch.zeros_like(preds, device=self.device)

        elif self.minority_class == 1:

            targets = torch.ones_like(preds, device=self.device)

        elif self.minority_class == 2:

            targets = torch.full_like(preds, 2.0, device=self.device)

        elif self.minority_class == 3:

            targets = torch.full_like(preds, 3.0, device=self.device)

        # targets = torch.ones_like(preds, device=self.device)

        criterion = nn.BCEWithLogitsLoss()

        loss = criterion(preds, targets)


        # Update generator weights

        loss.backward()

        self.opt_g.step()


        return loss.item()


class SG_fit:

    def __init__(

        self,

        epochs,

        lr,

        discriminator,

        generator,

        X_oversampled,

        train_dl,

        device,

        minority_class,

        majority_class,

        start_idx=1,

    ):

        self.epochs = epochs

        self.lr = lr

        self.discriminator = discriminator

        self.generator = generator

        self.X_oversampled = X_oversampled

        self.train_dl = train_dl

        self.device = device

        self.minority_class = minority_class

        self.majority_class = majority_class


    def __call__(self):

        torch.cuda.empty_cache()


        # Losses & scores

        losses_g = []

        losses_d = []

        real_scores = []

        fake_scores = []


        # Create optimizers

        opt_d = torch.optim.Adam(

            self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999)

        )

        opt_g = torch.optim.Adam(

            self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999)

        )


        for epoch in range(self.epochs):

            for real_data, _ in tqdm(self.train_dl):

                # Train discriminator

                train_disc = train_discriminator(

                    real_data,

                    self.X_oversampled,

                    opt_d,

                    self.generator,

                    self.discriminator,

                    self.device,

                    self.minority_class,

                    self.majority_class,

                )

                loss_d, real_score, fake_score = train_disc()

                # Train generator

                train_gen = train_generator(

                    self.X_oversampled,

                    opt_g,

                    self.generator,

                    self.discriminator,

                    self.device,

                    self.minority_class,

                )

                loss_g = train_gen()


            # Record losses & scores

            losses_g.append(loss_g)

            losses_d.append(loss_d)

            real_scores.append(real_score)

            fake_scores.append(fake_score)


            # Log losses & scores (last batch)

            print(

                "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(

                    epoch + 1, self.epochs, loss_g, loss_d, real_score, fake_score

                )

            )


            # Save generated images

            # save_samples(epoch+start_idx, fixed_latent, show=False)


        return losses_g, losses_d, real_scores, fake_scores


class G_fit:

    def __init__(

        self,

        epochs,

        lr,

        discriminator,

        generator,

        train_dl,

        device,

        minority_class,

        majority_class,

        start_idx=1,

    ):

        self.epochs = epochs

        self.lr = lr

        self.discriminator = discriminator

        self.generator = generator

        self.train_dl = train_dl

        self.device = device

        self.minority_class = minority_class

        self.majority_class = majority_class


    def __call__(self):

        torch.cuda.empty_cache()


        # Losses & scores

        losses_g = []

        losses_d = []

        real_scores = []

        fake_scores = []


        # Create optimizers

        opt_d = torch.optim.Adam(

            self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999)

        )

        opt_g = torch.optim.Adam(

            self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999)

        )


        for epoch in range(self.epochs):

            for real_data, _ in tqdm(self.train_dl):

                # Train discriminator

                latent_data = torch.randn(

                    real_data.shape[0], real_data.shape[1], device=self.device

                )

                train_disc = train_discriminator(

                    real_data,

                    latent_data,

                    opt_d,

                    self.generator,

                    self.discriminator,

                    self.device,

                    self.minority_class,

                    self.majority_class,

                )

                loss_d, real_score, fake_score = train_disc()

                # Train generator

                train_gen = train_generator(

                    latent_data,

                    opt_g,

                    self.generator,

                    self.discriminator,

                    self.device,

                    self.minority_class,

                )

                loss_g = train_gen()


            # Record losses & scores

            losses_g.append(loss_g)

            losses_d.append(loss_d)

            real_scores.append(real_score)

            fake_scores.append(fake_score)


            # Log losses & scores (last batch)

            print(

                "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(

                    epoch + 1, self.epochs, loss_g, loss_d, real_score, fake_score

                )

            )


            # Save generated images

            # save_samples(epoch+start_idx, fixed_latent, show=False)


        return losses_g, losses_d, real_scores, fake_scores


def get_generator_block(input_dim, output_dim):  # Generator Block

    return nn.Sequential(

        nn.Linear(input_dim, output_dim),

        nn.BatchNorm1d(output_dim),

        nn.ReLU(inplace=True),

    )


class GANs_Generator(nn.Module):  # Generator Model

    def __init__(self, z_dim, im_dim, hidden_dim):

        super(GANs_Generator, self).__init__()

        self.generator = nn.Sequential(

            get_generator_block(z_dim, hidden_dim),

            get_generator_block(hidden_dim, hidden_dim * 2),

            get_generator_block(hidden_dim * 2, hidden_dim * 4),

            get_generator_block(hidden_dim * 4, hidden_dim * 8),

            nn.Linear(hidden_dim * 8, im_dim),

            nn.Softplus(),

            #nn.Sigmoid(),

        )


    def forward(self, noise):

        return self.generator(noise)


    def get_generator(self):

        return self.generator


def get_discriminator_block(input_dim, output_dim):  # Discriminator Block

    return nn.Sequential(

        nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.2, inplace=True)

    )


class GANs_Discriminator(nn.Module):  # Discriminator Model

    def __init__(self, im_dim, hidden_dim):

        super(GANs_Discriminator, self).__init__()

        self.discriminator = nn.Sequential(

            get_discriminator_block(im_dim, hidden_dim * 4),

            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),

            get_discriminator_block(hidden_dim * 2, hidden_dim),

            nn.Linear(hidden_dim, 1),

        )


    def forward(self, image):

        return self.discriminator(image)


    def get_disc(self):

        return self.discriminator


def f1_sg(

    X_train,

    y_train,

    X_train_SMOTE,

    y_train_SMOTE,

    X_real,

    y_real,

    X_oversampled,

    device,

    lr,

    epochs,

    batch_size,

    minority_class,

    majority_class,

):  # Fetches us the trained generators

    # X_oversampled = X_train_SMOTE[(X_train.shape[0]):]

    # X_oversampled = torch.from_numpy(X_oversampled)

    # X_oversampled = to_device(X_oversampled.float(), device)


    # print(X_oversampled.shape)


    # X_real, y_real = GANs_two_class_real_data(X_train, y_train)


    ##### Wrapping all the tensors in a Tensor Dataset. #####

    # tensor_x = torch.Tensor(X_real)

    # tensor_y = torch.Tensor(y_real)

    my_dataset = TensorDataset(torch.Tensor(X_real), torch.Tensor(y_real))


    # lr = 0.0002

    # epochs = 150

    # batch_size = 128


    ##### Loading our Tensor Dataset into a Dataloader. #####

    train_dl = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

    train_dl = DeviceDataLoader(train_dl, device)


    ##### Initialising the generator and discriminator objects ######

    gen1 = GANs_Generator(X_train.shape[1], X_train.shape[1], 128)

    disc1 = GANs_Discriminator(X_train.shape[1], 128)


    ##### Loading the model in GPU #####

    generator_SG = to_device(gen1.generator, device)

    discriminator_SG = to_device(disc1.discriminator, device)


    SG_fit_func = SG_fit(

        epochs,

        lr,

        discriminator_SG,

        generator_SG,

        X_oversampled,

        train_dl,

        device,

        minority_class,

        majority_class,

    )  # fit function object initiated.

    history1 = SG_fit_func()  # Callable object


    return generator_SG


def f1_g(

    X_train,

    y_train,

    X_train_SMOTE,

    y_train_SMOTE,

    X_real,

    y_real,

    device,

    lr,

    epochs,

    batch_size,

    minority_class,

    majority_class,

):  # Fetches us the trained generators

    # X_oversampled = X_train_SMOTE[(X_train.shape[0]):]

    # X_oversampled = torch.from_numpy(X_oversampled)

    # X_oversampled = to_device(X_oversampled.float(), device)


    # print(X_oversampled.shape)


    # X_real, y_real = GANs_two_class_real_data(X_train, y_train)


    ##### Wrapping all the tensors in a Tensor Dataset. #####

    # tensor_x = torch.Tensor(X_real)

    # tensor_y = torch.Tensor(y_real)

    my_dataset = TensorDataset(torch.Tensor(X_real), torch.Tensor(y_real))


    # lr = 0.0002

    # epochs = 150

    # batch_size = 128


    ##### Loading our Tensor Dataset into a Dataloader. #####

    train_dl = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

    train_dl = DeviceDataLoader(train_dl, device)


    gen2 = GANs_Generator(X_train.shape[1], X_train.shape[1], 128)

    disc2 = GANs_Discriminator(X_train.shape[1], 128)


    generator_G = to_device(gen2.generator, device)

    discriminator_G = to_device(disc2.discriminator, device)


    G_fit_func = G_fit(

        epochs,

        lr,

        discriminator_G,

        generator_G,

        train_dl,

        device,

        minority_class,

        majority_class,

    )

    history2 = G_fit_func()


    return generator_G


from sklearn.metrics import (

    f1_score,

    accuracy_score,

    precision_score,

    recall_score,

    classification_report,

)


def shuffle_in_unison(a, b):  # Shuffling the features and labels in unison.

    assert (

        len(a) == len(b)

    )  # In Python, the assert statement is used to continue the execute if the given condition evaluates to True.

    shuffled_a = np.empty(

        a.shape, dtype=a.dtype

    )  # Return a new array of given shape and type, without initializing entries.

    shuffled_b = np.empty(b.shape, dtype=b.dtype)

    permutation = np.random.permutation(len(a))

    for old_index, new_index in enumerate(permutation):

        shuffled_a[new_index] = a[old_index]

        shuffled_b[new_index] = b[old_index]

    return shuffled_a, shuffled_b

class TimedSMOTE(BaseEstimator):
    """Thin wrapper measuring fit_resample time in ms."""
    sampling_time = 0

    def __init__(self, sampling_strategy="auto", k_neighbors=5, random_state=None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        start = time.perf_counter_ns()
        sampler = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        X_res, y_res = sampler.fit_resample(X, y)
        TimedSMOTE.sampling_time += (time.perf_counter_ns() - start) / 1e6
        return X_res, y_res

class TimedADASYN(BaseEstimator):
    """Thin wrapper measuring ADASYN fit_resample time in ms."""
    sampling_time = 0

    def __init__(self, sampling_strategy=0.95, n_neighbors=5, random_state=None):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        start = time.perf_counter_ns()
        sampler = ADASYN(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        X_res, y_res = sampler.fit_resample(X, y)
        TimedADASYN.sampling_time += (time.perf_counter_ns() - start) / 1e6
        return X_res, y_res

class HybridGAN(BaseEstimator):
    """SMOTE/ADASYN + GAN amalgamation as imblearn sampler.

    Wraps the existing SMOTE/ADASYN + GAN logic from this file
    (GANs_two_class_real_data / f1_sg) without modifying it.
    ``base_sampler`` is any imblearn over-sampler (SMOTE(), ADASYN(...))
    injected via __init__ so Pipeline.clone() works.
    """
    sampling_time = 0

    def __init__(self, base_sampler=None, epochs=150, lr=0.0002, batch_size=128, device=None, hidden_dim=128):
        self.base_sampler = base_sampler
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        # Preserve DataFrame column handling; Pipeline may pass ndarray or DataFrame
        start = time.perf_counter_ns()
        is_df = isinstance(X, pd.DataFrame)
        X_df = X if is_df else pd.DataFrame(X)
        y_s = pd.Series(y) if not isinstance(y, pd.Series) else y
        y_np = y_s.to_numpy()

        # minority / majority detection (same as runOnDataset)
        counts = y_s.value_counts()
        if len(counts) == 2:
            minority = counts.idxmin()
            majority = counts.idxmax()
        else:
            minority = 1 if (y_np == 1).sum() < (y_np == 0).sum() else 0
            majority = 1 - minority if minority in (0, 1) else 0

        device = self.device or torch.device("cpu")

        # 1. Over-sample with injected base_sampler (SMOTE or ADASYN)
        sampler = self.base_sampler if self.base_sampler is not None else SMOTE()
        X_over, y_over = sampler.fit_resample(X_df, y_s)

        # Ensure DataFrame for downstream helpers
        if not isinstance(X_over, pd.DataFrame):
            X_over = pd.DataFrame(X_over, columns=X_df.columns)

        # 2. Real minority data for GAN discriminator (uses existing helper)
        X_real, y_real = GANs_two_class_real_data(X_df, y_np, minority)

        # 3. Synthetic tail produced by base_sampler -> used as GAN latent input
        n_original = X_df.shape[0]
        X_tail = X_over.iloc[n_original:].to_numpy() if isinstance(X_over, pd.DataFrame) else X_over[n_original:]
        if X_tail.shape[0] == 0:
            return X_over, y_over

        X_tail_t = to_device(torch.from_numpy(X_tail).float(), device)

        # 4. Train GAN refining the over-sampled points (reuses existing f1_sg)
        generator = f1_sg(
            X_df, y_np, X_over, y_over, X_real, y_real, X_tail_t,
            device, self.lr, self.epochs, self.batch_size, minority, majority
        )
        X_syn = generator(X_tail_t.float().to(device)).cpu().detach().numpy()

        # 5. Re-assemble: original + GAN-refined synthetic points
        X_head = X_over.iloc[:n_original].to_numpy() if isinstance(X_over, pd.DataFrame) else X_over[:n_original]
        X_res = np.concatenate([X_head, X_syn], axis=0)

        # 6. Shuffle in unison with labels (reuses existing helper)
        if isinstance(y_over, pd.Series):
            y_over_np = y_over.to_numpy()
        else:
            y_over_np = np.array(y_over)
        X_res, y_res = shuffle_in_unison(X_res, y_over_np)

        if is_df:
            X_res = pd.DataFrame(X_res, columns=X_df.columns)

        end = time.perf_counter_ns()

        total_time = (end - start) / 1e6

        HybridGAN.sampling_time += total_time

        return X_res, y_res

class GANSampler(BaseEstimator):
    """Pure GAN sampler (f1_g) as imblearn sampler.

    Uses base_sampler (e.g. SMOTE()) only to determine how many
    synthetic points to generate (majority - minority). Trains GAN
    via existing f1_g on random noise, then concatenates original
    head with GAN samples.
    """
    sampling_time = 0

    def __init__(self, base_sampler=None, epochs=150, lr=0.0002, batch_size=128, device=None):
        self.base_sampler = base_sampler
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        start = time.perf_counter_ns()
        is_df = isinstance(X, pd.DataFrame)
        X_df = X if is_df else pd.DataFrame(X)
        y_s = pd.Series(y) if not isinstance(y, pd.Series) else y
        y_np = y_s.to_numpy()
        counts = y_s.value_counts()
        if len(counts) == 2:
            minority = counts.idxmin()
            majority = counts.idxmax()
        else:
            minority = 1 if (y_np == 1).sum() < (y_np == 0).sum() else 0
            majority = 1 - minority if minority in (0, 1) else 0
        device = self.device or torch.device("cpu")
        # determine synthetic count via base_sampler (SMOTE) tail length
        sampler = self.base_sampler if self.base_sampler is not None else SMOTE()
        X_over, y_over = sampler.fit_resample(X_df, y_s)
        if not isinstance(X_over, pd.DataFrame):
            X_over = pd.DataFrame(X_over, columns=X_df.columns)
        n_original = X_df.shape[0]
        n_synth = X_over.shape[0] - n_original
        if n_synth <= 0:
            return X_over, y_over
        X_real, y_real = GANs_two_class_real_data(X_df, y_np, minority)
        # train pure GAN (random noise latent)
        generator = f1_g(X_df, y_np, X_over, y_over, X_real, y_real, device, self.lr, self.epochs, self.batch_size, minority, majority)
        n_feat = X_df.shape[1]
        noise = torch.randn(n_synth, n_feat, device=device)
        X_syn = generator(noise.float().to(device)).cpu().detach().numpy()
        X_head = X_over.iloc[:n_original].to_numpy() if isinstance(X_over, pd.DataFrame) else X_over[:n_original]
        X_res = np.concatenate([X_head, X_syn], axis=0)
        if isinstance(y_over, pd.Series):
            y_over_np = y_over.to_numpy()
        else:
            y_over_np = np.array(y_over)
        X_res, y_res = shuffle_in_unison(X_res, y_over_np)
        if is_df:
            X_res = pd.DataFrame(X_res, columns=X_df.columns)

        end = time.perf_counter_ns()
        total_time = (end - start) / 1e6

        GANSampler.sampling_time += total_time
        return X_res, y_res




"""

def model_rf(X_train, y_train, X_test, y_test):

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    f1_mes = f1_score(y_test, y_pred, average='weighted')

    precision = precision_score(y_test, y_pred, average='weighted')

    recall = recall_score(y_test, y_pred, average='weighted')

    return accuracy, f1_mes, precision, recall, model

"""


def hellinger(p, q):

    return np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum())


def hellingerDistance(train, generated):

    df1 = pd.DataFrame(train)

    df2 = pd.DataFrame(generated)

    df2.columns = df1.columns


    common = pd.merge(df1, df2, how="inner")


    df2 = pd.concat([df2, common]).drop_duplicates(keep=False)


    df1 = df1.div(df1.sum(), axis=1)

    df2 = df2.div(df2.sum(), axis=1)


    # print((df2.head()))

    hellinger_distances = {}

    for col in df1.columns:

        dist = hellinger(df1[col], df2[col])

        hellinger_distances[col] = dist


    # print(hellinger_distances)

    average_distance = np.mean(list(hellinger_distances.values()))

    return average_distance


def model_rf(X, y, df,model,model_name,sampler=None):

    acc_arr = []

    f1_arr = []

    pre_arr = []

    rec_arr = []

    sampling_time_start = type(sampler).sampling_time if sampler else 0

    cv = 5
    outer_iteration = 30

    for i in range(outer_iteration):

        if sampler:
            pipeline = Pipeline([("sampler",sampler),("clf",model)])
        else:
            pipeline = Pipeline([("clf",model)])

        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        """

      results = cross_validate(estimator=model,

                                          X=X,

                                          y=y,

                                          cv=5,

                                          scoring=scoring)

      print(results)

      accuracy = results['test_accuracy']

      f1_mes = results['test_f1_score']

      precision = results['test_precision']

      recall = results['test_recall']

      return accuracy, f1_mes, precision, recall, model

      """


        df = pd.concat(

            [

                df,

                pd.DataFrame(classification_report(y, y_pred, output_dict=True))

                .transpose()

                .drop("macro avg")

                .reindex(["0", "1", "weighted avg", "accuracy"]),

            ]

        )

        # print(df)

        accuracy = accuracy_score(y, y_pred)

        acc_arr.append(accuracy)


        f1_mes = f1_score(y, y_pred, average="weighted")

        f1_arr.append(f1_mes)


        precision = precision_score(y, y_pred, average="weighted")

        pre_arr.append(precision)


        recall = recall_score(y, y_pred, average="weighted")

        rec_arr.append(recall)

    sampling_time_end = type(sampler).sampling_time if sampler else 0

    total_sampling_time = sampling_time_end - sampling_time_start


    return acc_arr, f1_arr, pre_arr, rec_arr, model, (total_sampling_time / (cv * outer_iteration)),df


def runOnDataset(file_name_without_extension: str, model, model_name):

    # reset class-level accumulators so each dataset's timing is isolated
    TimedSMOTE.sampling_time = 0
    TimedADASYN.sampling_time = 0
    HybridGAN.sampling_time = 0
    GANSampler.sampling_time = 0

    device = torch.device("cpu")

    minority = 1

    majority = 0

    df = pd.read_csv(f"./Datasets/{file_name_without_extension}.csv")

    df = df.dropna(axis=0)


    label_encoder = LabelEncoder()

    # onehot_encoder = OneHotEncoder(sparse=False)

    onehot_encoder = OneHotEncoder(sparse_output=False)


    if file_name_without_extension == "abalone":

        df = two_classes_Abalone(df)

        print(df)

        Sex_labelencoded = label_encoder.fit_transform(df["Sex"])

        Sex_labelencoded = Sex_labelencoded.reshape(len(Sex_labelencoded), 1)

        Sex_onehotencoded = onehot_encoder.fit_transform(Sex_labelencoded)


        df["Sex"] = Sex_onehotencoded

        # print(df)

        X = df.drop(["Class"], axis=1)

        df["Class"] = [int(x) for x in df["Class"]]

        y = df["Class"]


    elif file_name_without_extension == "ecoli":

        # encoding on first column just ike sex in abalone

        Seq_labelencoded = label_encoder.fit_transform(df["SEQUENCE_NAME"])

        Seq_labelencoded = Seq_labelencoded.reshape(len(Seq_labelencoded), 1)

        Seq_onehotencoded = onehot_encoder.fit_transform(Seq_labelencoded)


        df["SEQUENCE_NAME"] = Seq_onehotencoded

        # rename site column as Class

        # replace positive by 1 and negative by 0

        X = df.drop(["Class"], axis=1)

        df["Class"] = [

            int(x) for x in df["Class"]

        ]  # in case of float class e.g. (0.0,1.0)

        y = df["Class"]


    else:

        X = df.drop(["Class"], axis=1)

        df["Class"] = [

            int(x) for x in df["Class"]

        ]  # in case of float class e.g. (0.0,1.0)

        y = df["Class"]

    # print(y)


    # X_train, X_test, y_train, y_test = train_test_split(df.drop(['Class'], axis=1), df['Class'], test_size=0.2, random_state=10)


    #### Calculating train and test accuracy and f1 score of non oversampled training data ####


    one = df[df["Class"] == 1]

    print(len(one))


    zero = df[df["Class"] == 0]

    print(len(zero))


    if len(one) < len(zero):

        minority = 1

        majority = 0

    else:

        minority = 0

        majority = 1


    print("minority:", minority)

    print("majority:", majority)


    print("reached here 1")

    results = pd.DataFrame()


    print("reached here 2")

    minorityTrainData = X[y == minority]


    (

        Normal_accuracy,

        Normal_f1_score,

        Normal_precision,

        Normal_recall,

        model_normal,
        _,

        results,

    ) = model_rf(X, y, model=model,model_name=model_name,df=results)

    print("Normal_accuracy: ", Normal_accuracy)

    print("Normal_f1_score: ", Normal_f1_score)

    print("Normal_precision: ", Normal_precision)

    print("Normal_recall: ", Normal_recall)

    # results.loc[len(results)] = [0]*4

    # print(classification_report(y_test, model_normal.predict(X_test)))


    print("Before OverSampling, counts of label '0': {}".format(sum(y == 0)))

    print("Before OverSampling, counts of label '1': {}".format(sum(y == 1)))


    device = torch.device("cpu")

    lr = 0.0002
    epochs = 150
    batch_size = 128

    # All oversampling now happens inside model_rf via Pipeline (no manual fit_resample here)
    # X, y are unadulterated original data

    # placeholder hellinger / timing values
    sm_hell = g_hell = sg_hell = ad_hell = ag_hell = 0.0
    smote_time = g_time = sg_time = ada_time = ag_time = 0.0

    # SMOTE via Pipeline
    (
        Smote_accuracy,
        Smote_f1_score,
        Smote_precision,
        Smote_recall,
        model_smote,
        smote_time,
        results,
    ) = model_rf(X, y, results,model=model,model_name=model_name,sampler=TimedSMOTE())

    # SMOTified GAN via HybridGAN (SMOTE + f1_sg)
    SG_accuracy, SG_f1_score, SG_precision, SG_recall, model_SG, sg_time, results = model_rf(
        X, y, results,model=model,model_name=model_name, sampler=HybridGAN(base_sampler=SMOTE(), epochs=epochs, lr=lr, batch_size=batch_size, device=device)
    )

    # Pure GAN via GANSampler (random noise + f1_g, count from SMOTE)
    G_accuracy, G_f1_score, G_precision, G_recall, model_G, g_time, results = model_rf(
        X, y, results,model=model,model_name=model_name,sampler=GANSampler(base_sampler=SMOTE(), epochs=epochs, lr=lr, batch_size=batch_size, device=device)
    )

    if file_name_without_extension != "drd":
        ADA_accuracy, ADA_f1_score, ADA_precision, ADA_recall, model_ADA, ada_time, results = model_rf(
            X, y, results,model=model,model_name=model_name,sampler=TimedADASYN(sampling_strategy=0.95)
        )

        AG_accuracy, AG_f1_score, AG_precision, AG_recall, model_AG, ag_time, results = model_rf(
            X, y, results,model=model,model_name=model_name,sampler=HybridGAN(base_sampler=ADASYN(sampling_strategy=0.95), epochs=epochs, lr=lr, batch_size=batch_size, device=device)
        )

    if file_name_without_extension == "drd":

        output_df = pd.DataFrame(

            {
                "model_name": [model_name * len(Normal_accuracy)],

                "Normal_accuracy": Normal_accuracy,

                "Normal_f1_score": Normal_f1_score,

                "Normal_precision": Normal_precision,

                "Normal_recall": Normal_recall,

                "SMOTE_accuracy": Smote_accuracy,

                "SMOTE_f1_score": Smote_f1_score,

                "SMOTE_precision": Smote_precision,

                "SMOTE_recall": Smote_recall,

                "SG_accuracy": SG_accuracy,

                "SG_f1_score": SG_f1_score,

                "SG_precision": SG_precision,

                "SG_recall": SG_recall,

                "G_accuracy": G_accuracy,

                "G_f1_score": G_f1_score,

                "G_precision": G_precision,

                "G_recall": G_recall,

                "Smote_hell": sm_hell,

                "Gan_hell": g_hell,

                "SG_hell": sg_hell,

                "Smote_time(ms)": smote_time,

                "Gan_time": g_time,

                "SG_time": sg_time

            }

        )

    else:

        output_df = pd.DataFrame(

            {
                "model_name": [model_name * len(Normal_accuracy)],

                "Normal_accuracy": Normal_accuracy,

                "Normal_f1_score": Normal_f1_score,

                "Normal_precision": Normal_precision,

                "Normal_recall": Normal_recall,

                "SMOTE_accuracy": Smote_accuracy,

                "SMOTE_f1_score": Smote_f1_score,

                "SMOTE_precision": Smote_precision,

                "SMOTE_recall": Smote_recall,

                "ADASYN_accuracy": ADA_accuracy,

                "ADASYN_f1_score": ADA_f1_score,

                "ADASYN_precision": ADA_precision,

                "ADASYN_recall": ADA_recall,

                "SG_accuracy": SG_accuracy,

                "SG_f1_score": SG_f1_score,

                "SG_precision": SG_precision,

                "SG_recall": SG_recall,

                "AG_accuracy": AG_accuracy,

                "AG_f1_score": AG_f1_score,

                "AG_precision": AG_precision,

                "AG_recall": AG_recall,

                "G_accuracy": G_accuracy,

                "G_f1_score": G_f1_score,

                "G_precision": G_precision,

                "G_recall": G_recall,

                "Smote_hell": sm_hell,

                "Ada_hell": ad_hell,

                "Gan_hell": g_hell,

                "SG_hell": sg_hell,

                "AG_hell": ag_hell,

                "Smote_time(ms)": smote_time,

                "Ada_time": ada_time,

                "Gan_time": g_time,

                "SG_time": sg_time,

                "AG_time": ag_time

            }

        )

    output_df.to_csv(
        f"./NewResults/{file_name_without_extension}_result.csv",
        mode="a",
        index=False,
        header=not Path(
            f"./NewResults/{file_name_without_extension}_result.csv"
        ).exists(),
    )


def main():

    # cwd = Path.cwd()

    #

    # datasets = cwd / "Datasets"

    #

    # files = data.glob("*.csv")

    #

    # sorted_files = sorted(files, key=lambda f: f.stat().st_size)

    #

    #

    # for file in sorted_files:

    #     runOnDataset(file.stem)

    files = [

        # "drd","drp","dtcr","fhs","ggcm","pid","tsd"
        "bcwd"

    ]

    for f in files:

        #sys.stdout.write(f"\n\nOperating on {file}\n\n")

        runOnDataset(f, model=RandomForestClassifier(),model_name="random_forest_classifier")
        runOnDataset(f, model=SVC(),model_name="simple_vector_classifier")
        runOnDataset(f, model=LogisticRegression(),model_name="logistic_regression_classifier")


if __name__ == "__main__":

    main()
