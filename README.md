# Revisiting Orbital Minimization Method
for Neural Operator Decomposition

---

This repository contains the code used to generate the different experiments and plots contained in [the following paper](https://www.arxiv.org/pdf/2510.21952). One of our experiments is learning Laplacian environments as a follow up to [Proper Laplacian Representation Learning](https://arxiv.org/pdf/2310.10833).

To learn the Laplacian representation of an environment, run the following code:

```
python train_laprepr.py <some_experiment_label>
```

This will train an encoder whose input is the state and the output is the corresponding entry of the smallest $d$ eigenvectors of the Laplacian. Once training is done, a plot of each of the eigenvectors is stored in the folder `results`. 

By default, Joint Orbital Minimization Method (omm_joint.yaml) is used to train the Laplacian encoder. To change hyperparameters, including the optimization objective, you can either add arguments when running `train_laprepr.py`, or store them in a `.yaml` file in the folder `src/hyperparam` and set the `config_file`:

```
python train_laprepr.py <some_experiment_label> --config_file=you_hypers_file.yaml
```

The code requires Jax, Haiku and a few such dependencies.
