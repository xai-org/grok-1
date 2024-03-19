# Grok-1

This repository contains JAX example code for loading and running the Grok-1 open-weights model.

Make sure to download the checkpoint and place `ckpt-0` directory in `checkpoints` before run the project.

## 1. Downloading the weights

You can download the weights using a torrent client in the following magnet link:
```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

## 2. Installation

1. Install the project dependencies

```bash
pip install -r requirements.txt
```

2. Run the project

```bash
python run.py
```

The script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B/Billion parameters), a machine with enough GPU memory is required to test the model with the example code.

The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

## 3. Requirements

Make sure to attend the following requirements before run the project:

 - Needs either either a TPU or GPU (NVIDIA/AMD supported only)
 - They have to be 8 devices (in the context of TPUs (Tensor Processing Units) or GPUs (Graphics Processing Units), they are typically talking about having access to a total of 8 individual processing units)

# License

The code and associated Grok-1 weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Grok-1.
