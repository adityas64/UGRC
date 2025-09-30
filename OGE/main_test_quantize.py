from __future__ import division, absolute_import, print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim
from Models import modelpool  
from Preprocess import datapool  
from utils import seed_all, get_logger
from attack.OGE import OGE
from attack.data_conversion import hamming_distance
import torch.nn.functional as F
import copy
import time
import matplotlib.pyplot as plt
from helper import AverageMeter
import numpy as np 

# Argument parser
parser = argparse.ArgumentParser(
    description='Training SNN with OGE attack integration',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data and model setup
parser.add_argument('--data_path', default='./data/', type=str,
                    help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'],
                    required=True,
                    help='Choose between CIFAR-10/100, ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='vgg16', type=str,
                    help='Model architecture to use')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')


# Checkpointing and logging
parser.add_argument('--save_path', type=str, default='./checkpoints/',
                    help='Directory to save checkpoints and logs.')


# Bit Flip Attack parameters
parser.add_argument('--enable_bfa', action='store_true',
                    help='Enable the bit-flip attack')
parser.add_argument('--attack_sample_size', type=int, default=128,
                    help='Attack sample size')
parser.add_argument('--n_iter', type=int, default=500,
                    help='Number of attack iterations')
parser.add_argument('--k_top', type=int, default=10,
                    help='Top-k gradients for bit-level gradient check')

# Device configuration
parser.add_argument('--gpu_id', type=int, default=0,
                    help='ID of GPU to use.')
parser.add_argument('--workers', type=int, default=4,
                    help='Number of data loading workers.')
parser.add_argument('--manual_seed', type=int, default=42,
                    help='Manual seed for random number generators.')

args = parser.parse_args()

def main():
    # Set random seeds for reproducibility
    if args.manual_seed is not None:
        seed_all(args.manual_seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    args.use_cuda = torch.cuda.is_available()

    # Data loading
    train_loader, test_loader = datapool(args.dataset, args.batch_size)

    # Model initialization
    model = modelpool(args.arch, args.dataset)
    model.set_L(args.L)
    model.to(device)


    # Create save directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)


    # Initialize logger
    logger = get_logger(os.path.join(args.save_path, 'training.log'))

    if args.enable_bfa:
        num_runs = 1  # Number of times to perform the attack
        attack_numbers = []
        hamming_distances = []

        logger.info(f'Performing OGE attack {num_runs} times')

        for attack_idx in range(num_runs):
            # Create a new attacker for each run
            attacker = OGE(criterion, args.k_top)
            # Reset model to clean state
            model_attack = copy.deepcopy(model)
            model_clean = copy.deepcopy(model)

            num_attacks, hamming_distance, bits_flips, top1_accuracies = perform_oge_attack(
                attacker, model_attack, model_clean, train_loader, test_loader,
                criterion, args.n_iter, logger, attack_idx + 1)
            attack_numbers.append(bits_flips)
            hamming_distances.append(hamming_distance)

            logger.info(f'Attack {attack_idx+1}: Required {bits_flips} bits to drop accuracy below 11%')
            logger.info(f'Attack {attack_idx+1}: Hamming distance at that point: {hamming_distance}')

        logger.info(f'Attack numbers over {num_runs} runs: {attack_numbers}')
        logger.info(f'Hamming distances over {num_runs} runs: {hamming_distances}')

        # Optionally, save these arrays to a file or plot them
        np.save(os.path.join(args.save_path, 'attack_numbers.npy'), np.array(attack_numbers))
        np.save(os.path.join(args.save_path, 'hamming_distances.npy'), np.array(hamming_distances))

        # Plot the attack numbers and hamming distances
        plot_attack_results(bits_flips, hamming_distances, args.save_path, top1_accuracies)

        return  # Exit after performing the attack


def perform_oge_attack(attacker, model, model_clean, train_loader, test_loader,
                   criterion, N_iter, logger, attack_idx):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluation mode to handle BatchNorm layers correctly

    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()

    # Use a batch from the training data for the attack
    for _, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        # Ensure no label leaking by using model predictions as target
        with torch.no_grad():
            _, target = model(data).max(1)
        break  # Only need one batch for the attack

    # Initialize lists to store metrics
    bits_flipped = [0]
    top1_accuracies = []
    losses_list = []

    # Evaluate and record the initial (pre-attack) metrics
    val_loss, val_acc = val(model, device, test_loader, criterion, args.time_steps)
    top1_accuracies.append(val_acc)
    losses_list.append(val_loss)

    logger.info(f'[OGE Attack] Initial Validation Accuracy: {val_acc:.2f}%')
    print(f'[OGE Attack] Initial Validation Accuracy: {val_acc:.2f}%')
    logger.info(f'[OGE Attack] k_top = {args.k_top}, Attack sample size = {args.attack_sample_size}')


    end = time.time()
    num_attacks = N_iter  
    hamming_dist = 0

    # Attack iterations
    for i_iter in range(N_iter):
        logger.info('**********************************')
        attack_start_time = time.time()
        attacker.progressive_bit_search(model, data, target)

        # Measure attack time
        attack_time.update(time.time() - attack_start_time)
        end = time.time()

        h_dist = hamming_distance(model, model_clean)
        hamming_dist = h_dist  # Update hamming distance

        # Record the loss
        losses.update(attacker.loss_max, args.attack_sample_size)

        logger.info(
            f'Iteration: [{i_iter+1}/{N_iter}]   '
            f'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})')

        logger.info(f'[OGE Attack] Iteration: {i_iter+1}/{N_iter}\n'
                    f'    Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})\n'
                    f'    Loss before flip: {attacker.loss.item():.4f}\n'
                    f'    Loss after flip:  {attacker.loss_max:.4f}\n'
                    f'    Bit flips so far: {attacker.bit_counter}\n'
                    f'    Hamming distance: {h_dist}')

        # Validate after each attack iteration
        val_loss, val_acc = val(model, device, test_loader, criterion, args.time_steps)

        # Record metrics
        bits_flipped.append(attacker.bit_counter)
        top1_accuracies.append(val_acc)
        losses_list.append(val_loss)

        # Log validation accuracy
        logger.info(f'[OGE Attack] Validation Accuracy: {val_acc:.2f}%')
        print(f'[OGE Attack] After iteration {i_iter+1}: Acc={val_acc:.2f}%')

        # Check if validation accuracy has dropped below 11%
        if val_acc < 10.1:
            num_attacks = i_iter + 1  
            hamming_dist = h_dist     
            break  

        # Measure elapsed time
        iter_time.update(time.time() - end)
        logger.info(f'[OGE Attack] Iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})')
        end = time.time()

    # Plotting results
    plot_dir = os.path.join(args.save_path, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Accuracy vs. bit flips
    plt.figure(figsize=(10, 6))
    plt.plot(bits_flipped, top1_accuracies, marker='o')
    plt.title(f'OGE Attack {attack_idx}: {args.arch}')
    plt.xlabel('Total Bit Flips')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"oge_accuracy_vs_bits_{attack_idx}.png"))
    plt.close()

    # Loss vs. bit flips
    plt.figure(figsize=(10, 6))
    plt.plot(bits_flipped, losses_list, marker='^', color='red')
    plt.title(f'OGE Attack {attack_idx}: {args.arch}')
    plt.xlabel('Total Bit Flips')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"oge_loss_vs_bits_{attack_idx}.png"))
    plt.close()

    logger.info(f'[OGE Attack] Plots saved in {plot_dir}')
    print(f'[OGE Attack] Plots saved in {plot_dir}')

    return num_attacks, hamming_dist, bits_flipped[-1], top1_accuracies[-1]

def val(model, device, test_loader, criterion, T):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size
            if T > 0:
                outputs = model(data)
                output = outputs.mean(0)
            else:
                output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    average_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    return average_loss, accuracy
    
def plot_attack_results(bits_flips, hamming_distances, save_path, top1_accuracies):
    # Plot Number of Attacks vs Runs
    runs = bits_flips
    plt.figure()
    plt.bar(runs, bits_flips, color='blue')
    plt.title('Number of Attacks Required to Lower Accuracy to {}'.format(top1_accuracies))
    plt.xlabel('Run Number')
    plt.ylabel('Number of Bit Flips')
    plt.savefig(os.path.join(save_path, 'attack_numbers.png'))
    plt.close()

    # Plot Hamming Distances vs Runs
    plt.figure()
    plt.bar(runs, hamming_distances, color='green')
    plt.title('Number of Attacks Required to Lower Accuracy to {}'.format(top1_accuracies))
    plt.xlabel('Run Number')
    plt.ylabel('Hamming Distance')
    plt.savefig(os.path.join(save_path, 'hamming_distances.png'))
    plt.close()

if __name__ == "__main__":
    main()
