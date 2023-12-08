import pandas as pd
import numpy as np
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate(preds, gts_path):
    """
    Given the list of all model outputs (logits), and the path to the ground
    truth (val.pkl), calculate the AUC Score of the classified segments.
    Args:
        preds (List[torch.Tensor]): The model ouputs (logits). This is a
            list of all the tensors produced by the model for all samples in
            val.pkl. It should be a list of length 4332 (size of val). All
            tensors in the list should be of size 50 (number of classes).
        gts_path (str): The path to val.pkl
    Returns:
        auc_score (float): A float representing the AUC Score
    """
    #gts = torch.load(gts_path, map_location='cpu') # Ground truth labels, pass path to val.pkl
    gts = pd.read_pickle(gts_path)

    labels = []
    model_outs = []
    for i in range(len(preds)):
        #labels.append(gts[i][2].numpy())                             # A 50D Ground Truth binary vector
        labels.append(np.array(gts.iloc[i]['label']).astype(float))    # A 50D Ground Truth binary vector
        model_outs.append(preds[i]) # A 50D vector that assigns probability to each class

    labels = np.array(labels).astype(float)
    model_outs = np.array(model_outs)

    auc_score = roc_auc_score(y_true=labels, y_score=model_outs)

    if gts_path == '/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/val_labels.pkl':

        print("EVALUATION METRICS for Test Dataset:")
        print("-------------------------------------------------------------")
        print()
        print('AUC Score: {:.4f}'.format(auc_score))
        print()
        print("-------------------------------------------------------------")
    
    else:

        print("EVALUATION METRICS for Validation Dataset:")
        print("-------------------------------------------------------------")
        print()
        print('AUC Score: {:.4f}'.format(auc_score))
        print()
        print("-------------------------------------------------------------")

    

    label_mappings =    {
        0: 'ambient',
        1: 'baroque',
        2: 'bass',
        3: 'beat',
        4: 'cello',
        5: 'chant',
        6: 'choir',
        7: 'classical',
        8: 'country',
        9: 'dance',
        10: 'drum',
        11: 'eastern',
        12: 'electro',
        13: 'fast',
        14: 'female',
        15: 'flute',
        16: 'folk',
        17: 'foreign',
        18: 'guitar',
        19: 'hard',
        20: 'harp',
        21: 'harpsichord',
        22: 'heavy',
        23: 'india',
        24: 'jazz',
        25: 'loud',
        26: 'male',
        27: 'modern',
        28: 'new age',
        29: 'no beat',
        30: 'no piano',
        31: 'no singer',
        32: 'opera',
        33: 'orchestra',
        34: 'piano',
        35: 'pop',
        36: 'quiet',
        37: 'rock',
        38: 'singer',
        39: 'sitar',
        40: 'slow',
        41: 'soft',
        42: 'solo',
        43: 'strange',
        44: 'string',
        45: 'synth',
        46: 'techno',
        47: 'trance',
        48: 'violin',
        49: 'vocal'
    }

    #the code below is used for plotting the best and worst perofrming classes and their examples

    # if gts_path == '/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/val_labels.pkl':

    #     # # Convert model outputs to predicted classes
    #     predicted_classes = np.argmax(model_outs, axis=1)
    #     true_classes = np.argmax(labels, axis=1)

    #     # Compute AUC score for each class
    #     auc_scores = roc_auc_score(y_true=labels, y_score=model_outs, average=None, multi_class='ovr')

    #     # Identify the best and worst classes
    #     best_class_index = np.argmax(auc_scores)
    #     worst_class_index = np.argmin(auc_scores)

    #     # Get file names of examples from the best and worst classes
    #     best_class_examples_indices = np.where(true_classes == best_class_index)[0]
    #     worst_class_examples_indices = np.where(true_classes == worst_class_index)[0]

    #     # Get one good example and two bad examples
    #     good_example_index = best_class_examples_indices[0]
    #     bad_example_index1 = worst_class_examples_indices[0]
    #     bad_example_index2 = worst_class_examples_indices[1] if len(worst_class_examples_indices) > 1 else bad_example_index1

        
    #     print("Good example: ", gts.iloc[good_example_index]['file_path'], " True class: ", label_mappings[true_classes[good_example_index]], " Predicted class: ", label_mappings[predicted_classes[good_example_index]])
    #     print("Bad example 1: ", gts.iloc[bad_example_index1]['file_path'], " True class: ", label_mappings[true_classes[bad_example_index1]], " Predicted class: ", label_mappings[predicted_classes[bad_example_index1]])
    #     print("Bad example 2: ", gts.iloc[bad_example_index2]['file_path'], " True class: ", label_mappings[true_classes[bad_example_index2]], " Predicted class: ", label_mappings[predicted_classes[bad_example_index2]])


    #     # # Calculate the average AUC score
    #     # average_auc_score = np.mean(auc_scores)

    #     # # Identify the classes that perform above and below average
    #     # above_average_classes = np.where(auc_scores > average_auc_score)[0]
    #     # below_average_classes = np.where(auc_scores <= average_auc_score)[0]

    #     # # Identify the best and worst classes
    #     # best_class_index = np.argmax(auc_scores)
    #     # worst_class_index = np.argmin(auc_scores)

    #     # # Prepare data for plotting
    #     # plot_data = pd.DataFrame({
    #     #     'Class': [label_mappings[i] for i in range(len(auc_scores))],
    #     #     'AUC Score': auc_scores,
    #     #     'Performance': ['Best' if i == best_class_index else 'Worst' if i == worst_class_index else 'Above Average' if i in above_average_classes else 'Below Average' for i in range(len(auc_scores))]
    #     # })

    #     # # Plot the data
    #     # plt.figure(figsize=(20, 10))  # Increase the figure size
    #     # sns.barplot(x='Class', y='AUC Score', hue='Performance', data=plot_data, palette=['green', 'red', 'blue', 'orange'])
    #     # plt.title('Class Performance Based on AUC Score')
    #     # plt.xticks(rotation=45)  # Rotate x-axis labels

    #     # # Save the plot to a file
    #     # plt.savefig('/user/home/gv20319/cw', bbox_inches='tight')


    #     # sorted_indices = np.argsort(auc_scores)
    #     # top_3_classes = sorted_indices[-3:]
    #     # bottom_3_classes = sorted_indices[:3]

    #     # # Prepare data for plotting
    #     # plot_data = pd.DataFrame({
    #     #     'Class': [label_mappings[i] for i in np.concatenate((top_3_classes, bottom_3_classes))],
    #     #     'AUC Score': auc_scores[np.concatenate((top_3_classes, bottom_3_classes))],
    #     #     'Performance': ['Top 3' if i in top_3_classes else 'Bottom 3' for i in np.concatenate((top_3_classes, bottom_3_classes))]
    #     # })

    #     # # Plot the data
    #     # plt.figure(figsize=(10, 6))
    #     # sns.barplot(x='Class', y='AUC Score', hue='Performance', data=plot_data, palette=['green', 'red'])
    #     # plt.title('Top 3 and Bottom 3 Class Performance Based on AUC Score')
    #     # plt.xticks(rotation=45)  # Rotate x-axis labels

    #     # # Save the plot to a file
    #     plt.savefig('/user/home/gv20319/cw', bbox_inches='tight')


    return auc_score # Return scores if you wish to save to a file


    
