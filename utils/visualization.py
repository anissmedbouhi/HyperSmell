# scattorplot of 2d visualizations
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from scipy.special import softmax
from scipy.linalg import qr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
import numpy as np
import scipy
import torch
import os
import matplotlib
from distances import poincare_distance, project_to_poincare_disk_for_viz, cross_distance_matrix_for_viz, hyperbolic_kde_for_viz, estimate_hyperbolic_bandwidth_for_viz, hyperbolic_area_element_for_viz
plt.rcParams["font.size"] = 45
plt.rcParams['agg.path.chunksize'] = 10000

from constants import *
from constants import sagar_tasks
from methods import exp_map, log_map

from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_losses(i, args=None, save=False, losses=None,
                losses_pos=None, losses_neg=None):
    color_map = 'plasma'
    # latent_embeddings_norm = torch.norm(latent_embeddings, dim=-1)

    fig, ax = plt.subplots(2, 1, figsize=(30, 90), sharey=False)
    ax[0].plot(np.arange(len(losses)), losses, label='total')

    # ax[0].plot(np.arange(len(losses_pos)), losses_pos, label='positive')
    # ax[1].plot(np.arange(len(losses_neg)), losses_neg, label='negative')

    fig.subplots_adjust(hspace=0.3)
    # showing the legend
    ax[0].legend()
    ax[1].legend()
    plt.title(
        f'dataset_name={args.dataset_name}, lr = {args.lr}, latent_dim = {args.latent_dim}, epochs = {args.num_epochs}, \n batch_size = {args.batch_size}, normalize = {args.normalize}, distance_method = {args.distance_method},\n  model = {args.model_name}, optimizer = {args.optimizer_type}, latent_dist_fun = {args.latent_dist_fun} \n temperature = {args.temperature}, depth = {args.depth}')
    f_path = f"losses/{args.dataset_name}/{args.distance_method}/{args.latent_dist_fun}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.epsilon}/{args.batch_size}"
    # create a folder if it does not exist
    if save:
        if not os.path.exists(
                f_path):
            os.makedirs(
                f_path)
        plt.savefig(
            f_path+f"{args.random_string}_{i}_{args.seed}_{args.dataset_name}_losses.png")
        plt.close()
    else:
        plt.show()
        

def scatterplot_2d(i, latent_embeddings, input_embeddings, CIDs, labels, subjects=None, color_by='entropy', label_dim = None, shape_by='none', args=None, save=False, plot_edges=False,
                    hyperbolic_boundary=True, saving_path='figs2'):
    color_map = 'plasma'
    #
    # data_dist_matrix = scipy.spatial.distance.cdist(input_embeddings, input_embeddings, metric='hamming') * \
    #                    input_embeddings.shape[-1]

    # f_path = f"figs2/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/"
    if args != None:
        f_path = f"{saving_path}/{args.dataset_name}/{args.distance_method}/{args.latent_dist_fun}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.epsilon}/{args.batch_size}"
    else:
        f_path = f"{saving_path}/"
    fig, ax = plt.subplots(1, 1, figsize=(30, 30), sharey=False)
    markers = ["o", "s", "D", "P", "X", "v", ">", "<", "^", "d", "p", "*", "h", "H", "+", "x", "|", "_"]
    if shape_by=='subject':
        groups = subjects.unique()
        selected_groups = subjects
    elif shape_by=='descriptor':
        groups = np.arange(labels.shape[1])
        selected_groups = [np.argmax(label[2:]) + 2 for label in labels]
    elif shape_by=='none':
        groups = None
        markers = ["o"]*subjects.max()

    else:
        raise ValueError('shape_by not recognized')

    if color_by=='input_norm':
        c = torch.norm(input_embeddings, dim=-1)
    elif color_by=='entropy':
        entropy = softmax(input_embeddings, -1)
        c = -(entropy * np.log(entropy)).sum(-1)
    elif color_by=='cid':
        unique_vals = np.unique(CIDs)
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        c = np.vectorize(mapping.get)(CIDs)
    elif color_by=='distance':
        #todo
        c = torch.norm(latent_embeddings, dim=-1)
    elif color_by=='none':
        c =np.ones(latent_embeddings.shape[0])
    elif color_by=='color':
        c = input_embeddings
    elif color_by == 'label_dim':
        if torch.is_tensor(labels):
            c = labels[:, label_dim].detach().cpu().numpy()
        else:
            c = labels[:, label_dim]

    else:
        raise ValueError('color_by not recognized')


    #radius = np.sqrt(np.sum(np.square(latent_embeddings), axis=1))
    radius = poincare_distance(torch.from_numpy(latent_embeddings), torch.zeros((1, 2)))
    corr = np.corrcoef(radius, c)
    # print('corr', corr)

    #Plotting the correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(radius, c, color='#6a0dad', alpha=0.7, s=50)  # Scatter plot # edgecolor='k'
    
    # Calculate the line of best fit
    slope, intercept = np.polyfit(radius, c, 1)  # Linear regression
    line = slope * radius + intercept  # Calculate the y values for the line
    
    # Plot the regression line
    plt.plot(radius, line, color='#ffbf00', linewidth=2)  # Add the line to the plot
    
    plt.xlabel('Hyperbolic radius', fontsize=30)
    plt.ylabel('Entropy', fontsize=30)
    plt.title('Correlation between Entropy and Radius', fontsize=16)  # Added title
    plt.grid(True)
    #plt.legend()  # Show legend
    plt.tight_layout()  # Adjust layout for better spacing

    plt.xticks([])
    plt.yticks([])
    
    #plt.axis('off')
    
    if save:
        # Save the figure with 'corr' in the filename
        plt.savefig(f'{saving_path}/{i}_corr.png')  # Changed filename to include 'corr'
        plt.savefig(f'{saving_path}/{i}_corr.pdf')
    plt.close()  # Close the plot to avoid display if running in a script




    if plot_edges:
        colors = sns.color_palette("plasma", data_dist_matrix.shape[0])
        for i in range(data_dist_matrix.shape[0]):
            for j in range(i + 1, data_dist_matrix.shape[1]):
                if data_dist_matrix[i, j] <= 1.01:
                    ax.plot([latent_embeddings[i, 0], latent_embeddings[j, 0]],
                               [latent_embeddings[i, 1], latent_embeddings[j, 1]], color=colors[i], linewidth=5.)

    ax.axis('equal')
    ax.axis('off')
    if (args != None and args.normalize == True) or (args == None and hyperbolic_boundary == True):
        ax.set_ylim(-1.09, 1.09)
        ax.set_xlim(-1.09, 1.09)


    ## ax[0].scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], c=np.linalg.norm(input_embeddings, axis=-1), cmap=color_map, s=300, zorder=10   )
    
    # for subject in subjects.unique():
    #     if subject==3:
    #         idx = subjects == subject
    #         ax.scatter(latent_embeddings[idx, 0], latent_embeddings[idx, 1], c=c[idx], cmap="plasma" , s=300, zorder=10,marker=markers[subject-1])

    if groups is None:
        ax.scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], c=c, cmap=color_map, s=300, zorder=10)
    else:
        for group in groups:
            idx = selected_groups == group
            ax.scatter(latent_embeddings[idx, 0], latent_embeddings[idx, 1], c=c[idx], cmap=color_map , s=300, zorder=10,marker=markers[group-1],vmin = c.min(), vmax = c.max())



    if hyperbolic_boundary:
        circle = plt.Circle((0, 0), 1., color='gray', fill=False, linewidth=10)
        ax.add_patch(circle)


    # create a folder if it does not exist
    if save:
        if not os.path.exists(
                f_path):
            os.makedirs(
                f_path)
        if args!=None:
            plt.savefig(f_path+f"{args.random_string}_{i}_{args.seed}_{args.dataset_name}_embeddings.png")
        else:
            plt.savefig(f'{saving_path}/{i}_embedding.pdf')
            plt.savefig(f'{saving_path}/{i}_embedding.png')
        plt.close()
    else:
        plt.show()



def scatterplot_2d_new(
    i,
    latent_embeddings,
    input_embeddings,
    CIDs,
    labels,
    subjects=None,
    color_by='entropy',
    label_dim=None,
    show_label_gradient=False,
    shape_by='none',
    args=None,
    save=False,
    plot_edges=False,
    hyperbolic_boundary=True,
    saving_path='figs2'
):
    color_map = 'plasma'
    corr_scalar = np.nan

    if args is not None:
        f_path = f"{saving_path}/{args.dataset_name}/{args.distance_method}/{args.latent_dist_fun}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.epsilon}/{args.batch_size}"
    else:
        f_path = f"{saving_path}/"

    fig, ax = plt.subplots(1, 1, figsize=(30, 30), sharey=False)

    markers = ["o", "s", "D", "P", "X", "v", ">", "<", "^", "d", "p", "*", "h", "H", "+", "x", "|", "_"]

    if shape_by == 'subject':
        groups = subjects.unique()
        selected_groups = subjects

    elif shape_by == 'descriptor':
        groups = np.arange(labels.shape[1])
        selected_groups = [np.argmax(label[2:]) + 2 for label in labels]

    elif shape_by == 'none':
        groups = None
        markers = ["o"] * 20

    else:
        raise ValueError('shape_by not recognized')

    # ------------------------------------------------------------
    # Choose coloring variable
    # ------------------------------------------------------------

    if color_by == 'input_norm':
        c = torch.norm(input_embeddings, dim=-1)
        if torch.is_tensor(c):
            c = c.detach().cpu().numpy()

    elif color_by == 'entropy':
        entropy = softmax(input_embeddings, -1)
        c = -(entropy * np.log(entropy + 1e-12)).sum(-1)
        if torch.is_tensor(c):
            c = c.detach().cpu().numpy()

    elif color_by == 'cid':
        unique_vals = np.unique(CIDs)
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        c = np.vectorize(mapping.get)(CIDs)

    elif color_by == 'distance':
        if torch.is_tensor(latent_embeddings):
            c = torch.norm(latent_embeddings, dim=-1).detach().cpu().numpy()
        else:
            c = np.linalg.norm(latent_embeddings, axis=-1)

    elif color_by == 'none':
        c = np.ones(latent_embeddings.shape[0])

    elif color_by == 'color':
        c = input_embeddings

    elif color_by == 'label_dim':
        if torch.is_tensor(labels):
            c = labels[:, label_dim].detach().cpu().numpy()
        else:
            c = labels[:, label_dim]

    else:
        raise ValueError('color_by not recognized')

    # Make sure latent_embeddings is numpy
    if torch.is_tensor(latent_embeddings):
        latent_embeddings_np = latent_embeddings.detach().cpu().numpy()
    else:
        latent_embeddings_np = latent_embeddings

    # ------------------------------------------------------------
    # Optional correlation plot: radius versus coloring variable
    # ------------------------------------------------------------

    try:
        radius = poincare_distance(
            torch.tensor(latent_embeddings_np, dtype=torch.float32),
            torch.zeros((1, 2), dtype=torch.float32)
        )

        if torch.is_tensor(radius):
            radius = radius.detach().cpu().numpy()

        if np.asarray(c).ndim == 1:
            c_np_corr = np.asarray(c)

            valid_corr = np.isfinite(radius) & np.isfinite(c_np_corr)

            if valid_corr.sum() > 2:
                corr_scalar = np.corrcoef(radius[valid_corr], c_np_corr[valid_corr])[0, 1]
            else:
                corr_scalar = np.nan

            print(
                f"label_dim={label_dim} | "
                f"corr(radius, label)={corr_scalar:.4f}"
            )

            plt.figure(figsize=(10, 6))
            plt.scatter(radius, c, color='#6a0dad', alpha=0.7, s=50)

            slope, intercept = np.polyfit(radius, c, 1)
            line = slope * radius + intercept

            plt.plot(radius, line, color='#ffbf00', linewidth=2)

            plt.xlabel('Hyperbolic radius', fontsize=30)

            if color_by == 'label_dim':
                plt.ylabel(f'Label dim {label_dim}', fontsize=30)
                plt.title(f'Correlation between label dim {label_dim} and radius', fontsize=16)
            else:
                plt.ylabel(color_by, fontsize=30)
                plt.title(f'Correlation between {color_by} and radius', fontsize=16)

            plt.grid(True)
            plt.tight_layout()
            plt.xticks([])
            plt.yticks([])

            if save:
                plt.savefig(f'{saving_path}/{i}_corr.png')
                

            plt.close()

    except Exception as e:
        print("Skipping radius-correlation plot:", e)

    # ------------------------------------------------------------
    # Optional edges
    # ------------------------------------------------------------

    if plot_edges:
        colors = sns.color_palette("plasma", data_dist_matrix.shape[0])
        for k in range(data_dist_matrix.shape[0]):
            for j in range(k + 1, data_dist_matrix.shape[1]):
                if data_dist_matrix[k, j] <= 1.01:
                    ax.plot(
                        [latent_embeddings_np[k, 0], latent_embeddings_np[j, 0]],
                        [latent_embeddings_np[k, 1], latent_embeddings_np[j, 1]],
                        color=colors[k],
                        linewidth=5.
                    )

    ax.axis('equal')
    ax.axis('off')

    if (args is not None and args.normalize == True) or (args is None and hyperbolic_boundary == True):
        ax.set_ylim(-1.09, 1.09)
        ax.set_xlim(-1.09, 1.09)

    # ------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------

    if groups is None:
        sc = ax.scatter(
            latent_embeddings_np[:, 0],
            latent_embeddings_np[:, 1],
            c=c,
            cmap=color_map,
            s=300,
            zorder=10
        )
    else:
        sc = None
        for group in groups:
            idx = selected_groups == group
            sc = ax.scatter(
                latent_embeddings_np[idx, 0],
                latent_embeddings_np[idx, 1],
                c=c[idx],
                cmap=color_map,
                s=300,
                zorder=10,
                marker=markers[group - 1],
                vmin=np.min(c),
                vmax=np.max(c)
            )

    # Colorbar only when coloring by a continuous label dimension
    if color_by == 'label_dim':
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{sagar_tasks[label_dim]}", fontsize=90)
        cbar.ax.tick_params(labelsize=50)

    # ------------------------------------------------------------
    # Direction of maximal increase in the Poincaré disk
    # ------------------------------------------------------------

    if show_label_gradient and color_by == 'label_dim':
        X_disk = latent_embeddings_np[:, :2]
        c_np = np.asarray(c)

        # Keep only valid points inside the disk
        r = np.linalg.norm(X_disk, axis=1)
        valid = np.isfinite(c_np) & (r < 1.0)

        X_disk_valid = X_disk[valid]
        c_valid = c_np[valid]

        if len(c_valid) > 2:
            # 1. Map Poincaré points to tangent space at origin        
            X_disk_torch = torch.tensor(X_disk_valid, dtype=torch.float32)
            Z_torch = log_map(X_disk_torch)
            Z = Z_torch.detach().cpu().numpy()

            # 2. Fit label value ≈ beta_x * z_x + beta_y * z_y + intercept
            A = np.column_stack([Z[:, 0], Z[:, 1], np.ones(Z.shape[0])])
            beta_x, beta_y, intercept = np.linalg.lstsq(A, c_valid, rcond=None)[0]

            direction = np.array([beta_x, beta_y])
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-8:
                direction = direction / direction_norm
                
                angle_rad = np.arctan2(direction[1], direction[0])
                angle_deg = np.degrees(angle_rad)

                # Optional: convert from [-180, 180] to [0, 360]
                if angle_deg < 0:
                    angle_deg += 360

                # R² score
                pred = A @ np.array([beta_x, beta_y, intercept])
                ss_res = np.sum((c_valid - pred) ** 2)
                ss_tot = np.sum((c_valid - c_valid.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                
                print(
                    f"label_dim={label_dim} | "
                    f"angle={angle_deg:.2f} degrees | "
                    f"R2_direction={r2:.4f}"
                )

                # 3. Draw arrow as a geodesic diameter in the disk
                direction_torch = torch.tensor(direction[None, :], dtype=torch.float32)
                tangent_length = 3.0
                start = exp_map(-tangent_length * direction_torch)[0].detach().cpu().numpy()
                end = exp_map(tangent_length * direction_torch)[0].detach().cpu().numpy()

                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops=dict(
                        arrowstyle="->",
                        linewidth=8,
                        color="black"
                    ),
                    zorder=50
                )

#                 ax.text(
#                     end[0],
#                     end[1],
#                     f"dim {label_dim} increases\nR²={r2:.2f}",
#                     fontsize=28,
#                     color="black",
#                     ha="center",
#                     va="center",
#                     zorder=51
#                 )

    # ------------------------------------------------------------
    # Poincaré boundary
    # ------------------------------------------------------------

    if hyperbolic_boundary:
        circle = plt.Circle((0, 0), 1., color='gray', fill=False, linewidth=10)
        ax.add_patch(circle)

    # ------------------------------------------------------------
    # Save or show
    # ------------------------------------------------------------

    if save:
        if not os.path.exists(f_path):
            os.makedirs(f_path)

        if args is not None:
            save_file = f_path + f"/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_embeddings_{label_dim}.pdf"
        else:
            save_file = f_path + f"/{i}_embeddings_{label_dim}.pdf"

        plt.savefig(save_file, bbox_inches='tight', pad_inches=0.15)
        #plt.savefig(saving_path, bbox_inches='tight', pad_inches=0.15)
        #plt.savefig(f'{saving_path}')
        plt.close()
    else:
        plt.show()        
        
        
        
        
        
        
        
        

def save_embeddings(i, args, latent_embeddings, losses=[], losses_pos=[], losses_neg=[]):
    if not os.path.exists(
            f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/"):
        os.makedirs(
            f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/")

    np.save(
        f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_embeddings.npy",
        latent_embeddings)

    np.save(
        f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_losses.npy",
        losses)
    np.save(
        f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_lossespos.npy",
        losses_pos)
    np.save(
        f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_lossesneg.npy",
        losses_neg)



def pom_frame(pom_embeds, y, required_desc, title, size1, size2, size3, reduction_method=None, perplexity=None):
    sns.set_style("ticks")
    sns.despine()
    plt.rcParams["font.size"] = 35

    # pom_embeds = model.predict_embedding(dataset)
    # y_preds = model.predict(dataset)
    # required_desc = list(dataset.tasks)
    type1 = {'floral': '#F3F1F7', 'subs': {'muguet': '#FAD7E6', 'lavender': '#8883BE', 'jasmin': '#BD81B7'}}
    type2 = {'meaty': '#F5EBE8', 'subs': {'savory': '#FBB360', 'beefy': '#7B382A', 'roasted': '#F7A69E'}}
    type3 = {'ethereal': '#F2F6EC', 'subs': {'cognac': '#BCE2D2', 'fermented': '#79944F', 'alcoholic': '#C2DA8F'}}

    # Assuming you have your features in the 'features' array
    if reduction_method == 'PCA':
        pca = PCA(n_components=2,
                  iterated_power=10)  # You can choose the number of components you want (e.g., 2 for 2D visualization)
        reduced_features = pca.fit_transform(pom_embeds)  # try different variations
        variance_explained = pca.explained_variance_ratio_
        variance_pc1 = variance_explained[0]
        variance_pc2 = variance_explained[1]
        print(variance_pc1, variance_pc2)

    elif reduction_method == 'tsne':
        tsne = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,

        )
        reduced_features = tsne.fit_transform(pom_embeds)
    elif reduction_method == 'UMAP':
        reduced_features = umap.UMAP(n_components=2, n_neighbors=perplexity, min_dist=0.0,
                                     metric='euclidean').fit_transform(X=pom_embeds)
    elif reduction_method is None:
        reduced_features = pom_embeds
        reduction_method = 'None'

    else:
        raise ValueError('Invalid reduction method')

    # if is_preds:
    #     y = np.where(y_preds>threshold, 1.0, 0.0) # try quartile range (or rank)
    # else:
    #     y = dataset.y

    # Generate grid points to evaluate the KDE on (try kernel convolution)
    x_grid, y_grid = np.meshgrid(np.linspace(reduced_features[:, 0].min(), reduced_features[:, 0].max(), 500),
                                 np.linspace(reduced_features[:, 1].min(), reduced_features[:, 1].max(), 500))
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
    print(reduced_features[:, 0].min(), reduced_features[:, 0].max(), reduced_features[:, 1].min(),
          reduced_features[:, 1].max())

    def get_kde_values(label):
        plot_idx = required_desc.index(label)
        # print(y[:, plot_idx])
        label_indices = np.where(y[:, plot_idx] == 1)[0]
        kde_label = gaussian_kde(reduced_features[label_indices].T)
        kde_values_label = kde_label(grid_points)
        kde_values_label = kde_values_label.reshape(x_grid.shape)
        return kde_values_label

    def plot_contours(type_dictionary, bbox_to_anchor):
        main_label = list(type_dictionary.keys())[0]
        plt.contourf(x_grid, y_grid, get_kde_values(main_label), levels=1,
                     colors=['#00000000', type_dictionary[main_label], type_dictionary[main_label]])
        axes = plt.gca()  # Getting the current axis

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        legend_elements = []
        for label, color in type_dictionary['subs'].items():
            plt.contour(x_grid, y_grid, get_kde_values(label), levels=1, colors=color, linewidths=2)
            legend_elements.append(Patch(facecolor=color, label=label))
        legend = plt.legend(handles=legend_elements, title=main_label, bbox_to_anchor=bbox_to_anchor, prop={'size': 30})
        legend.get_frame().set_facecolor(type_dictionary[main_label])
        plt.gca().add_artist(legend)

    fig = plt.figure(figsize=(15, 15), dpi=700)
    # ax.spines[['right', 'top']].set_visible(False)
    # plt.title('KDE Density Estimation with Contours in Reduced Space')
    # plt.xlabel(f'Principal Component 1 ({round(variance_pc1*100, ndigits=2)}%)')
    # plt.ylabel(f'Principal Component 2 ({round(variance_pc2*100, ndigits=2)}%)')
    plt.xlabel('Principal Component 1', fontsize=35)
    plt.ylabel('Principal Component 2', fontsize=35)
    plot_contours(type_dictionary=type1, bbox_to_anchor=size1)
    plot_contours(type_dictionary=type2, bbox_to_anchor=size2)
    plot_contours(type_dictionary=type3, bbox_to_anchor=size3)
    # plt.colorbar(label='Density')
    # plt.show()
    # png_file = os.path.join(dir, 'pom_frame.png')
    # plt.savefig(png_file)
    plt.savefig("figs/islands/realign_islands_" + title + "_" + reduction_method + "_" + str(perplexity) + ".svg")
    plt.savefig("figs/islands/realign_islands_" + title + "_" + reduction_method + "_" + str(perplexity) + ".pdf")
    plt.savefig("figs/islands/realign_islands_" + title + "_" + reduction_method + "_" + str(perplexity) + ".jpg")

    # plt.show()
    # plt.close()
    
    
    
    
    
# islands visualization handling both Euclidean and Hyperbolic cases

# def viz_frame_probamass(
#     i,
#     embeds,
#     y,
#     required_desc,
#     title,
#     categories,
#     reduction_method=None,
#     perplexity=None,
#     target_mass_main=0.8,
#     target_mass_sub=0.5,
#     islands_alpha=0.65,
#     space='euclidean',
#     bandwidth_scale=1.0,
#     forced_bandwidth=None,
#     grid_res=500,
#     euclidean_grid_padding=0.05,
#     show_embeddings=True,
#     embedding_dot_size=5,
#     embedding_dot_alpha=0.4,
#     embedding_dot_color="grey",
#     embedding_dot_zorder=1
# ):

#     if space not in ['euclidean', 'poincare']:
#         raise ValueError("space must be either 'euclidean' or 'poincare'")

#     sns.set_style("ticks")
#     sns.despine()
#     plt.rcParams["font.size"] = 35

# #     if type1 == None:
# #         type1 = {
# #             'floral': '#F3F1F7',
# #             'subs': {
# #                 'muguet': '#FAD7E6',
# #                 'lavender': '#8883BE',
# #                 'jasmin': '#BD81B7'
# #             }
# #         }

# #     if type2 == None:
# #         type2 = {
# #             'meaty': '#F5EBE8',
# #             'subs': {
# #                 'savory': '#FBB360',
# #                 'beefy': '#7B382A',
# #                 'roasted': '#F7A69E'
# #             }
# #         }

# #     if type3 == None:
# #         type3 = {
# #             'ethereal': '#F2F6EC',
# #             'subs': {
# #                 'cognac': '#BCE2D2',
# #                 'fermented': '#79944F',
# #                 'alcoholic': '#C2DA8F'
# #             }
# #         }

#     # ------------------------------------------------------------
#     # 1. Prepare / reduce features
#     # ------------------------------------------------------------

#     if space == 'poincare':
#         # In Poincaré mode, assume embeds are already 2D Poincaré coordinates.
#         # We do not apply PCA/t-SNE/UMAP because those are Euclidean reductions.
#         reduced_features = embeds

#         if torch.is_tensor(reduced_features):
#             reduced_features = reduced_features.detach().cpu().numpy()

#         reduced_features_torch = torch.tensor(reduced_features, dtype=torch.float32)
#         reduced_features_torch = project_to_poincare_disk_for_viz(reduced_features_torch)
#         reduced_features = reduced_features_torch.cpu().numpy()

#         reduction_label = 'Poincare'

#     else:
#         # Euclidean mode: use your original reduction logic.
#         if reduction_method == 'PCA':
#             pca = PCA(n_components=2, iterated_power=10)
#             reduced_features = pca.fit_transform(embeds)

#             variance_explained = pca.explained_variance_ratio_
#             variance_pc1 = variance_explained[0]
#             variance_pc2 = variance_explained[1]
#             print(variance_pc1, variance_pc2)

#         elif reduction_method == 'tsne':
#             tsne = manifold.TSNE(
#                 n_components=2,
#                 init="random",
#                 random_state=0,
#                 perplexity=perplexity,
#             )
#             reduced_features = tsne.fit_transform(embeds)

#         elif reduction_method == 'UMAP':
#             reduced_features = umap.UMAP(
#                 n_components=2,
#                 n_neighbors=perplexity,
#                 min_dist=0.0,
#                 metric='euclidean'
#             ).fit_transform(X=embeds)

#         elif reduction_method is None:
#             reduced_features = embeds
#             reduction_method = 'None'

#         else:
#             raise ValueError('Invalid reduction method')

#         if torch.is_tensor(reduced_features):
#             reduced_features = reduced_features.detach().cpu().numpy()

#         reduction_label = reduction_method

#     print(
#         reduced_features[:, 0].min(),
#         reduced_features[:, 0].max(),
#         reduced_features[:, 1].min(),
#         reduced_features[:, 1].max()
#     )

#     # ------------------------------------------------------------
#     # 2. Build grid depending on geometry
#     # ------------------------------------------------------------

#     if space == 'poincare':
#         x_grid, y_grid = np.meshgrid(
#             np.linspace(-0.999, 0.999, grid_res),
#             np.linspace(-0.999, 0.999, grid_res)
#         )

#         disk_mask = x_grid ** 2 + y_grid ** 2 < 0.999 ** 2

#         grid_points = np.column_stack([
#             x_grid.ravel(),
#             y_grid.ravel()
#         ])

#         valid_grid_points = grid_points[disk_mask.ravel()]

#     else:
#         x_min, x_max = reduced_features[:, 0].min(), reduced_features[:, 0].max()
#         y_min, y_max = reduced_features[:, 1].min(), reduced_features[:, 1].max()

#         x_pad = euclidean_grid_padding * (x_max - x_min)
#         y_pad = euclidean_grid_padding * (y_max - y_min)

#         if x_pad == 0:
#             x_pad = 1e-3
#         if y_pad == 0:
#             y_pad = 1e-3

#         x_grid, y_grid = np.meshgrid(
#             np.linspace(x_min - x_pad, x_max + x_pad, grid_res),
#             np.linspace(y_min - y_pad, y_max + y_pad, grid_res)
#         )

#         grid_points = np.vstack([
#             x_grid.ravel(),
#             y_grid.ravel()
#         ])

#         disk_mask = None
#         valid_grid_points = None

#     # ------------------------------------------------------------
#     # 3. KDE function depending on geometry
#     # ------------------------------------------------------------

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     bandwidth_cache = {}

#     def get_kde_values(label):
#         plot_idx = required_desc.index(label)
#         label_indices = np.where(y[:, plot_idx] == 1)[0]

#         label_points = reduced_features[label_indices]

#         if len(label_points) < 2:
#             return np.full(x_grid.shape, np.nan)

#         if space == 'euclidean':
#             kde_label = gaussian_kde(label_points.T)
#             kde_values_label = kde_label(grid_points)
#             kde_values_label = kde_values_label.reshape(x_grid.shape)
#             return kde_values_label

#         else:
#             label_points_torch = torch.tensor(
#                 label_points,
#                 dtype=torch.float32,
#                 device=device
#             )

#             label_points_torch = project_to_poincare_disk_for_viz(label_points_torch)

#             if label not in bandwidth_cache:
#                 base_bandwidth = estimate_hyperbolic_bandwidth_for_viz(
#                     label_points_torch,
#                     distance_func=poincare_distance,
#                     device=device
#                 )

#                 bandwidth_cache[label] = bandwidth_scale * base_bandwidth

#                 if forced_bandwidth is not None:
#                     bandwidth_cache[label] = forced_bandwidth

#             bandwidth = bandwidth_cache[label]

#             kde_valid = hyperbolic_kde_for_viz(
#                 valid_grid_points,
#                 label_points_torch,
#                 bandwidth=bandwidth,
#                 distance_func=poincare_distance,
#                 device=device
#             )

#             kde_values_label = np.full(x_grid.shape, np.nan)
#             kde_values_label.ravel()[disk_mask.ravel()] = kde_valid

#             return kde_values_label

#     # ------------------------------------------------------------
#     # 4. Probability-mass contour level depending on geometry
#     # ------------------------------------------------------------

#     def get_probability_contour_level(kde_values, x_grid, y_grid, target_mass=0.5):
#         dens = kde_values.ravel()

#         valid = np.isfinite(dens)
#         dens = dens[valid]

#         if len(dens) == 0:
#             return np.nan

#         dx = x_grid[0, 1] - x_grid[0, 0]
#         dy = y_grid[1, 0] - y_grid[0, 0]

#         if space == 'euclidean':
#             cell_area = dx * dy
#             cell_mass = np.full_like(dens, cell_area, dtype=np.float64)

#         else:
#             area_element = hyperbolic_area_element_for_viz(
#                 x_grid,
#                 y_grid
#             ).ravel()[valid]

#             cell_mass = area_element * dx * dy

#         order = np.argsort(dens)[::-1]

#         dens_sorted = dens[order]
#         mass_sorted = dens_sorted * cell_mass[order]

#         cum_mass = np.cumsum(mass_sorted)

#         if cum_mass[-1] <= 0:
#             return np.nan

#         cum_mass = cum_mass / cum_mass[-1]

#         idx = np.searchsorted(cum_mass, target_mass)

#         return dens_sorted[min(idx, len(dens_sorted) - 1)]

#     # ------------------------------------------------------------
#     # 5. Plot contours and embeddings
#     # ------------------------------------------------------------

#     def plot_contours(
#         type_dictionary,
#         bbox_to_anchor,
#         target_mass_main=target_mass_main,
#         target_mass_sub=target_mass_sub,
#         islands_alpha=islands_alpha
#     ):
#         main_label = list(type_dictionary.keys())[0]

#         kde_main = np.ma.masked_invalid(get_kde_values(main_label))

#         main_level = get_probability_contour_level(
#             kde_main.filled(np.nan),
#             x_grid,
#             y_grid,
#             target_mass=target_mass_main
#         )

#         if np.isfinite(main_level) and kde_main.max() > main_level:
#             plt.contourf(
#                 x_grid,
#                 y_grid,
#                 kde_main,
#                 levels=[main_level, kde_main.max()],
#                 colors=[type_dictionary[main_label]],
#                 alpha=islands_alpha
#             )

#         axes = plt.gca()
#         axes.spines['top'].set_visible(False)
#         axes.spines['right'].set_visible(False)

# #         legend_elements = []

# #         for label, color in type_dictionary['subs'].items():
# #             kde_sub = np.ma.masked_invalid(get_kde_values(label))

# #             sub_level = get_probability_contour_level(
# #                 kde_sub.filled(np.nan),
# #                 x_grid,
# #                 y_grid,
# #                 target_mass=target_mass_sub
# #             )

# #             if np.isfinite(sub_level) and kde_sub.max() > sub_level:
# #                 plt.contour(
# #                     x_grid,
# #                     y_grid,
# #                     kde_sub,
# #                     levels=[sub_level],
# #                     colors=color,
# #                     linewidths=2
# #                 )

# #             legend_elements.append(Patch(facecolor=color, label=label))

# #         legend = plt.legend(
# #             handles=legend_elements,
# #             title=main_label,
# #             bbox_to_anchor=bbox_to_anchor,
# #             prop={'size': 30}
# #         )

# #         legend.get_frame().set_facecolor(type_dictionary[main_label])
# #         plt.gca().add_artist(legend)

#         legend_elements = []

#         subs = type_dictionary.get('subs', {})

#         for label, color in subs.items():
#             kde_sub = np.ma.masked_invalid(get_kde_values(label))

#             sub_level = get_probability_contour_level(
#                 kde_sub.filled(np.nan),
#                 x_grid,
#                 y_grid,
#                 target_mass=target_mass_sub
#             )

#             if np.isfinite(sub_level) and kde_sub.max() > sub_level:
#                 plt.contour(
#                     x_grid,
#                     y_grid,
#                     kde_sub,
#                     levels=[sub_level],
#                     colors=color,
#                     linewidths=2
#                 )

#             legend_elements.append(Patch(facecolor=color, label=label))

#         # If there are no subs, show the main category itself in the legend
#         if len(legend_elements) == 0:
#             legend_elements.append(
#                 Patch(facecolor=type_dictionary[main_label], label=main_label)
#             )
#             legend_title = None
#         else:
#             legend_title = main_label

#         legend = plt.legend(
#             handles=legend_elements,
#             title=legend_title,
#             bbox_to_anchor=bbox_to_anchor,
#             prop={'size': 30}
#         )

#         legend.get_frame().set_facecolor(type_dictionary[main_label])
#         plt.gca().add_artist(legend)
        
        
#     def plot_embeddings():
#         if not show_embeddings:
#             return

#         points_to_plot = reduced_features

#         if space == 'poincare':
#             r2 = points_to_plot[:, 0] ** 2 + points_to_plot[:, 1] ** 2
#             points_to_plot = points_to_plot[r2 < 1.0]

#         plt.scatter(
#             points_to_plot[:, 0],
#             points_to_plot[:, 1],
#             s=embedding_dot_size,
#             c=embedding_dot_color,
#             alpha=embedding_dot_alpha,
#             linewidths=0,
#             zorder=embedding_dot_zorder,
#             rasterized=True
#         )

#     # ------------------------------------------------------------
#     # 6. Build final figure
#     # ------------------------------------------------------------

#     fig = plt.figure(figsize=(15, 15), dpi=700)

#     for category_dict in categories:
#         plot_contours(type_dictionary=category_dict, bbox_to_anchor=category_dict['size'])
# #     plot_contours(type_dictionary=type2, bbox_to_anchor=size2)
# #     plot_contours(type_dictionary=type3, bbox_to_anchor=size3)
#     plot_embeddings()

#     ax = plt.gca()
#     ax.axis("off")

#     if space == 'poincare':
#         circle = plt.Circle(
#             (0, 0),
#             1.0,
#             fill=False,
#             linewidth=2,
#             color="black"
#         )

#         ax.add_patch(circle)
#         ax.set_aspect("equal")
#         ax.set_xlim(-1.02, 1.02)
#         ax.set_ylim(-1.02, 1.02)

# #         plt.xlabel('Poincaré disk x', fontsize=35)
# #         plt.ylabel('Poincaré disk y', fontsize=35)

#     else:
#         ax.set_aspect("auto")

# #         if reduction_method == 'PCA':
# #             plt.xlabel('Principal Component 1', fontsize=35)
# #             plt.ylabel('Principal Component 2', fontsize=35)
# #         elif reduction_method == 'tsne':
# #             plt.xlabel('t-SNE 1', fontsize=35)
# #             plt.ylabel('t-SNE 2', fontsize=35)
# #         elif reduction_method == 'UMAP':
# #             plt.xlabel('UMAP 1', fontsize=35)
# #             plt.ylabel('UMAP 2', fontsize=35)
# #         else:
# #             plt.xlabel('Dimension 1', fontsize=35)
# #             plt.ylabel('Dimension 2', fontsize=35)

#     save_path = (
#         "figs/islands/GSLF_islands_"
#         + title
#         + "_"
#         + str(space)
#         + "_"
#         + str(reduction_label)
#         + "_"
#         + str(perplexity)
#         + "_"
#         + str(i)
#         + ".pdf"
#     )

#     #plt.savefig(save_path)
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
#     plt.show()
#     plt.close()


def viz_frame_probamass(
    i,
    embeds,
    y,
    required_desc,
    title,
    categories,
    reduction_method=None,
    perplexity=None,
    target_mass_main=0.8,
    target_mass_sub=0.5,
    islands_alpha=0.65,
    space='euclidean',
    bandwidth_scale=1.0,
    forced_bandwidth=None,
    grid_res=500,
    euclidean_grid_padding=0.05,
    show_embeddings=True,
    embedding_dot_size=10,
    embedding_dot_alpha=0.4,
    embedding_dot_color="grey",
    embedding_dot_zorder=1
):

    if space not in ['euclidean', 'poincare']:
        raise ValueError("space must be either 'euclidean' or 'poincare'")

    sns.set_style("ticks")
    sns.despine()
    plt.rcParams["font.size"] = 35

#     if type1 == None:
#         type1 = {
#             'floral': '#F3F1F7',
#             'subs': {
#                 'muguet': '#FAD7E6',
#                 'lavender': '#8883BE',
#                 'jasmin': '#BD81B7'
#             }
#         }

#     if type2 == None:
#         type2 = {
#             'meaty': '#F5EBE8',
#             'subs': {
#                 'savory': '#FBB360',
#                 'beefy': '#7B382A',
#                 'roasted': '#F7A69E'
#             }
#         }

#     if type3 == None:
#         type3 = {
#             'ethereal': '#F2F6EC',
#             'subs': {
#                 'cognac': '#BCE2D2',
#                 'fermented': '#79944F',
#                 'alcoholic': '#C2DA8F'
#             }
#         }

    # ------------------------------------------------------------
    # 1. Prepare / reduce features
    # ------------------------------------------------------------

    if space == 'poincare':
        # In Poincaré mode, assume embeds are already 2D Poincaré coordinates.
        # We do not apply PCA/t-SNE/UMAP because those are Euclidean reductions.
        reduced_features = embeds

        if torch.is_tensor(reduced_features):
            reduced_features = reduced_features.detach().cpu().numpy()

        reduced_features_torch = torch.tensor(reduced_features, dtype=torch.float32)
        reduced_features_torch = project_to_poincare_disk_for_viz(reduced_features_torch)
        reduced_features = reduced_features_torch.cpu().numpy()

        reduction_label = 'Poincare'

    else:
        # Euclidean mode: use your original reduction logic.
        if reduction_method == 'PCA':
            pca = PCA(n_components=2, iterated_power=10)
            reduced_features = pca.fit_transform(embeds)

            variance_explained = pca.explained_variance_ratio_
            variance_pc1 = variance_explained[0]
            variance_pc2 = variance_explained[1]
            print(variance_pc1, variance_pc2)

        elif reduction_method == 'tsne':
            tsne = manifold.TSNE(
                n_components=2,
                init="random",
                random_state=0,
                perplexity=perplexity,
            )
            reduced_features = tsne.fit_transform(embeds)

        elif reduction_method == 'UMAP':
            reduced_features = umap.UMAP(
                n_components=2,
                n_neighbors=perplexity,
                min_dist=0.0,
                metric='euclidean'
            ).fit_transform(X=embeds)

        elif reduction_method is None:
            reduced_features = embeds
            reduction_method = 'None'

        else:
            raise ValueError('Invalid reduction method')

        if torch.is_tensor(reduced_features):
            reduced_features = reduced_features.detach().cpu().numpy()

        reduction_label = reduction_method

    print(
        reduced_features[:, 0].min(),
        reduced_features[:, 0].max(),
        reduced_features[:, 1].min(),
        reduced_features[:, 1].max()
    )

    # ------------------------------------------------------------
    # 2. Build grid depending on geometry
    # ------------------------------------------------------------

    if space == 'poincare':
        x_grid, y_grid = np.meshgrid(
            np.linspace(-0.999, 0.999, grid_res),
            np.linspace(-0.999, 0.999, grid_res)
        )

        disk_mask = x_grid ** 2 + y_grid ** 2 < 0.999 ** 2

        grid_points = np.column_stack([
            x_grid.ravel(),
            y_grid.ravel()
        ])

        valid_grid_points = grid_points[disk_mask.ravel()]

    else:
        x_min, x_max = reduced_features[:, 0].min(), reduced_features[:, 0].max()
        y_min, y_max = reduced_features[:, 1].min(), reduced_features[:, 1].max()

        x_pad = euclidean_grid_padding * (x_max - x_min)
        y_pad = euclidean_grid_padding * (y_max - y_min)

        if x_pad == 0:
            x_pad = 1e-3
        if y_pad == 0:
            y_pad = 1e-3

        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min - x_pad, x_max + x_pad, grid_res),
            np.linspace(y_min - y_pad, y_max + y_pad, grid_res)
        )

        grid_points = np.vstack([
            x_grid.ravel(),
            y_grid.ravel()
        ])

        disk_mask = None
        valid_grid_points = None

    # ------------------------------------------------------------
    # 3. KDE function depending on geometry
    # ------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bandwidth_cache = {}

    def get_kde_values(label):
        plot_idx = required_desc.index(label)
        label_indices = np.where(y[:, plot_idx] == 1)[0]

        label_points = reduced_features[label_indices]

        if len(label_points) < 2:
            return np.full(x_grid.shape, np.nan)

        if space == 'euclidean':
            kde_label = gaussian_kde(label_points.T)
            kde_values_label = kde_label(grid_points)
            kde_values_label = kde_values_label.reshape(x_grid.shape)
            return kde_values_label

        else:
            label_points_torch = torch.tensor(
                label_points,
                dtype=torch.float32,
                device=device
            )

            label_points_torch = project_to_poincare_disk_for_viz(label_points_torch)

            if label not in bandwidth_cache:
                base_bandwidth = estimate_hyperbolic_bandwidth_for_viz(
                    label_points_torch,
                    distance_func=poincare_distance,
                    device=device
                )

                bandwidth_cache[label] = bandwidth_scale * base_bandwidth

                if forced_bandwidth is not None:
                    bandwidth_cache[label] = forced_bandwidth

            bandwidth = bandwidth_cache[label]

            kde_valid = hyperbolic_kde_for_viz(
                valid_grid_points,
                label_points_torch,
                bandwidth=bandwidth,
                distance_func=poincare_distance,
                device=device
            )

            kde_values_label = np.full(x_grid.shape, np.nan)
            kde_values_label.ravel()[disk_mask.ravel()] = kde_valid

            return kde_values_label

    # ------------------------------------------------------------
    # 4. Probability-mass contour level depending on geometry
    # ------------------------------------------------------------

    def get_probability_contour_level(kde_values, x_grid, y_grid, target_mass=0.5):
        dens = kde_values.ravel()

        valid = np.isfinite(dens)
        dens = dens[valid]

        if len(dens) == 0:
            return np.nan

        dx = x_grid[0, 1] - x_grid[0, 0]
        dy = y_grid[1, 0] - y_grid[0, 0]

        if space == 'euclidean':
            cell_area = dx * dy
            cell_mass = np.full_like(dens, cell_area, dtype=np.float64)

        else:
            area_element = hyperbolic_area_element_for_viz(
                x_grid,
                y_grid
            ).ravel()[valid]

            cell_mass = area_element * dx * dy

        order = np.argsort(dens)[::-1]

        dens_sorted = dens[order]
        mass_sorted = dens_sorted * cell_mass[order]

        cum_mass = np.cumsum(mass_sorted)

        if cum_mass[-1] <= 0:
            return np.nan

        cum_mass = cum_mass / cum_mass[-1]

        idx = np.searchsorted(cum_mass, target_mass)

        return dens_sorted[min(idx, len(dens_sorted) - 1)]

    # ------------------------------------------------------------
    # 5. Plot contours and embeddings
    # ------------------------------------------------------------

    def plot_contours(
        type_dictionary,
        bbox_to_anchor,
        target_mass_main=target_mass_main,
        target_mass_sub=target_mass_sub,
        islands_alpha=islands_alpha
    ):
        
        main_candidates = [
            k for k in type_dictionary.keys()
            if k not in ['subs', 'size']
        ]

        main_label = main_candidates[0] if len(main_candidates) > 0 else None

        if main_label is not None:
            kde_main = np.ma.masked_invalid(get_kde_values(main_label))

            main_level = get_probability_contour_level(
                kde_main.filled(np.nan),
                x_grid,
                y_grid,
                target_mass=target_mass_main
            )

            if np.isfinite(main_level) and kde_main.max() > main_level:
                plt.contourf(
                    x_grid,
                    y_grid,
                    kde_main,
                    levels=[main_level, kde_main.max()],
                    colors=[type_dictionary[main_label]],
                    alpha=islands_alpha
                )

        axes = plt.gca()
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)

        legend_elements = []

        subs = type_dictionary.get('subs', {})

        for label, color in subs.items():
            kde_sub = np.ma.masked_invalid(get_kde_values(label))

            sub_level = get_probability_contour_level(
                kde_sub.filled(np.nan),
                x_grid,
                y_grid,
                target_mass=target_mass_sub
            )

            if np.isfinite(sub_level) and kde_sub.max() > sub_level:
                plt.contour(
                    x_grid,
                    y_grid,
                    kde_sub,
                    levels=[sub_level],
                    colors=color,
                    linewidths=2
                )

            legend_elements.append(Patch(facecolor=color, label=label))

        if len(legend_elements) == 0:
            if main_label is not None:
                legend_elements.append(
                    Patch(facecolor=type_dictionary[main_label], label=main_label)
                )
                legend_title = None
            else:
                return  # no main class and no subclasses, so nothing to show
        else:
            legend_title = main_label if main_label is not None else None

        legend = plt.legend(
            handles=legend_elements,
            title=legend_title,
            bbox_to_anchor=bbox_to_anchor,
            prop={'size': 30}
        )

        if main_label is not None:
            legend.get_frame().set_facecolor(type_dictionary[main_label])
        else:
            legend.get_frame().set_facecolor('#FFFFFF')

        plt.gca().add_artist(legend)
        
        
    def plot_embeddings():
        if not show_embeddings:
            return

        points_to_plot = reduced_features

        if space == 'poincare':
            r2 = points_to_plot[:, 0] ** 2 + points_to_plot[:, 1] ** 2
            points_to_plot = points_to_plot[r2 < 1.0]

        plt.scatter(
            points_to_plot[:, 0],
            points_to_plot[:, 1],
            s=embedding_dot_size,
            c=embedding_dot_color,
            alpha=embedding_dot_alpha,
            linewidths=0,
            zorder=embedding_dot_zorder,
            rasterized=True
        )

    # ------------------------------------------------------------
    # 6. Build final figure
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(15, 15), dpi=500)

    for category_dict in categories:
        plot_contours(type_dictionary=category_dict, bbox_to_anchor=category_dict['size'])

    plot_embeddings()

    ax = plt.gca()
    ax.axis("off")

    if space == 'poincare':
        circle = plt.Circle(
            (0, 0),
            1.0,
            fill=False,
            linewidth=5,
            color="gray"
        )


        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)

#         plt.xlabel('Poincaré disk x', fontsize=35)
#         plt.ylabel('Poincaré disk y', fontsize=35)

    else:
        ax.set_aspect("auto")

#         if reduction_method == 'PCA':
#             plt.xlabel('Principal Component 1', fontsize=35)
#             plt.ylabel('Principal Component 2', fontsize=35)
#         elif reduction_method == 'tsne':
#             plt.xlabel('t-SNE 1', fontsize=35)
#             plt.ylabel('t-SNE 2', fontsize=35)
#         elif reduction_method == 'UMAP':
#             plt.xlabel('UMAP 1', fontsize=35)
#             plt.ylabel('UMAP 2', fontsize=35)
#         else:
#             plt.xlabel('Dimension 1', fontsize=35)
#             plt.ylabel('Dimension 2', fontsize=35)

    save_path = (
        "figs/islands/GSLF_islands_"
        + title
        + "_"
        + str(space)
        + "_"
        + str(reduction_label)
        + "_"
        + str(perplexity)
        + "_"
        + str(i)
        + ".pdf"
    )

    #plt.savefig(save_path)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    save_path = (
        "figs/islands/GSLF_islands_"
        + title
        + "_"
        + str(space)
        + "_"
        + str(reduction_label)
        + "_"
        + str(perplexity)
        + "_"
        + str(i)
        + ".png"
    )
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.close()
    
    
### Visualizations for Sagar used for paper ###

def to_numpy(a):
    """
    Converts torch tensors or numpy arrays to numpy arrays.
    """
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def compute_label_entropy1(
    y,
    method="binary",
    normalize=True,
    temperature=1.0,
    eps=1e-12,
    empty_value=np.nan,
):
    """
    Compute one entropy value per sample.

    Parameters
    ----------
    y : array-like, shape (n_samples, n_labels)
        Label / descriptor matrix.

    method : str
        "binary"  : for 0/1 labels. Entropy over active labels.
        "sum"     : for non-negative labels. Normalize each row by its sum.
        "softmax" : for real-valued labels, possibly negative.
        "softmax_of_squares": for real-valued labels, possibly negative, squares the values and then apply the entropy.
        "energy": square divided by sum of squares.

    normalize : bool
        If True, divide entropy by log(n_labels).

    temperature : float
        Used only for softmax. Smaller values make the distribution sharper.

    empty_value :
        Value used for rows with no active labels, e.g. [0, 0, ..., 0].

    Returns
    -------
    H : np.ndarray, shape (n_samples,)
        Entropy for each sample.
    """
    y = to_numpy(y).astype(float)

    if y.ndim != 2:
        raise ValueError(f"Expected y to have shape (n_samples, n_labels), got {y.shape}")

    n_samples, n_labels = y.shape

    if method == "binary":
        unique_values = np.unique(y)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError("method='binary' expects only 0/1 values.")

        row_sums = y.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > eps

        P = np.zeros_like(y)
        P[valid] = y[valid] / row_sums[valid]

        H = np.full(n_samples, empty_value, dtype=float)
        H[valid] = -np.sum(P[valid] * np.log(P[valid] + eps), axis=1)

    elif method == "sum":
        if np.any(y < 0):
            raise ValueError("method='sum' requires non-negative labels.")

        row_sums = y.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > eps

        P = np.zeros_like(y)
        P[valid] = y[valid] / row_sums[valid]

        H = np.full(n_samples, empty_value, dtype=float)
        H[valid] = -np.sum(P[valid] * np.log(P[valid] + eps), axis=1)

    elif method == "softmax":
        P = softmax(y / temperature, axis=1)
        H = -np.sum(P * np.log(P + eps), axis=1)
    
    elif method == "softmax_of_squares":
        P2 = softmax(y**2 / temperature, axis=1)
        H = -np.sum(P2 * np.log(P2 + eps), axis=1)

    elif method == "energy":
        energy = y ** 2
        energy_sum = energy.sum(axis=1, keepdims=True)
        valid = energy_sum[:, 0] > eps

        P = np.zeros_like(energy)
        P[valid] = energy[valid] / energy_sum[valid]

        H = np.full(y.shape[0], empty_value, dtype=float)
        H[valid] = -np.sum(P[valid] * np.log(P[valid] + eps), axis=1)

    else:
        raise ValueError("method must be one of: 'binary', 'sum', 'softmax'.")

    if normalize:
        H = H / np.log(n_labels)

    return H


def plot_embedding_entropy_and_radius_correlation1(
    X,
    y,
    radius=None,
    entropy_method="binary",
    normalize_entropy=True,
    temperature=1.0,
    title=None,
    point_size=8,
    alpha=0.8,
    draw_unit_circle=True,
):
    """
    Plot:
        1. Embedding colored by entropy.
        2. Entropy versus radius.

    Also prints Pearson and Spearman correlations.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        Embeddings.

    y : array-like, shape (n_samples, n_labels)
        Labels.

    radius : array-like, shape (n_samples,), optional
        Precomputed radius. For example:
            radius = poincare_distance(X, torch.zeros((1, 2)))

        If None, Euclidean radius ||X|| is used.

    entropy_method : str
        "binary", "sum", "softmax", "softmax_of_squares", or "energy".
    """
    X_np = to_numpy(X).astype(float)
    y_np = to_numpy(y).astype(float)

    if X_np.ndim != 2 or X_np.shape[1] != 2:
        raise ValueError(f"Expected X to have shape (n_samples, 2), got {X_np.shape}")

    if y_np.ndim != 2:
        raise ValueError(f"Expected y to have shape (n_samples, n_labels), got {y_np.shape}")

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X.shape[0]={X_np.shape[0]} and y.shape[0]={y_np.shape[0]}."
        )

    H = compute_label_entropy1(
        y_np,
        method=entropy_method,
        normalize=normalize_entropy,
        temperature=temperature,
    )

    if radius is None:
        radius_np = np.linalg.norm(X_np, axis=1)
    else:
        radius_np = to_numpy(radius).astype(float).reshape(-1)

    if radius_np.shape[0] != X_np.shape[0]:
        raise ValueError(
            f"radius must have shape (n_samples,), got {radius_np.shape} "
            f"while X has {X_np.shape[0]} samples."
        )

    valid = ~np.isnan(H)

    X_valid = X_np[valid]
    H_valid = H[valid]
    radius_valid = radius_np[valid]

    pearson_r, pearson_p = pearsonr(radius_valid, H_valid)
    spearman_rho, spearman_p = spearmanr(radius_valid, H_valid)

    print(f"Entropy method: {entropy_method}")
    print(f"Number of valid samples: {valid.sum()} / {len(valid)}")
    print(f"Pearson r  = {pearson_r:.4f}, p = {pearson_p:.3e}")
    print(f"Spearman ρ = {spearman_rho:.4f}, p = {spearman_p:.3e}")

    # --------------------------------------------------------
    # Plot 1: embeddings colored by entropy
    # --------------------------------------------------------

    plt.figure(figsize=(7, 6))

    sc = plt.scatter(
        X_valid[:, 0],
        X_valid[:, 1],
        c=H_valid,
        s=point_size,
        alpha=alpha,
        cmap="plasma",
    )

    if draw_unit_circle:
        circle = plt.Circle(
            (0, 0),
            1,
            fill=False,
            linestyle="--",
            linewidth=1.2,
        )
        plt.gca().add_patch(circle)
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)

    plt.axis("equal")
#     plt.xlabel("Embedding dimension 1")
#     plt.ylabel("Embedding dimension 2")

    if title is None:
        title = f"Embedding colored by {entropy_method} entropy"

    plt.title(title)
    plt.colorbar(sc, label="Normalized entropy" if normalize_entropy else "Entropy")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 2: entropy vs radius
    # --------------------------------------------------------

    plt.figure(figsize=(6, 4))

    plt.scatter(
        radius_valid,
        H_valid,
        s=point_size,
        alpha=alpha,
        c=H_valid,
        cmap="plasma"
    )

    a, b = np.polyfit(radius_valid, H_valid, 1)
    xs = np.linspace(radius_valid.min(), radius_valid.max(), 200)
    plt.plot(xs, a * xs + b, c='purple')

    plt.xlabel("Radius")
    plt.ylabel("Normalized entropy" if normalize_entropy else "Entropy")
    plt.title(
        f"Entropy vs radius\n"
        f"Pearson r={pearson_r:.3f}, Spearman ρ={spearman_rho:.3f}"
    )

    plt.tight_layout()
    plt.show()

    return {
        "entropy": H,
        "radius": radius_np,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def compute_label_entropy2(
    y,
    method="binary",
    normalize=True,
    temperature=1.0,
    eps=1e-12,
    empty_value=np.nan,
):
    """
    Compute one entropy value per sample.

    Parameters
    ----------
    y : array-like, shape (n_samples, n_labels)
        Label / descriptor matrix.

    method : str
        "binary"  : for 0/1 labels. Entropy over active labels.
        "sum"     : for non-negative labels. Normalize each row by its sum.
        "softmax" : for real-valued labels, possibly negative.
        "softmax_of_squares": for real-valued labels, possibly negative, squares the values and then apply the entropy.
        "energy": square divided by sum of squares.

    normalize : bool
        If True, divide entropy by log(n_labels).

    temperature : float
        Used only for softmax. Smaller values make the distribution sharper.

    empty_value :
        Value used for rows with no active labels, e.g. [0, 0, ..., 0].

    Returns
    -------
    H : np.ndarray, shape (n_samples,)
        Entropy for each sample.
    """
    y = to_numpy(y).astype(float)

    if y.ndim != 2:
        raise ValueError(f"Expected y to have shape (n_samples, n_labels), got {y.shape}")

    n_samples, n_labels = y.shape

    if method == "binary":
        unique_values = np.unique(y)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError("method='binary' expects only 0/1 values.")

        row_sums = y.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > eps

        P = np.zeros_like(y)
        P[valid] = y[valid] / row_sums[valid]

        H = np.full(n_samples, empty_value, dtype=float)
        H[valid] = -np.sum(P[valid] * np.log(P[valid] + eps), axis=1)

    elif method == "sum":
        if np.any(y < 0):
            raise ValueError("method='sum' requires non-negative labels.")

        row_sums = y.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > eps

        P = np.zeros_like(y)
        P[valid] = y[valid] / row_sums[valid]

        H = np.full(n_samples, empty_value, dtype=float)
        H[valid] = -np.sum(P[valid] * np.log(P[valid] + eps), axis=1)

    elif method == "softmax":
        P = softmax(y / temperature, axis=1)
        H = -np.sum(P * np.log(P + eps), axis=1)
    
    elif method == "softmax_of_squares":
        P2 = softmax(y**2 / temperature, axis=1)
        H = -np.sum(P2 * np.log(P2 + eps), axis=1)

    elif method == "energy":
        energy = y ** 2
        energy_sum = energy.sum(axis=1, keepdims=True)
        valid = energy_sum[:, 0] > eps

        P = np.zeros_like(energy)
        P[valid] = energy[valid] / energy_sum[valid]

        H = np.full(y.shape[0], empty_value, dtype=float)
        H[valid] = -np.sum(P[valid] * np.log(P[valid] + eps), axis=1)

    else:
        raise ValueError("method must be one of: 'binary', 'sum', 'softmax'.")

    if normalize:
        H = H / np.log(n_labels)

    return H


def plot_embedding_entropy_and_radius_correlation2(
    X,
    y,
    radius=None,
    entropy_method="binary",
    normalize_entropy=True,
    temperature=1.0,
    title=None,
    point_size=8,
    alpha=0.8,
    draw_unit_circle=True,
    save_embedding_path=None,
    save_correlation_path=None,
    entropy_vmin=None,
    entropy_vmax=None,
):
    """
    Plot:
        1. Embedding colored by entropy.
        2. Entropy versus radius.

    Also prints Pearson and Spearman correlations.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        Embeddings.

    y : array-like, shape (n_samples, n_labels)
        Labels.

    radius : array-like, shape (n_samples,), optional
        Precomputed radius. For example:
            radius = poincare_distance(X, torch.zeros((1, 2)))

        If None, Euclidean radius ||X|| is used.

    entropy_method : str
        "binary", "sum", "softmax", "softmax_of_squares", or "energy".
    """
    X_np = to_numpy(X).astype(float)
    y_np = to_numpy(y).astype(float)

    if X_np.ndim != 2 or X_np.shape[1] != 2:
        raise ValueError(f"Expected X to have shape (n_samples, 2), got {X_np.shape}")

    if y_np.ndim != 2:
        raise ValueError(f"Expected y to have shape (n_samples, n_labels), got {y_np.shape}")

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X.shape[0]={X_np.shape[0]} and y.shape[0]={y_np.shape[0]}."
        )

    H = compute_label_entropy2(
        y_np,
        method=entropy_method,
        normalize=normalize_entropy,
        temperature=temperature,
    )

    if radius is None:
        radius_np = np.linalg.norm(X_np, axis=1)
    else:
        radius_np = to_numpy(radius).astype(float).reshape(-1)

    if radius_np.shape[0] != X_np.shape[0]:
        raise ValueError(
            f"radius must have shape (n_samples,), got {radius_np.shape} "
            f"while X has {X_np.shape[0]} samples."
        )

    valid = ~np.isnan(H)

    X_valid = X_np[valid]
    H_valid = H[valid]
    radius_valid = radius_np[valid]

    pearson_r, pearson_p = pearsonr(radius_valid, H_valid)
    spearman_rho, spearman_p = spearmanr(radius_valid, H_valid)

    print(f"Entropy method: {entropy_method}")
    print(f"Number of valid samples: {valid.sum()} / {len(valid)}")
    print(f"Pearson r  = {pearson_r:.4f}, p = {pearson_p:.3e}")
    print(f"Spearman ρ = {spearman_rho:.4f}, p = {spearman_p:.3e}")

    # --------------------------------------------------------
    # Plot 1: embeddings colored by entropy
    # --------------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(30, 30))

    sc = ax1.scatter(
        X_valid[:, 0],
        X_valid[:, 1],
        c=H_valid,
        s=300,
        alpha=alpha,
        cmap="plasma",
        vmin=entropy_vmin,
        vmax=entropy_vmax,
    )

    if draw_unit_circle:
        circle = plt.Circle(
            (0, 0),
            1,
            color='gray',
            fill=False,
            #linestyle="--",
            linewidth=10,
        )
        ax1.add_patch(circle)
        ax1.set_xlim(-1.05, 1.05)
        ax1.set_ylim(-1.05, 1.05)

    ax1.axis('equal')
    ax1.axis('off')

#     if title is None:
#         title = f"Embedding colored by {entropy_method} entropy"

    ax1.set_title(title)
#     cbar= fig1.colorbar(sc, ax=ax1)
#     cbar.set_label("Normalized entropy" if normalize_entropy else "Entropy", fontsize=90)
    divider = make_axes_locatable(ax1)

    cax = divider.append_axes(
        "right",
        size="4%",     # width of colorbar
        pad=0.15       # space between plot and colorbar
    )

    cbar = fig1.colorbar(sc, cax=cax)

    cbar.set_label(
        "Normalized entropy" if normalize_entropy else "Entropy",
        fontsize=90
    )

    cbar.ax.tick_params(labelsize=50)
    
    fig1.tight_layout()

    if save_embedding_path is not None:
        fig1.savefig(save_embedding_path, bbox_inches="tight", pad_inches=0.2)
        
    plt.show()
    plt.close(fig1)

    # --------------------------------------------------------
    # Plot 2: entropy vs radius
    # --------------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.scatter(
        radius_valid,
        H_valid,
        s=point_size,
        alpha=alpha,
        c=H_valid,
        cmap="plasma",
        vmin=entropy_vmin,
        vmax=entropy_vmax,
    )

    a, b = np.polyfit(radius_valid, H_valid, 1)
    xs = np.linspace(radius_valid.min(), radius_valid.max(), 200)
    ax2.plot(xs, a * xs + b, c='purple')

    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_xlabel("Hyperbolic radius", fontsize=30)
    ax2.set_ylabel(
        "Normalized entropy" if normalize_entropy else "Entropy",
        fontsize=30
    )

    fig2.tight_layout()

    if save_correlation_path is not None:
        fig2.savefig(save_correlation_path, bbox_inches="tight", pad_inches=0.2)

    plt.show()
    plt.close(fig2)
#     plt.title(
#         f"Entropy vs radius\n"
#         f"Pearson r={pearson_r:.3f}, Spearman ρ={spearman_rho:.3f}"
#     )


    return {
        "entropy": H,
        "radius": radius_np,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }

def plot_embedding_intensity_and_radius_correlation(
    X,
    labels,
    radius=None,
    descriptor_index=0,
    descriptor_name="Intensity",
    title=None,
    point_size=8,
    alpha=0.8,
    draw_unit_circle=True,
    save_embedding_path=None,
    save_correlation_path=None,
):
    """
    Plot:
        1. Embedding colored by intensity.
        2. Intensity versus radius.

    Also prints Pearson and Spearman correlations.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        Embeddings.

    labels : array-like, shape (n_samples, n_labels)
        Descriptor matrix. For the Sagar dataset, intensity is labels[:, 0].

    radius : array-like, shape (n_samples,), optional
        Precomputed hyperbolic radius. For example:
            radius = poincare_distance(X, torch.zeros((1, 2)))

        If None, Euclidean radius ||X|| is used.

    descriptor_index : int
        Index of the descriptor to plot. For intensity in Sagar, use 0.

    descriptor_name : str
        Name used in plots and printed output.
    """

    X_np = to_numpy(X).astype(float)
    labels_np = to_numpy(labels).astype(float)

    if X_np.ndim != 2 or X_np.shape[1] != 2:
        raise ValueError(f"Expected X to have shape (n_samples, 2), got {X_np.shape}")

    if labels_np.ndim != 2:
        raise ValueError(
            f"Expected labels to have shape (n_samples, n_labels), got {labels_np.shape}"
        )

    if X_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"X and labels must have the same number of samples. "
            f"Got X.shape[0]={X_np.shape[0]} and labels.shape[0]={labels_np.shape[0]}."
        )

    if descriptor_index < 0 or descriptor_index >= labels_np.shape[1]:
        raise ValueError(
            f"descriptor_index={descriptor_index} is invalid for labels with "
            f"{labels_np.shape[1]} descriptors."
        )

    # Intensity descriptor for Sagar: labels[:, 0]
    y = labels_np[:, descriptor_index]

    if radius is None:
        radius_np = np.linalg.norm(X_np, axis=1)
    else:
        radius_np = to_numpy(radius).astype(float).reshape(-1)

    if radius_np.shape[0] != X_np.shape[0]:
        raise ValueError(
            f"radius must have shape (n_samples,), got {radius_np.shape} "
            f"while X has {X_np.shape[0]} samples."
        )

    valid = np.isfinite(y) & np.isfinite(radius_np)

    X_valid = X_np[valid]
    y_valid = y[valid]
    radius_valid = radius_np[valid]

    pearson_r, pearson_p = pearsonr(radius_valid, y_valid)
    spearman_rho, spearman_p = spearmanr(radius_valid, y_valid)

    print(f"Descriptor: {descriptor_name}")
    print(f"Number of valid samples: {valid.sum()} / {len(valid)}")
    print(f"Pearson r  = {pearson_r:.4f}, p = {pearson_p:.3e}")
    print(f"Spearman ρ = {spearman_rho:.4f}, p = {spearman_p:.3e}")

    # --------------------------------------------------------
    # Plot 1: embedding colored by intensity
    # --------------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(30, 30))

    sc = ax1.scatter(
        X_valid[:, 0],
        X_valid[:, 1],
        c=y_valid,
        s=300,
        alpha=alpha,
        cmap="plasma",
    )

    if draw_unit_circle:
        circle = plt.Circle(
            (0, 0),
            1,
            color="gray",
            fill=False,
            linewidth=10,
        )
        ax1.add_patch(circle)
        ax1.set_xlim(-1.05, 1.05)
        ax1.set_ylim(-1.05, 1.05)

    ax1.axis("equal")
    ax1.axis("off")

#     if title is None:
#         title = f"Embedding colored by {descriptor_name.lower()}"

    ax1.set_title(title)

    divider = make_axes_locatable(ax1)

    cax = divider.append_axes(
        "right",
        size="4%",
        pad=0.15,
    )

    cbar = fig1.colorbar(sc, cax=cax)
    cbar.set_label(descriptor_name, fontsize=90)
    cbar.ax.tick_params(labelsize=50)

    fig1.tight_layout()

    if save_embedding_path is not None:
        fig1.savefig(save_embedding_path, bbox_inches="tight", pad_inches=0.2)

    plt.show()
    plt.close(fig1)

    # --------------------------------------------------------
    # Plot 2: intensity vs radius
    # --------------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.scatter(
        radius_valid,
        y_valid,
        s=point_size,
        alpha=alpha,
        c=y_valid,
        cmap="plasma",
    )

    a, b = np.polyfit(radius_valid, y_valid, 1)
    xs = np.linspace(radius_valid.min(), radius_valid.max(), 200)
    ax2.plot(xs, a * xs + b, c="purple")

    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_xlabel("Hyperbolic radius", fontsize=30)
    ax2.set_ylabel(descriptor_name, fontsize=30)

    fig2.tight_layout()

    if save_correlation_path is not None:
        fig2.savefig(save_correlation_path, bbox_inches="tight", pad_inches=0.2)

    plt.show()
    plt.close(fig2)

    return {
        "descriptor_values": y,
        "radius": radius_np,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


# Visualizations GSLF entropies
def project_inside_poincare_disk(X, eps=1e-5):
    """
    Project points numerically inside the open Poincare disk.
    """
    X = to_numpy(X).astype(np.float32)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"Expected an embedding with shape (n_samples, 2), got {X.shape}."
        )

    norms = np.linalg.norm(X, axis=1, keepdims=True)

    scale = np.where(
        norms >= 1.0 - eps,
        (1.0 - eps) / np.maximum(norms, eps),
        1.0,
    )

    return X * scale


def compute_hyperbolic_radius(X):
    """
    Compute the Poincare distance between each point and the origin.
    """
    X = project_inside_poincare_disk(X)

    X_torch = torch.tensor(X, dtype=torch.float32)
    origin = torch.zeros((1, 2), dtype=torch.float32)

    with torch.no_grad():
        radius = poincare_distance(X_torch, origin)

    return to_numpy(radius).reshape(-1)



def compute_active_label_entropy3(
    binary_labels,
    normalize=True,
    eps=1e-12,
):
    """
    Active label entropy for the binary GSLF descriptors.

    For a molecule with K active descriptors:

        H = log(K)

    before optional normalization.
    """
    binary_labels = to_numpy(binary_labels).astype(float)

    if binary_labels.ndim != 2:
        raise ValueError(
            "Expected binary_labels to have shape "
            f"(n_samples, n_descriptors), got {binary_labels.shape}."
        )

    unique_values = np.unique(binary_labels)

    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            "Active label entropy requires a binary matrix containing only 0 and 1."
        )

    n_samples, n_descriptors = binary_labels.shape

    active_counts = binary_labels.sum(axis=1, keepdims=True)
    valid = active_counts[:, 0] > 0

    probabilities = np.zeros_like(binary_labels)
    probabilities[valid] = (
        binary_labels[valid] / active_counts[valid]
    )

    entropy = np.full(n_samples, np.nan, dtype=float)

    entropy[valid] = -np.sum(
        probabilities[valid]
        * np.log(probabilities[valid] + eps),
        axis=1,
    )

    if normalize:
        entropy[valid] /= np.log(n_descriptors)

    return entropy


def compute_orthogonalized_descriptors_entropy3(
    binary_labels,
    normalize=True,
    temperature=1.0,
    n_components=None,
    eps=1e-12,
):
    """
    Orthogonalize the descriptor matrix using PCA, without standardization,
    and compute entropy after applying a softmax to the PCA coordinates.

    By default, all descriptor dimensions are retained. Therefore, PCA
    changes the coordinate system but does not reduce dimensionality.

    Returns
    -------
    entropy : np.ndarray
        Orthogonalized descriptors entropy for each molecule.

    pca_scores : np.ndarray
        PCA coordinates.

    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    """
    binary_labels = to_numpy(binary_labels).astype(float)

    if binary_labels.ndim != 2:
        raise ValueError(
            "Expected binary_labels to have shape "
            f"(n_samples, n_descriptors), got {binary_labels.shape}."
        )

    n_samples, n_descriptors = binary_labels.shape

    if n_components is None:
        n_components = n_descriptors

    n_components = min(
        n_components,
        n_samples,
        n_descriptors,
    )

    # PCA centers the columns automatically.
    # No StandardScaler is applied.
    pca = PCA(
        n_components=n_components,
        svd_solver="full",
    )

    pca_scores = pca.fit_transform(binary_labels)

    probabilities = softmax(
        pca_scores / temperature,
        axis=1,
    )

    entropy = -np.sum(
        probabilities * np.log(probabilities + eps),
        axis=1,
    )

    if normalize:
        entropy /= np.log(n_components)

    return entropy, pca_scores, pca



def plot_entropy_embedding_and_correlation3(
    X,
    entropy,
    radius,
    entropy_label,
    output_prefix,
    output_directory="gslf_entropy_figures",
    point_size_embedding=12,
    point_size_correlation=8,
    alpha=0.75,
    cmap="plasma",
    regression_line_color="purple",
    embedding_figsize=(8, 8),
    correlation_figsize=(8, 5),
    hide_correlation_ticks=True,
):
    """
    Generate two separate figures:

    1. Poincare embedding colored by entropy.
    2. Entropy as a function of hyperbolic radius.

    Files are saved as both PDF and PNG.
    """
    X = to_numpy(X).astype(float)
    entropy = to_numpy(entropy).astype(float).reshape(-1)
    radius = to_numpy(radius).astype(float).reshape(-1)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"Expected X to have shape (n_samples, 2), got {X.shape}."
        )

    if not (
        X.shape[0]
        == entropy.shape[0]
        == radius.shape[0]
    ):
        raise ValueError(
            "X, entropy, and radius must contain the same number of observations."
        )

    valid = (
        np.isfinite(entropy)
        & np.isfinite(radius)
        & np.all(np.isfinite(X), axis=1)
    )

    X_valid = X[valid]
    entropy_valid = entropy[valid]
    radius_valid = radius[valid]

    if len(entropy_valid) < 3:
        raise ValueError("Too few valid observations for correlation analysis.")

    pearson_result = pearsonr(radius_valid, entropy_valid)
    spearman_result = spearmanr(radius_valid, entropy_valid)

    pearson_r = float(
        getattr(pearson_result, "statistic", pearson_result[0])
    )
    spearman_rho = float(
        getattr(spearman_result, "statistic", spearman_result[0])
    )

    print(f"\n{entropy_label}")
    print(f"Valid molecules: {valid.sum()} / {len(valid)}")
    print(f"Pearson r:  {pearson_r:.4f}")
    print(f"Spearman r: {spearman_rho:.4f}")

    entropy_vmin = np.nanmin(entropy_valid)
    entropy_vmax = np.nanmax(entropy_valid)

    os.makedirs(output_directory, exist_ok=True)

    embedding_pdf = os.path.join(
        output_directory,
        f"{output_prefix}_embedding.pdf",
    )
    embedding_png = os.path.join(
        output_directory,
        f"{output_prefix}_embedding.png",
    )
    correlation_pdf = os.path.join(
        output_directory,
        f"{output_prefix}_radius_correlation.pdf",
    )
    correlation_png = os.path.join(
        output_directory,
        f"{output_prefix}_radius_correlation.png",
    )

    # --------------------------------------------------------
    # Embedding colored by entropy
    # --------------------------------------------------------

    fig_embedding, ax_embedding = plt.subplots(
        figsize=embedding_figsize
    )

    scatter_embedding = ax_embedding.scatter(
        X_valid[:, 0],
        X_valid[:, 1],
        c=entropy_valid,
        s=point_size_embedding,
        alpha=alpha,
        cmap=cmap,
        vmin=entropy_vmin,
        vmax=entropy_vmax,
        edgecolors="none",
        rasterized=True,
    )

    unit_circle = plt.Circle(
        (0, 0),
        1,
        fill=False,
        linewidth=2,
    )

    ax_embedding.add_patch(unit_circle)
    ax_embedding.set_xlim(-1.03, 1.03)
    ax_embedding.set_ylim(-1.03, 1.03)
    ax_embedding.set_aspect("equal")
    ax_embedding.axis("off")

    divider = make_axes_locatable(ax_embedding)

    colorbar_axis = divider.append_axes(
        "right",
        size="4%",
        pad=0.15,
    )

    colorbar = fig_embedding.colorbar(
        scatter_embedding,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        entropy_label,
        fontsize=18,
    )
    colorbar.ax.tick_params(labelsize=13)

    fig_embedding.tight_layout()

    fig_embedding.savefig(
        embedding_pdf,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig_embedding.savefig(
        embedding_png,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.show()
    plt.close(fig_embedding)

    # --------------------------------------------------------
    # Entropy versus hyperbolic radius
    # --------------------------------------------------------

    fig_correlation, ax_correlation = plt.subplots(
        figsize=correlation_figsize
    )

    ax_correlation.scatter(
        radius_valid,
        entropy_valid,
        c=entropy_valid,
        s=point_size_correlation,
        alpha=alpha,
        cmap=cmap,
        vmin=entropy_vmin,
        vmax=entropy_vmax,
        edgecolors="none",
        rasterized=True,
    )

    slope, intercept = np.polyfit(
        radius_valid,
        entropy_valid,
        deg=1,
    )

    radius_grid = np.linspace(
        radius_valid.min(),
        radius_valid.max(),
        300,
    )

    ax_correlation.plot(
        radius_grid,
        slope * radius_grid + intercept,
        color=regression_line_color,
        linewidth=2.5,
    )

    ax_correlation.set_xlabel(
        "Hyperbolic radius",
        fontsize=18,
    )
    ax_correlation.set_ylabel(
        entropy_label,
        fontsize=18,
    )

    ax_correlation.tick_params(
        axis="both",
        labelsize=13,
    )

    if hide_correlation_ticks:
        ax_correlation.set_xticks([])
        ax_correlation.set_yticks([])

    fig_correlation.tight_layout()

    fig_correlation.savefig(
        correlation_pdf,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig_correlation.savefig(
        correlation_png,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.show()
    plt.close(fig_correlation)

    return {
        "entropy": entropy,
        "radius": radius,
        "valid": valid,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "embedding_pdf": embedding_pdf,
        "correlation_pdf": correlation_pdf,
    }
