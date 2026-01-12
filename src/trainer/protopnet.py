import torch
from tqdm import tqdm
import sys
import time
from copy import deepcopy
from .base import Trainer
import numpy as np
from ..receptive_field import compute_rf_prototype
import cv2
import os 
import matplotlib.pyplot as plt
import wandb

class ProtoPNetTrainer(Trainer):

  model: torch.nn.Module

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # the metrics to keep track of
    self._metrics = {
      "total_loss" : 0.0, 
      "cross_entropy" : 0.0, 
      "cluster" : 0.0, 
      "separation" : 0.0,
      "l1" : 0.0,
      "n_correct" : 0, 
      "n_samples" : 0, 
      "accuracy" : 0.0,
      "time" : 0.0
    }

    self.metrics = {
      "train" : deepcopy(self._metrics),
      "val" : deepcopy(self._metrics),
      "test" : deepcopy(self._metrics)
    }
    self.ce_coef = float(kwargs["loss"]["ce"])
    self.cluster_coef = float(kwargs["loss"]["cluster"])
    self.separation_coef = float(kwargs["loss"]["separation"])
    self.l1_coef = float(kwargs["loss"]["l1"])

  @torch.enable_grad
  def train(self, initial=False, stage="warm"): 
    if stage == "warm": 
      self.model.warm_only() 
      optimizer = self.warm_optimizer
    elif stage == "joint": 
      self.model.joint() 
      optimizer = self.joint_optimizer
    elif stage == "last": 
      self.model.last_only()
      optimizer = self.last_optimizer
    else: 
      raise ValueError("Not a valid stage.")

    train_metrics = deepcopy(self._metrics) 
    start = time.time()
    self.model.train()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.train_dl, leave=False, file=tqdm_file):
      label_cpu = batch["y"]
      x, label = batch["x"].to(self.device), batch["y"].to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = 128 # shouldn't be hardcoded, idk why it's set to what it is in original repo 
      _, predicted = torch.max(logits.data, 1)
      train_metrics["n_correct"] += (predicted == label).sum().item()

      # calculate cluster cost by looking at minimum distances between prototype and nearest patch. 
      # 1. Filter prototype_class_identity: (2000, 200) -> (2000, B). It consists of rows that represent each prototype. 
      # 2. If we transpose it, then (B, 2000). Given the ith sample from batch, matrix[i] is a vector that 
      # shows all prototypes associated with it of form: [0 ... 0 1 ... 1 0 ... 0]
      prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)
      # prototypes_of_correct_class[i][j] = 1 if jth prototype corresponds to class in ith sample of batch

      # we have the min_distances of shape (B, 2000). We don't care about the distances from ith sample to a prototype that isn't associated with it, 
      # so we just mask it by doing A = min_distances * prototypes_of_correct_class, of shape (B, 2000) 
      # A[i][j] is the distance from the jth prototype to the class that the prototype corresponds to. 
      # invert them and then compute the max (so min distance) across the prototype dimension. This ignores the majority of distances which are 0. 
      # and finds the prototype that has the min distance. 
      inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

      # we can't just take the min of the masked min_distances directly since it will always be 0 
      # we want to minimize only over the elements that are not masked
      # this is why we take the max of inverted distances, mask them out, and then invert again. 
      cluster_cost = torch.mean(max_dist - inverted_distances)

      # calculate separation cost
      prototypes_of_wrong_class = 1 - prototypes_of_correct_class
      inverted_distances_to_nontarget_prototypes, _ = \
          torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
      separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

      l1_cost = self.model.last_layer.weight.norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      if not initial: 
        self.model.zero_grad()
        total_loss.backward()
        optimizer.step()

      train_metrics["cross_entropy"] += ce_loss.item() 
      train_metrics["cluster"] += cluster_cost.item() 
      train_metrics["separation"] += separation_cost.item() 
      train_metrics["l1"] += l1_cost.item() 
      train_metrics["total_loss"] += total_loss.item() 

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      train_metrics["n_correct"] += correct
      train_metrics["n_samples"] += label.size(0)

      del label
      del predicted
      del min_distances

    train_metrics["time"] = time.time() - start
    train_metrics["total_loss"] /= train_metrics["n_samples"]
    train_metrics["accuracy"] = train_metrics["n_correct"] / train_metrics["n_samples"]
    train_metrics["cross_entropy"] /= train_metrics["n_samples"]
    train_metrics["cluster"] /= train_metrics["n_samples"]
    train_metrics["separation"] /= train_metrics["n_samples"]
    train_metrics["l1"] /= train_metrics["n_samples"]

    self.metrics["train"] = train_metrics
    if not initial: self.epoch += 1 
    return train_metrics

  @torch.no_grad
  def val(self):
    val_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.val_dl, leave=False, file=tqdm_file):
      label_cpu = batch["y"]
      x, label = batch["x"].to(self.device), batch["y"].to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = 128 # shouldn't be hardcoded, idk why it's set to what it is in original repo 
      _, predicted = torch.max(logits.data, 1)
      val_metrics["n_correct"] += (predicted == label).sum().item()

      prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)

      inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
      cluster_cost = torch.mean(max_dist - inverted_distances)

      # calculate separation cost
      prototypes_of_wrong_class = 1 - prototypes_of_correct_class
      inverted_distances_to_nontarget_prototypes, _ = \
          torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
      separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

      l1_cost = self.model.last_layer.weight.norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      val_metrics["cross_entropy"] += ce_loss.item() 
      val_metrics["cluster"] += cluster_cost.item() 
      val_metrics["separation"] += separation_cost.item() 
      val_metrics["l1"] += l1_cost.item() 
      val_metrics["total_loss"] += total_loss.item() 

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      val_metrics["n_correct"] += correct
      val_metrics["n_samples"] += label.size(0)

      del label
      del predicted
      del min_distances

    val_metrics["time"] = time.time() - start
    val_metrics["total_loss"] /= val_metrics["n_samples"]
    val_metrics["accuracy"] = val_metrics["n_correct"] / val_metrics["n_samples"]
    val_metrics["cross_entropy"] /= val_metrics["n_samples"]
    val_metrics["cluster"] /= val_metrics["n_samples"]
    val_metrics["separation"] /= val_metrics["n_samples"]
    val_metrics["l1"] /= val_metrics["n_samples"]

    self.metrics["val"] = val_metrics
    return val_metrics

  @torch.no_grad
  def test(self): 
    test_metrics = deepcopy(self._metrics)
    start = time.time()
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for batch in tqdm(self.test_dl, leave=False, file=tqdm_file):
      label_cpu = batch["y"]
      x, label = batch["x"].to(self.device), batch["y"].to(self.device)
      logits, min_distances = self.model(x) 
      ce_loss = torch.nn.functional.cross_entropy(logits, label) 

      max_dist = 128 # shouldn't be hardcoded, idk why it's set to what it is in original repo 
      _, predicted = torch.max(logits.data, 1)
      test_metrics["n_correct"] += (predicted == label).sum().item()

      prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)
      inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

      cluster_cost = torch.mean(max_dist - inverted_distances)

      # calculate separation cost
      prototypes_of_wrong_class = 1 - prototypes_of_correct_class
      inverted_distances_to_nontarget_prototypes, _ = \
          torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
      separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

      l1_cost = self.model.last_layer.weight.norm(p=1)

      total_loss = self.ce_coef * ce_loss \
          + self.cluster_coef * cluster_cost \
          + self.separation_coef * separation_cost \
          + self.l1_coef * l1_cost

      test_metrics["cross_entropy"] += ce_loss.item() 
      test_metrics["cluster"] += cluster_cost.item() 
      test_metrics["separation"] += separation_cost.item() 
      test_metrics["l1"] += l1_cost.item() 
      test_metrics["total_loss"] += total_loss.item() 

      # Calculate accuracy
      pred_labels = logits.argmax(dim=1)
      correct = (pred_labels == label).sum().item() 
      test_metrics["n_correct"] += correct
      test_metrics["n_samples"] += label.size(0)

      del label
      del predicted
      del min_distances
    test_metrics["time"] = time.time() - start
    test_metrics["total_loss"] /= test_metrics["n_samples"]
    test_metrics["accuracy"] = test_metrics["n_correct"] / test_metrics["n_samples"]
    test_metrics["cross_entropy"] /= test_metrics["n_samples"]
    test_metrics["cluster"] /= test_metrics["n_samples"]
    test_metrics["separation"] /= test_metrics["n_samples"]
    test_metrics["l1"] /= test_metrics["n_samples"]

    self.metrics["test"] = test_metrics
    return test_metrics

  @torch.no_grad
  def push(self, visualize=True): 

    def find_high_activation_crop(activation_map, percentile=95):
      threshold = np.percentile(activation_map, percentile)
      mask = np.ones(activation_map.shape)
      mask[activation_map < threshold] = 0
      lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
      for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
          lower_y = i
          break
      for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
          upper_y = i
          break
      for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
          lower_x = j
          break
      for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
          upper_x = j
          break
      return lower_y, upper_y+1, lower_x, upper_x+1

    self.model.eval()
    prototype_shape = self.model.prototype_shape
    n_prototypes, _, proto_h, proto_w = prototype_shape

    # global_min_proto_dist[i] closest distance from prototype to any patch in any sample of corresponding class
    global_min_proto_dist = np.full(n_prototypes, np.inf)  # (2000)

    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(prototype_shape)   # (2000, 128, 1, 1)

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)

    for push_iter, batch in enumerate(tqdm(self.push_dl, desc="push")): 
      search_batch = batch["x"] 
      search_y = batch["y"]

      start_index_of_search_batch = push_iter * self.push_dl.batch_size

      with torch.no_grad(): 
        search_batch = search_batch.cuda() 

        # we will need access to the patches themselves, along with ALL distances 
        # from each patch to each prototype, BEFORE minimizing (so we can track index) 
        # latent features, distances
        protoL_input_torch, proto_dist_torch = self.model.push_forward(search_batch) # (B, 128, 7, 7), (B, 2000, 7, 7) 

      protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
      proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

      del protoL_input_torch, proto_dist_torch

      # we want to focus on each class, so keep a mapping from each class index to the sample index of the batch
      class_to_img_index_dict = {key: [] for key in range(self.model.num_classes)}
      # img_y is the image's integer label
      for img_index, img_y in enumerate(search_y):
        img_label = img_y.item()
        class_to_img_index_dict[img_label].append(img_index)


      # now iterate through prototypes 
      for j in range(n_prototypes): 
        target_class = torch.argmax(self.model.prototype_class_identity[j]).item()
        # if there is not images of the target_class from this batch
        # we go on to the next prototype
        if len(class_to_img_index_dict[target_class]) == 0:
          continue

        # take the distances from prototypes to all patches in a batch  
        # filter out the samples in batch by target class, say B = 70 and there are K samples left corresponding to prototype j 
        proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]] # (K, 2000, 7, 7)

        # now just focus on the prototype j 
        proto_dist_j = proto_dist_j[:,j,:,:]    # (K, 7, 7)
        # proto_dist_j[a, b, c] = distance of prototype j on the (b, c)th patch of sample a in our batch 

        # now just find the minimum distance, and update the global dist with the correct index and patch if it is smaller 
        batch_min_proto_dist_j = np.amin(proto_dist_j) 
        if batch_min_proto_dist_j < global_min_proto_dist[j]: 
          # this is the specific index (a, b, c) \in (K, 7, 7) corresponding to new min
          batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))

          # change the argmin index from the index among images of the target class to the index in the entire search batch
          batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]] 

          # location of best patch so far 
          img_index_in_batch, fmap_height_start_index, fmap_width_start_index = batch_argmin_proto_dist_j
          fmap_height_end_index, fmap_width_end_index = fmap_height_start_index + 1, fmap_width_start_index + 1 

          # now grab the specific patch of shape (128, 1, 1) from the batch 
          batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                 :,
                                                 fmap_height_start_index:fmap_height_end_index,
                                                 fmap_width_start_index:fmap_width_end_index]
          # finally update the global minimmizers
          global_min_proto_dist[j] = batch_min_proto_dist_j
          global_min_fmap_patches[j] = batch_min_fmap_patch_j 

        """
        Everything past this is visualization. Can ignore this if you don't want to worry about implementing visualization. 
        """

        if visualize: 
          # get the receptive field boundary of the image patch that generates the representation
          rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, self.model.proto_layer_rf_info)
          
          # get the whole image
          original_img_j = search_batch[rf_prototype_j[0]]
          original_img_j = original_img_j.clone().cpu().numpy()
          original_img_j = np.transpose(original_img_j, (1, 2, 0))
          original_img_size = original_img_j.shape[0]
          
          # crop out the receptive field
          rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                    rf_prototype_j[3]:rf_prototype_j[4], :]
          
          # save the prototype receptive field information
          proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
          proto_rf_boxes[j, 1] = rf_prototype_j[1]
          proto_rf_boxes[j, 2] = rf_prototype_j[2]
          proto_rf_boxes[j, 3] = rf_prototype_j[3]
          proto_rf_boxes[j, 4] = rf_prototype_j[4]
          if proto_rf_boxes.shape[1] == 6 and search_y is not None:
            proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

          # find the highly activated region of the original image
          proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
          proto_act_img_j = 128 - proto_dist_img_j
          upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                           interpolation=cv2.INTER_CUBIC)
          proto_bound_j = find_high_activation_crop(upsampled_act_img_j)

          # save the prototype boundary (rectangular boundary of highly activated region)
          proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
          proto_bound_boxes[j, 1] = proto_bound_j[0]
          proto_bound_boxes[j, 2] = proto_bound_j[1]
          proto_bound_boxes[j, 3] = proto_bound_j[2]
          proto_bound_boxes[j, 4] = proto_bound_j[3]
          proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

          # overlay (upsampled) self activation on original image and save the result
          rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
          rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
          heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
          heatmap = np.float32(heatmap) / 255
          heatmap = heatmap[...,::-1]
          overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
          # plt.imsave(os.path.join("saved", f"prototype_{j}.png"), overlayed_original_img_j, vmin=0.0, vmax=1.0)
          wandb.log({f"prototype_{j}": wandb.Image(overlayed_original_img_j, caption=f"Prototype {j}")})
                    
    # now that we have all patches for which prototypes should get pushed to, push. 
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    self.model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
