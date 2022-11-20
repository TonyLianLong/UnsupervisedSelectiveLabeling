# %%
import os
os.environ["USL_MODE"] = "USL"

import numpy as np
import torch
import torch.nn.functional as F
import utils
from utils import cfg, logger, print_b
import torchvision.models as models

utils.init(default_config_file="configs/ImageNet_usl_moco_0.2.yaml")

logger.info(cfg)

# %%
print_b("Loading model")

clip = cfg.MODEL.ARCH.startswith("CLIP-")

if not clip:
    checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)
    model = models.__dict__[cfg.MODEL.ARCH](pretrained=False)
    model.fc = utils.build_hidden_head(num_mlp=2, dim_mlp=2048, dim=128, normed=True)
    def single_model_encoder_q(state_dict):
        search_str = "module.encoder_q."
        return {k.replace(search_str, ""): v for k, v in state_dict.items() if search_str in k}
    model.load_state_dict(single_model_encoder_q(checkpoint["state_dict"]))
    model.cuda()
    model.eval()
    transform_override = None
else:
    model_name = cfg.MODEL.ARCH[5:]
    model, transform_override = utils.get_clip_model(model_name)

# %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in [
    "imagenet", "imagenet100"], f"{cfg.DATASET.NAME} is not imagenet or imagenet100"
imagenet100 = cfg.DATASET.NAME == "imagenet100"
num_classes = 100 if imagenet100 else 1000

train_memory_dataset, train_memory_loader = utils.train_memory_imagenet(
    transform_name=cfg.DATASET.TRANSFORM_NAME,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS,
    transform_override=transform_override)

targets = torch.tensor(train_memory_dataset.targets)
targets.shape

# %%
print_b("Loading feat list")
feats_list = utils.get_feats_list(
    model, train_memory_loader, recompute=cfg.RECOMPUTE_ALL, CLIP=clip, force_no_extra_kwargs=True)

if clip:
    feats_list = F.normalize(feats_list, dim=1)

# %%
print_b("Calculating first order kNN density estimation")
d_knns, ind_knns = utils.partitioned_kNN(
    feats_list, K=cfg.USL.KNN_K, recompute=cfg.RECOMPUTE_ALL)
neighbors_dist = d_knns.mean(dim=1)
score_first_order = 1/neighbors_dist

# %%
final_sample_num = cfg.USL.NUM_SELECTED_SAMPLES
if imagenet100:
    num_centroids, chosen_percent = utils.get_sample_info_imagenet100(
    final_sample_num=final_sample_num)
else:
    num_centroids, chosen_percent = utils.get_sample_info_imagenet(
        final_sample_num=final_sample_num)
logger.info("num_centroids: {}, final_sample_num: {}".format(
    num_centroids, final_sample_num))

# %%
recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP
for kMeans_seed in cfg.USL.SEEDS:
    print_b(f"Running k-Means with seed {kMeans_seed}")

    # This has side-effect: it calls torch.manual_seed to ensure the seed in k-Means is set.
    # Note: NaN in centroids happens when there is no corresponding sample which belongs to the centroid
    cluster_labels, centroids = utils.run_kMeans(feats_list, num_centroids, final_sample_num, Niter=cfg.USL.K_MEANS_NITERS,
                                                 recompute=recompute_num_dependent, seed=kMeans_seed, force_no_lazy_tensor=False)

    print_b("Getting selections")
    if cfg.USL.REG.NITERS > 1:
        get_selection_fn = utils.get_selection_with_reg_imagenet
    else:
        get_selection_fn = utils.get_selection_without_reg
    
    selected_inds = utils.get_selection(get_selection_fn, feats_list, neighbors_dist, cluster_labels, num_centroids, final_sample_num=final_sample_num, iters=cfg.USL.REG.NITERS, w=cfg.USL.REG.W,
                                        momentum=cfg.USL.REG.MOMENTUM, horizon_num=cfg.USL.REG.HORIZON_NUM, alpha=cfg.USL.REG.ALPHA, exclude_same_cluster=cfg.USL.REG.EXCLUDE_SAME_CLUSTER, verbose=True, seed=kMeans_seed, recompute=True, save=True)

    counts = np.bincount(np.array(train_memory_dataset.targets)[selected_inds])

    print("Class counts:", sum(counts > 0))
    print(counts.tolist())

    print("max: {}, min: {}".format(counts.max(), counts.min()))

    print("Number of selected indices:", len(selected_inds))
    print("Selected IDs:")
    print(repr(selected_inds))

    ours_filename_part = cfg.RUN_NAME if kMeans_seed == 0 else f"{cfg.RUN_NAME}_seed{kMeans_seed}"
    utils.save_data(gen_mode="ours", stratified_density_selected_data_output=selected_inds,
         ours_filename_part=ours_filename_part, feats_list=feats_list, final_sample_num=final_sample_num, 
         chosen_percent=chosen_percent, train_memory_dataset=train_memory_dataset)
