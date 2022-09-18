import torch
from pytorch3d import transforms


def transform_object_model(model, rot_mat, trans):
    t1 = transforms.Transform3d().rotate(rot_mat).translate(trans).to(model.device)
    return t1.transform_points(model)


def prepare_model_from_pointcloud(end_points):
    model = end_points["xyz"]
    translate_gt = end_points["translate_label"]
    translate_pred = end_points["translate_pred"]
    rot_mat_gt = transforms.axis_angle_to_matrix(end_points["axag_label"])
    rot_mat_pred = transforms.axis_angle_to_matrix(end_points["axag_pred"])
    model_gt = transform_object_model(model, rot_mat_gt, translate_gt)
    model_pred = transform_object_model(model, rot_mat_pred, translate_pred)
    return model_gt, model_pred


def get_ADD_loss(model_gt, model_pred):
    diff = model_gt - model_pred
    distances = torch.norm(diff, dim=2)
    return torch.mean(distances, dim=1)


def get_ADS_loss(model_gt, model_pred):
    I2, I1 = torch.meshgrid(
        torch.arange(model_gt.shape[1]),
        torch.arange(model_pred.shape[1]),
        indexing="xy",
    )
    # Gather values at all the indexes for each batch
    # torch.gather does not work
    points_gt = model_gt[:, I1.reshape((-1,))]
    points_pred = model_pred[:, I2.reshape((-1,))]
    distances = torch.norm(points_gt - points_pred, dim=2)
    distances = distances.reshape((-1, model_gt.shape[1], model_pred.shape[1]))
    distances, _ = torch.min(distances, dim=2)
    return torch.mean(distances, dim=1)


def get_ADD_ADS(end_points, point_class):
    ad_loss = {}
    ads_loss = {}

    model_gt, model_pred = prepare_model_from_pointcloud(end_points)
    ad_loss["per"] = get_ADD_loss(model_gt, model_pred)
    ads_loss["per"] = get_ADS_loss(model_gt, model_pred)
    ad_loss["total"] = torch.mean(ad_loss["per"])
    ads_loss["total"] = torch.mean(ads_loss["per"])
    ad_loss["perCls"] = torch.unsqueeze(ad_loss["per"], dim=0).t() * point_class
    ads_loss["perCls"] = torch.unsqueeze(ads_loss["per"], dim=0).t() * point_class
    ad_loss["cls"] = torch.sum(ad_loss["perCls"], dim=0) / torch.sum(point_class, dim=0)
    ads_loss["cls"] = torch.sum(ads_loss["perCls"], dim=0) / torch.sum(
        point_class, dim=0
    )
    return ad_loss, ads_loss
