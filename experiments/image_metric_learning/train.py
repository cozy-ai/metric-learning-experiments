from torch.utils.data import random_split
from torch.utils.data import DataLoader

## @todo move cfg updates to the package (get_config?)
## @todo test on test dataset

CFG = './cfgs/cfg_siamese.json'

if __name__ == '__main__':
    import sys
    try:
        import cozyai
    except ModuleNotFoundError:
        sys.path.append('../..')
        import cozyai
    from cozyai.utils import get_config, get_optimizer, get_scheduler
    from cozyai.data import get_transforms, get_dataset, get_dataset_wrapper
    from cozyai.models import get_model, get_model_wrapper

    cfg = get_config(CFG)
    print(cfg)

    train_transforms = get_transforms(cfg.train_transforms)
    eval_transforms = get_transforms(cfg.eval_transforms)

    train_dataset = get_dataset(cfg.dataset, train=True, transform=train_transforms, **cfg.dataset_args)
    test_dataset = get_dataset(cfg.dataset, train=False, transform=eval_transforms, **cfg.dataset_args)

    n_val = len(train_dataset)//5
    train_dataset, val_dataset = random_split(train_dataset, [len(train_dataset)-n_val, n_val])
    val_dataset.transform=eval_transforms

    print(f'Train Dataset : {len(train_dataset)}, Valid Dataset : {len(val_dataset)}, Test Dataset : {len(test_dataset)}')

    dataset_wrapper = get_dataset_wrapper(cfg.dataset_wrapper)
    
    train_dataset = dataset_wrapper(train_dataset, **cfg.dataset_wrapper_args)
    val_dataset = dataset_wrapper(val_dataset, **cfg.dataset_wrapper_args)
    test_dataset = dataset_wrapper(test_dataset, **cfg.dataset_wrapper_args)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    val_loader =  DataLoader(val_dataset, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader =  DataLoader(test_dataset, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    cfg.neck_args['out_dim'] = cfg.embed_dim
    #model = BaseEncoder(cfg.backbone, cfg.backbone_args, cfg.neck, cfg.neck_args)#, cfg.flatten_args)
    model = get_model(cfg.model, cfg.backbone, cfg.backbone_args, cfg.neck, cfg.neck_args)#, cfg.flatten_args)
    print('# Total params : %d'%(sum(p.numel() for p in model.parameters())))

    model_wrapper = get_model_wrapper(cfg.model_wrapper)
    for criteria in ["criteria_score", "criteria_loss"]:
        cfg.model_wrapper_args[criteria] = getattr(cfg, criteria) if hasattr(cfg, criteria) else None
    model = model_wrapper(model=model,
                          embed_dim=cfg.embed_dim,
                          model_name=cfg.model_name,
                          dist_metric=cfg.dist_metric,
                          device=cfg.device,
                          log_path=cfg.log_path,
                          criterion=cfg.criterion,
                          eval_metrics=cfg.eval_metrics,
                          **cfg.model_wrapper_args)
    cfg.save_json(f'{cfg.log_path}/{model.model_name}/cfg.json')

    optimizer = get_optimizer(cfg.optimizer, model, **cfg.optimizer_args)
    
    if hasattr(cfg, 'scheduler'):
        scheduler = get_scheduler(cfg.scheduler, optimizer, **cfg.scheduler_args)
    else:
        scheduler=None

    model.run_train(train_loader=train_loader, optimizer=optimizer, epoch=cfg.epoch, val_loader=val_loader, scheduler=scheduler)