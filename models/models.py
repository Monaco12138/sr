import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator
#   每个类初始化声明时均会调用，添加 models[name] = CLASS

def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
        
    else:
        model_args = model_spec['args']

    model = models[model_spec['name']](**model_args)          
    # model = models[ 'edsr' ](**model_args)        对应调用类的初始化函数
    if load_sd:
        model.load_state_dict(model_spec['sd'])

    return model
