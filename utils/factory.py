from models.protocl import PROTOCL

def get_model(model_name, args):
    name = model_name.lower()
    if name == 'icarl':
        return iCaRL(args)
    elif name == 'end2end':
        return End2End(args)
    elif name == 'dr':
        return DR(args)
    elif name == 'ucir':
        return UCIR(args)
    elif name == 'bic':
        return BiC(args)
    elif name == 'lwm':
        return LwM(args)
    elif name == 'protocl':
        return PROTOCL(args)
