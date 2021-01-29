def print_net(net_dict):
    print("+"*30+"  MACS CACULATION TABLE  "+"+"*30)
    print("{}   {}   {}".format("Module".center(50),"Params".center(12),"MACs".center(12)))
    for key in net_dict:
        if 'aggregated_results' in key:
            continue
        module_info = net_dict[key]
        name = module_info['name'].center(50)
        params_size = str(module_info['params']).center(12)
        macs = str(module_info['macs']).center(12)
        print("{}   {}   {}".format(name,params_size,macs))
    print("+"*80)
    print("SUM UP: total params {}    total MACs {}".format(net_dict['aggregated_results']['params'],net_dict['aggregated_results']['macs']))
        