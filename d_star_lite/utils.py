def stateNameToCoords(name):
    # 将节点名称分割并提取x和y坐标
    return [int(name.split('x')[1].split('y')[0]), int(name.split('x')[1].split('y')[1])]
