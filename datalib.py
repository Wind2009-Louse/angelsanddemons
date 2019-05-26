import time
import re
import os
import random
from multiprocessing import Process, Queue

data_spliters = [[1, 32, 46, 76, 135], [207, 209], [211,247]]
MODEL_SIZE = 76901

# 数据中含有的Tag
class TData():
    def __init__(self, tag, tag_value):
        if tag_value == "":
            self.value = 0
        else:
            self.value = float(tag_value)
        result = re.match(r"([A-Z]+)(\d)-([A-Z])-([\d])", tag)
        if result is None:
            self.name = "UNKNOWN"
            self.name_id = 0
            self.type = ""
            self.type_value = 0
        else:
            self.name = result.group(1)
            self.name_id = int(result.group(2))
            self.type = result.group(3)
            self.type_value = int(result.group(4))
    def tostring(self):
        return "%s%d-%s-%d"%(self.name, self.name_id, self.type, self.type_value)

# 数据的Data
class MData():
    '''
    data: 0~5
        data[0]: len=31
        data[1]: len=14
        data[2]: len=30
        data[3]: len=59
        data[4]: len=1
        data[5]: len=35
    date: 日期格式(时间戳)
    id: 样本的身份号(IW9123E1)
    index: 样本的编号
    tag: 样本标签（见TData）
        name
        name_id
        type
        value
    types: {"type_id": [type_value_1, type_value_2]...}
    '''
    def __init__(self, data_str):
        tags = data_str.split(",")
        tags[-1] = tags[-1][0:-1]

        self.index = int(tags[0])
        self.data = []
        self.types = {}
        self.date = time.mktime(time.strptime(tags[209],"%Y-%m-%d-%H.%M.%S.000000"))
        self.id = tags[210]
        self.tag = TData(tags[247], tags[248])

        self.result = None

        for spliter in data_spliters:
            last_spliter = spliter[0]
            # 分割
            for spliter in spliter[1:]:
                self.data.append(list(map(lambda x: float(x) if x != "" else 0.0 ,tags[last_spliter:spliter])))
                last_spliter = spliter

        self.data[0][0] /= 1000000000
        self.data[1][0] /= 1000000000
        self.data[3][11] /= 1000000000
        self.data[0][0] -= 128
        self.data[1][0] -= 128
        self.data[3][11] -= 128

        # 读取标签
        for types_count in range(24):
            type_name = tags[types_count * 3 + 135]
            if type_name == "":
                continue
            data_1 = float(tags[types_count * 3 + 136])
            data_2 = float(tags[types_count * 3 + 136])
            self.types[type_name] = [data_1, data_2]

class read_datas_process(Process):
    def __init__(self,in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
    def run(self):
        while(True):
            data_list = self.in_queue.get()
            if data_list is None:
                break
            data = MData(data_list)
            self.out_queue.put(data)
        self.out_queue.put(None)
        return

def read_datas(filename="train.csv"):
    if not os.path.exists("data"):
        os.mkdir("data")
    full_filename = os.path.join("data",filename)
    if not os.path.exists(os.path.join("data",filename)):
        error_msg = "Unable to open %s!"%full_filename
        err = input(error_msg)
        raise Exception(error_msg)

    print("Reading datas from %s..."%filename)
    process_count = 3

    # 新建读取线程
    to_process_queue = Queue()
    from_process_queue = Queue()

    processes = []
    for p_count in range(process_count):
        pcs = read_datas_process(to_process_queue, from_process_queue)
        pcs.start()
        processes.append(pcs)

    # 读取值
    print("Sending datas...")
    read_data_count = 0
    with open(os.path.join("data",filename),"r") as f:
        while(True):
            data_str = f.readline()
            if not data_str:
                break
            to_process_queue.put(data_str)
            read_data_count += 1
    # 提示结束
    for p_count in range(process_count):
        to_process_queue.put(None)

    # 获取数据
    rest_count = process_count
    recv_count = read_data_count
    train_datas = [None]*read_data_count
    while(rest_count>0):
        datas = from_process_queue.get()
        if datas is None:
            rest_count -= 1
            continue
        recv_count -= 1
        print("recving datas, remain %d ..."%recv_count, end="\r")
        idx = datas.index
        train_datas[idx] = datas

    # 读取数据
    if filename[0:5] == "train":
        label_name = "label"+filename[5:]
        with open(os.path.join("data",label_name),"r") as f:
            f.readline()
            while(True):
                data_str = f.readline()
                if not data_str:
                    break
                data_str = data_str.split(",")
                data_index = int(data_str[0])
                data_result = int(data_str[1])
                train_datas[data_index].result = data_result
                
    return train_datas