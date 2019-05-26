from datalib import *
import os
import json
import numpy
import scipy
import pickle
import base64
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

'''
model(dict):
    index(int): 模型编号
    type(str): 模型类型
    subtype(int): 模型子类型
    values(list/dict): 参数
    accuracy(float): 模型的可信度
'''

# 不同模型出现的概率

PICKLE_TYPE = ["DT_0", "DT_2", "DT_3","DT_5"]

MODEL_TYPES = {
    "DT_0":10,
	"DT_2":10,
    "DT_3":10,
    "SKLR_0":5,
    "date":5,}
'''
MODEL_TYPES = {
    "id_poss": 30,
    "LR_0": 31,
    "LR_1": 14,
    "LR_2": 30,
    "LR_3": 59,
    "LR_4": 1,
    "LR_5": 35,
    "date": 10,
    "tag": 5,
    "types": 5}
'''

def get_model_types():
    total_model_types = sum(MODEL_TYPES.values())
    type_rand = random.randint(0,total_model_types-1)
    for model_name, model_poss in MODEL_TYPES.items():
        if type_rand < model_poss:
            return model_name
        else:
            type_rand -= model_poss

def default_model():
    while(True):
        model_index = int(random.random() * 10000)
        model_name = get_model_name(model_index)
        if os.path.exists(model_name):
            continue
        return {
            "index": model_index,
            "accuracy": 0,
            "type": get_model_types(),
            "subtype": None,
            "value": None}

def model_dirjudge():
    os.chdir(os.getcwd())
    file_dir = os.path.join(os.getcwd(),"models")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

def get_model_name(index):
    model_dirjudge()
    if type(index) == type(1):
        return os.path.join(os.getcwd(),"models","model_%d.json"%index)
    else:
        return os.path.join(os.getcwd(),"models","%s"%index)

def load_model(index):
    full_path = get_model_name(index)
    if os.path.exists(full_path):
        fjson = open(full_path)
        text = fjson.read()
        fjson.close()
        data = json.loads(text)
        if data["type"] in PICKLE_TYPE:
            model_byte = base64.b64decode(data["value"].encode('utf-8'))
            data["value"] = pickle.loads(model_byte)
        return data

def save_model(model):
    if model["type"] in PICKLE_TYPE:
        model["value"] = str(base64.b64encode(pickle.dumps(model["value"])),'utf-8')
    model_json = json.dumps(model, indent=2)
    fullname = get_model_name(model["index"])
    new_file = open(fullname, 'w')
    new_file.write(model_json)
    new_file.close()

def model_judge(model, sample):
    if model["type"][0:2] == "LR" or model["type"][0:4] == "SKLR":
        rl_type = int(model["type"][-1])
        np_rl_value = numpy.array(model["value"])
        sample_value = sample.data[rl_type]
        sample_value.append(1)
        np_sample_value = numpy.array(sample_value)
        sample_value.pop()
        result = np_sigmoid(np_sample_value.dot(np_rl_value))
        if model["type"][0:2] == "LR":
            result += model["subtype"]
        return result
    if model["type"] == "id_poss":
        id = sample.id
        if id not in model["value"]:
            id = "None"
        poss = model["value"][id]
        return poss
    if model["type"] == "date":
        for model_idx in range(len(model["value"])-1):
            this_timestep = model["value"][model_idx]
            next_timestep = model["value"][model_idx+1]
            # 在第一个时间段之前
            if model_idx == 0 and sample.date < this_timestep[0]:
                return this_timestep[2]
            # 在这个时间段中
            if sample.date >= this_timestep[0] and sample.date <= this_timestep[1]:
                return this_timestep[2]
            # 在这个时间段和下一个时间段之间
            if sample.date > this_timestep[1] and sample.date < next_timestep[0]:
                # 比值
                this_dest = sample.date - this_timestep[1]
                next_dest = next_timestep[0] - sample.date
                result = this_timestep[2] * this_dest / (this_dest + next_dest)
                result += next_timestep[2] * next_dest / (this_dest + next_dest)
                return result
        # 遍历结束，选择最后一个
        return model["value"][-1][2]
    if model["type"] == "tag":
        true_chance = None
        sample_tag_str = sample.tag.tostring()
        if sample_tag_str not in model["value"].keys():
            sample_tag_str = 'UNKNOWN0--0'
        if sample_tag_str not in model["value"].keys():
            true_chance = model["subtype"]
        else:
            value_line = model["value"][sample_tag_str]
            if sample.tag.value <= value_line[0][0]:
                true_chance = value_line[0][1]
            else:
                for idx in range(len(value_line)-1):
                    if sample.tag.value >= value_line[idx][0] and sample.tag.value < value_line[idx][0]:
                        this_dest = sample.tag.value - value_line[idx][0]
                        next_dest = value_line[idx+1][0] - sample.tag.value
                        if this_dest > next_dest:
                            true_chance = value_line[idx][1]
                        else:
                            true_chance = value_line[idx+1][1]
                        break
                if true_chance is None:
                    true_chance = value_line[len(value_line)-1][1]
        return true_chance#1 if random_result < true_chance else 0
    if model["type"][0:2] == "DT":
        data = int(model["type"][-1])
        sample_value = sample.data[data]
        np_sample_value = numpy.array([sample_value])
        try:
            prediction = model["value"].predict(np_sample_value)
        except:
            print("Error while predicting:")
            print("Type: %s"%model["type"])
            print("Value: %s"%model["value"])
        return prediction.tolist()[0]

def np_sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

class train_process(Process):
    def __init__(self, samples, tests):
        super().__init__()
        self.samples = samples
        self.tests = tests
        self.model = default_model()
        # 结果类型，用来做权重
        self.zero_count = 0
        self.one_count = 0
        for m in self.samples:
            if m.result == 0:
                self.zero_count += 1
            elif m.result == 1:
                self.one_count += 1
            else:
                raise Exception("Error")
        if self.zero_count * self.one_count == 0:
            raise Exception("Error")
    def run(self):
        print("Index: %d, Type: %s"%(self.model["index"],self.model["type"]))
        # 根据模型类型选择不同的训练方式
        if self.model["type"][0:4] == "SKLR":
            self.train_rl_sklearn()
        elif self.model["type"][0:2] == "LR":
            self.train_rl()
        elif self.model["type"][0:2] == "DT":
            self.train_DT()
        elif self.model["type"] == "id_poss":
            self.train_id_poss()
        elif self.model["type"] == "date":
            self.train_date()
        elif self.model["type"] == "tag":
            self.train_tag()
        else:
            print("Invalid model type: %s"%self.model["type"])
        # AUC判断
        self.get_accuracy()
        # 保存模型
        save_model(self.model)
        print("Made model %d with AUC %.3f."%(self.model["index"], self.model["accuracy"]))
        return

    def get_accuracy(self):
        real_result = []
        expected_result = []
        for t in self.tests:
            expected_result.append(model_judge(self.model, t))
            real_result.append(t.result)
        np_real_result = numpy.array(real_result)
        np_expected_result = numpy.array(expected_result)
        del real_result
        del expected_result
        self.model["accuracy"] = roc_auc_score(np_real_result, np_expected_result)        

    def train_rl_sklearn(self):
        # 初始化
        self.model["subtype"] = self.one_count / len(self.samples)-0.5
        rl_type = int(self.model["type"][-1])

        # 样本矩阵
        matrix = []
        for s in self.samples:
            matrix.append(s.data[rl_type].copy())
        np_matrix = numpy.array(matrix)
        del matrix

        # 结果矩阵
        result_matrix = numpy.array(list(s.result for s in self.samples))

        # 定义模型
        rl_model = LogisticRegression(solver="sag", max_iter=1000000)

        rl_model.fit(np_matrix, result_matrix)

        self.model["value"] = rl_model.coef_[0].tolist()
        self.model["value"].append(rl_model.intercept_.tolist()[0])

    def train_rl(self):
        # 初始化
        self.model["subtype"] = self.one_count / len(self.samples)-0.5
        rl_type = int(self.model["type"][-1])
        rl_values = [random.random()]*len(self.samples[0].data[rl_type])
        rl_values.append(1)

        # 样本矩阵
        matrix = []
        for s in self.samples:
            matrix.append(s.data[rl_type].copy())
            matrix[-1].append(1)
        np_matrix = numpy.array(matrix)
        del matrix

        # 结果矩阵
        result_matrix = numpy.array(list(s.result for s in self.samples))

        shift_result = self.one_count/len(self.samples)-0.5
        # 10W次调用
        for times in range(100000):
            # 计算wT
            w_T = numpy.dot(np_matrix, rl_values)
            # 计算每个样本的sigmoid值
            sigmoided = np_sigmoid(w_T)
            '''
            expect_result = numpy.around(sigmoided+shift_result)
            # 全部判断正确，提前结束
            if (expect_result==result_matrix).all():
                break
            '''

            # 错误率
            error_rate = sigmoided - result_matrix

            rl_values -= numpy.dot(np_matrix.transpose(),error_rate) * 0.00001

        # 输出训练参数
        self.model["value"] = rl_values.tolist()

    def train_id_poss(self):
        result_count = {}
        for s in self.samples:
            s_id = s.id
            if s_id not in result_count:
                result_count[s_id] = [0,0]
            result_count[s_id][s.result] += 1
        result_count["None"] = [self.zero_count, self.one_count]
        results = {}
        for result_id in result_count.keys():
            total_sum = sum(result_count[result_id])
            results[result_id] = result_count[result_id][1] / total_sum
        self.model["value"] = results

    def train_date(self):
        timeline = []
        for sample in self.samples:
            timeline.append([sample.date, sample.result])
        timeline.sort(key=lambda x: x[0])
        timecore = []
        current_result = None
        time_begins = None
        time_ends = None
        for tp in timeline:
            if current_result is None:
                time_begins = tp[0]
                current_result = tp[1]
            # 新一段，清空数据
            if current_result != tp[1]:
                timecore.append([time_begins, time_ends, current_result])
                time_begins = tp[0]
                current_result = tp[1]
            time_ends = tp[0]
        timecore.append([time_begins, time_ends, current_result])
        self.model["value"] = timecore

    def train_tag(self):
        self.model["subtype"] = self.one_count / len(self.samples)
        tag_results = {}
        tag_collection = {}
        for s in self.samples:
            tag_str = s.tag.tostring()
            tag_collection.setdefault(tag_str,{})
            tag_collection[tag_str].setdefault(s.tag.value,[0,0])
            tag_collection[tag_str][s.tag.value][s.result] += 1
        for tag in tag_collection.keys():
            tag_results.setdefault(tag,[])
            for t_value in tag_collection[tag].keys():
                tag_collection[tag][t_value][1] *= self.zero_count
                tag_collection[tag][t_value][0] *= self.one_count
                t_v_poss = tag_collection[tag][t_value][1] / (tag_collection[tag][t_value][0] + tag_collection[tag][t_value][1])
                tag_results[tag].append([t_value, t_v_poss])
            tag_results[tag].sort(key=lambda x: x[0])
           
        self.model["value"] = tag_results

    def train_DT(self):
        # 初始化
        self.model["subtype"] = self.one_count / len(self.samples)
        data_type = int(self.model["type"][-1])

        # 样本矩阵
        matrix = []
        for s in self.samples:
            matrix.append(s.data[data_type].copy())
        np_matrix = numpy.array(matrix)
        del matrix

        # 结果矩阵
        result_matrix = numpy.array(list(s.result for s in self.samples))

        # 定义模型
        if random.random() < 0.5:
            DT_model = DecisionTreeClassifier(criterion="entropy")
        else:
            DT_model = DecisionTreeClassifier(criterion="gini")

        DT_model.fit(np_matrix, result_matrix)
        self.model["value"] = DT_model

def get_train_models(filename="train.csv"):
    # 读取数据
    datas = read_datas(filename)
    datas_size = len(datas)
    
    print("Making processes to make models...")
    process_list = []
    
    p_idx = 0
    for huge_range in range(15):
        for p_id in range(2):
            # 抽取数据
            train_samples = []
            test_samples = []
            for train_idx in range(20000):
                index = random.randint(0, datas_size-1)
                train_samples.append(datas[index])
            for test_idx in range(10000):
                index = random.randint(0, datas_size-1)
                test_samples.append(datas[index])
            
            # 加入到进程中
            test_p = train_process(train_samples, test_samples)
            test_p.start()
            process_list.append(test_p)
        
        for p in process_list:
            p.join()
        process_list.clear()