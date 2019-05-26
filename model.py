from modeltrain import *
import random
import time

from sklearn.metrics import roc_auc_score

READ_PROCESS_COUNT = 3

pcs_list = []

def debug_log(msg):
    time_str = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    with open("model_cut_log.log","a") as f:
        f.write(time_str+msg+"\n")
    print(msg)

def load_allmodels():
    return_models = []
    model_dirjudge()
    model_files = [x for x in os.listdir("models") if
                        os.path.isfile(os.path.join("models", x))]
    for file_name in model_files:
        model = load_model(file_name)
        return_models.append(model)
    return return_models

class judge_process(Process):
    def __init__(self, models, in_queue, out_queue):
        super().__init__()
        self.models = models
        self.in_queue = in_queue
        self.out_queue = out_queue
    def run(self):
        while(True):
            data = self.in_queue.get()
            if data is None:
                break
            possibilitise = 0
            for model in self.models:
                possibilitise += model_judge(model, data[1]) * model["accuracy"]
            possibilitise /= len(self.models)
            self.out_queue.put([data[0], possibilitise])
        self.out_queue.put(None)
        return
    def update_models(self, models):
        self.models = models

def pcs_reset(queuelist):
    for pcs in pcs_list:
        queuelist[0].put(None)
    for pcs in pcs_list:
        pcs.join()
    pcs_list.clear()

def testdata_judge(test_filename="test.csv", models=None, print_result=True, judge_data=None, 
    msg_queue=None, create_process=True,delete_process=True):
    if models is None:
        models = load_allmodels()
    if judge_data is None:
        judge_data = read_datas(test_filename)
    results = []
    for i in range(len(judge_data)):
        results.append([0])

    # 多进程准备
    if msg_queue is not None:
        to_process_queue = msg_queue[0]
        from_process_queue = msg_queue[1]
    else:
        to_process_queue = Queue()
        from_process_queue = Queue()

    process_count = READ_PROCESS_COUNT
    if create_process:
        print("Making processes...")
        for p_idx in range(process_count):
            pcs = judge_process(models, to_process_queue, from_process_queue)
            pcs.start()
            pcs_list.append(pcs)

    print("Sending datas...")
    for data_idx in range(len(judge_data)):
        data = judge_data[data_idx]
        to_process_queue.put([data_idx, data])
    
    # 发送结束
    if delete_process:
        for p_idx in range(process_count):
            to_process_queue.put(None)
    
    # 接收信息
    recved = 0
    rest_process = process_count
    while(rest_process>0 and recved < len(judge_data)):
        recv_data = from_process_queue.get()
        if recv_data is None:
            rest_process -= 1
            continue
        recved += 1
        print("Recving results... %d/%d"%(recved, len(judge_data)), end='\r')
        results[recv_data[0]] = recv_data[1]

    if delete_process:
        for pcs in pcs_list:
            pcs.join()
        pcs_list.clear()
    # 输出
    if print_result:
        with open("result.csv",'w') as f:
            f.write("Id,label\n")
            for id in range(len(results)):
                #result = 0 if results[id][0] > results[id][1] else 1
                result = results[id]
                f.write("%d,%.6f\n"%(id, result))
    
    return results

def show_models_state(minimum=None):
    models = load_allmodels()
    models.sort(key=lambda x: -x['accuracy'])
    print("index\ttype\taccuracy")
    for model in models:
        print("%d\t%s\t%.3f"%(model["index"],model["type"],model["accuracy"]))
        if minimum is not None:
            if model["accuracy"] < minimum:
                model_name = get_model_name(model['index'])
                os.remove(model_name)

def calculate_auc(models, train_datas, queues=None, firsttime=True):
    real_result = []
    for t in train_datas:
        real_result.append(t.result)
    np_real_result = numpy.array(real_result)
    del real_result

    expected_result = testdata_judge(models=models, print_result=False, judge_data=train_datas,
        msg_queue=queues, create_process=firsttime, delete_process=False)

    np_expected_result = numpy.array(expected_result)
    del expected_result
    return roc_auc_score(np_real_result, np_expected_result)

def model_cut():
    all_models = load_allmodels()
    all_datas = read_datas()
    times = 0
    while(True):
        times += 1
        debug_log("Running the %d times of cutting...\nCurrently %d models remained..."%(times, len(all_models)))

        random.shuffle(all_models)
        next_models = []
        deleted = False
        # 选择测试数据
        test_datas_set = []
        for i in range(6):
            test_datas = []
            for j in range(5000):
                add_data = random.choice(all_datas)
                test_datas.append(add_data)
            test_datas_set.append(test_datas)
        
        old_auc_set = []
        old_firsttime = True
        old_queue = [Queue(), Queue()]
        for i in range(6):
            old_auc_set.append(calculate_auc(all_models, test_datas_set[i], old_queue, old_firsttime))
            old_firsttime = False
        pcs_reset(old_queue)

        # 删除模版测试
        for model in all_models:
            new_models = all_models.copy()
            new_models.remove(model)
            update_count = 0
            load_count = 0
            new_firsttime = True
            new_queue = [Queue(), Queue()]
            for load_count in range(6):
                new_auc = calculate_auc(new_models, test_datas_set[load_count],new_queue, new_firsttime)
                new_firsttime = False
                debug_log("AUC: %.6f/%.6f"%(new_auc, old_auc_set[load_count]))
                if new_auc > old_auc_set[load_count]:
                    update_count += 1
                # 提前结束
                if update_count > 3 or load_count - update_count > 2:
                    break
            pcs_reset(new_queue)

            # 删除
            if update_count > 3:
                debug_log("Delete %d with type %s, AUC %.3f"%(model["index"], model["type"], model["accuracy"]))
                model_name = get_model_name(model['index'])
                os.remove(model_name)
                deleted = True
            else:
                debug_log("Save %d with remove count %d, load count %d."%(model["index"], update_count, load_count))
                next_models.append(model)
        
        # 无法降低AUC则退出
        if not deleted:
            break
        
        all_models = next_models
    
    debug_log("Finish cutting.")
        
