import json
from MMC.mmc_text_classifier import non_neural_knn_exp
from tools.utils import NCD, agg_by_concat_space


def prompt_load(prompt_file):
    with open(prompt_file, encoding="utf-8") as file:
        data = json.load(file)

    all_prompts = []
    all_label = []
    label = -1
    for category in data:
        label = label + 1
        for item in category.get("prompt", []):
            if item.get("prompt"):
                all_prompts.append(item["prompt"])
        for i in range(10):
            all_label.append(label)
    return all_prompts, all_label


def mmc(img_description, label_str, all_prompts, all_label):
    k = 1
    para = True
    test_list = [img_description]
    test_label = [label_str]
    # print("------------------Output test-------------------")
    # print(test_list)
    # print(test_label)
    # print(all_prompts)
    # print(all_label)
    # distance4i = KnnExpText.calc_dis_single_multi_add(all_prompts, img_description)
    pred, correct = non_neural_knn_exp("gzip", test_list, test_label, all_prompts, all_label, agg_by_concat_space, NCD, k, para)

    # print("--------------------Final output result-----------------------")
    # print(pred, correct)
    return pred, correct
