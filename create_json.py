import tsfel
import json

# 获取统计领域的特征集字典
features_dict = tsfel.get_features_by_domain()

# 将特征集字典保存为 JSON 文件
with open('./json/all_features.json', 'w') as json_file:
    json.dump(features_dict, json_file, indent=4)

