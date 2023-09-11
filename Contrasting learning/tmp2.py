import torch

def explain_NT_Xent_loss():
    # 虚构的数据，批处理大小为 3，表示维度为 5
    batch_size = 3
    dim = 5

    # 创建虚构的正例和负例相似度矩阵
    similarity_matrix = torch.randn(2 * batch_size, 2 * batch_size)

    print(similarity_matrix)
    # 模拟左侧正例得分和右侧正例得分
    l_pos = torch.diag(similarity_matrix, batch_size)
    r_pos = torch.diag(similarity_matrix, -batch_size)
    print(l_pos)
    print(r_pos)

    # 拼接正例得分
    positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

    # 模拟负例得分
    mask_samples_from_same_repr = torch.zeros(2 * batch_size, 2 * batch_size).bool()
    for i in range(2 * batch_size):
        for j in range(2 * batch_size):
            if i != j and i < batch_size and j >= batch_size:
                mask_samples_from_same_repr[i][j] = True

    negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

    # 拼接正例和负例得分
    logits = torch.cat((positives, negatives), dim=1)

    # 温度参数
    temperature = 1.0

    # 创建全零标签张量
    labels = torch.zeros(2 * batch_size).long()

    # 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits / temperature, labels)

    # 打印各个变量的形状和损失值
    print("Shape of similarity_matrix:", similarity_matrix.shape)
    print("Shape of l_pos:", l_pos.shape)
    print("Shape of r_pos:", r_pos.shape)
    print("Shape of positives:", positives.shape)
    print("Shape of negatives:", negatives.shape)
    print("Shape of logits:", logits.shape)
    print("Loss:", loss.item())

# 调用函数以演示损失计算过程
explain_NT_Xent_loss()
