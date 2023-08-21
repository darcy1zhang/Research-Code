import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train_sig = np.load("./data/train_without_resp.npy")

# 载入数据，特征矩阵 X 和目标向量 y
X = np.load("./data/train_new_features.npy")
y = train_sig[:, 1004]

# 分割训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义基因操作（加法、减法、乘法、除法）
# operations = ['+', '-', '*', '/']
# 0 不动 1 * 2 /

# 定义遗传算法参数
population_size = 100
num_generations = 50
mutation_rate = 0.1


def evaluate_fitness(individual):
    # 根据操作序列 individual 构建特征组合
    feature_combination = build_feature_combination(X, individual)
    print(feature_combination.shape)
    # 训练模型并计算适应度（例如，使用线性回归模型）
    model = LinearRegression()
    model.fit(feature_combination, y)
    predictions = model.predict(feature_combination)
    fitness = np.mean(np.abs(predictions - y))  # mae作为适应度

    return fitness


def build_feature_combination(X, individual):
    # 根据操作序列 individual 构建特征组合
    flag = 0

    for i in range(len(individual)):
        if individual[i] == 0:
            continue
        elif individual[i] == 1:
            if flag == 0:
                combined_feature = X[:, i]
                flag = 1
            else:
                combined_feature = combined_feature * X[:, i]
        else:
            if flag == 0:
                combined_feature = 1 / X[:, i]
                flag = 1
            else:
                combined_feature = combined_feature / X[:, i]

    return combined_feature.reshape(-1, 1)


def genetic_algorithm():
    # 初始化种群
    population = np.random.randint(0, 3, size=(population_size, len(X[0])))

    for generation in range(num_generations):
        # 计算适应度
        fitness_scores = [evaluate_fitness(individual) for individual in population]

        # 选择父代个体
        parents = np.argsort(fitness_scores)[:population_size // 2]

        # 创建子代个体
        children = []
        for i in range(population_size):
            parent1 = population[np.random.choice(parents)]
            parent2 = population[np.random.choice(parents)]
            child = np.where(np.random.rand(len(X[0])) < 0.5, parent1, parent2)

            # 变异操作
            if np.random.rand() < mutation_rate:
                mutated_gene = np.random.randint(0, 3)
                mutation_position = np.random.randint(len(X[0]))
                child[mutation_position] = mutated_gene

            children.append(child)

        population = np.array(children)

    # 找到最优个体
    best_individual = population[np.argmin(fitness_scores)]

    return best_individual


best_individual = genetic_algorithm()

# 使用最优个体生成特征组合并进行预测
best_feature_combination = build_feature_combination(X, best_individual)
model = LinearRegression()
model.fit(best_feature_combination, y)
predictions = model.predict(best_feature_combination)
