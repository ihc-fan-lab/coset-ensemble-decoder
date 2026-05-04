import os
import numpy as np
import json
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from config import DATA_COLLECTION_CONFIG
from collections import defaultdict

class EmptyDataCollector:
    """空的数据收集器，用于禁用数据收集时使用"""
    def __init__(self, save_dir=None):
        pass
    
    def start_new_experiment(self, L, noise_level):
        pass
    
    def save_cluster_data(self, syndrome, code_structure, cluster_candidates, actual_error):
        pass
    
    def process_data_for_training(self):
        pass

class ClusterDataCollector:
    def __init__(self, save_dir=None):
        """初始化数据收集器
        Args:
            save_dir: 数据保存的根目录，如果为None则使用配置中的目录
        """
        # 设置基本属性
        self.save_dir = save_dir if save_dir is not None else DATA_COLLECTION_CONFIG['save_dir']
        self.current_experiment = None
        self._is_empty = not DATA_COLLECTION_CONFIG['enable_data_collection']
        self.data_count = 0  # 添加数据计数器
        
        # 如果不是空收集器，创建目录
        if not self._is_empty:
            self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        if self._is_empty:
            return
            
        # 创建主目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建子目录
        subdirs = ['raw_data', 'processed_data', 'models']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.save_dir, subdir), exist_ok=True)
    
    def start_new_experiment(self, L, noise_level):
        """开始新的实验记录
        Args:
            L: code distance
            noise_level: 噪声水平
        """
        if self._is_empty:
            return
            
        # 创建实验目录，格式为 L{size}_noise{level}
        experiment_name = f'L{L}_noise{noise_level}'
        self.current_experiment = experiment_name
        self.data_count = 0  # 重置计数器
        
        # 创建实验目录
        experiment_dir = os.path.join(self.save_dir, 'raw_data', experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 如果code_structure.json不存在，创建它
        code_structure_file = os.path.join(experiment_dir, 'code_structure.json')
        if not os.path.exists(code_structure_file):
            # 这里不保存code_structure，因为它在save_data方法中会被保存
            pass
    
    def save_data(self, syndrome_array, code_structure, cluster_candidates, actual_error, predicted_logicals, all_corrections, noise_level, is_zero_syndrome=False, is_training_data=False):
        """保存数据
        Args:
            syndrome_array: syndrome数组
            code_structure: 代码结构
            cluster_candidates: 候选解列表
            actual_error: 实际错误
            predicted_logicals: 预测的逻辑错误
            all_corrections: 所有修正
            noise_level: 噪声水平
            is_zero_syndrome: 是否为全0 syndrome
            is_training_data: 是否为训练数据
        """
        # 检查当前实验是否有效
        if not self.current_experiment:
            return
            
        # 构建完整的实验目录路径
        experiment_dir = os.path.join(self.save_dir, 'raw_data', self.current_experiment)
        os.makedirs(experiment_dir, exist_ok=True)
            
        # 保存code_structure信息（如果还没有保存）
        if not hasattr(self, 'code_structure_saved'):
            code_structure_path = os.path.join(experiment_dir, 'code_structure.json')
            with open(code_structure_path, 'w') as f:
                json.dump({
                    'L': code_structure.L,
                    'code_type': code_structure.code_type,
                    'periodic': code_structure.periodic
                }, f)
            self.code_structure_saved = True
            
        # 创建case目录
        case_dir = os.path.join(experiment_dir, f'case_{self.data_count}')
        os.makedirs(case_dir, exist_ok=True)
        
        # 准备case数据
        case_data = {
            'syndrome': syndrome_array.tolist(),
            'cluster_candidates': [
                candidate.tolist() if hasattr(candidate, 'tolist') else candidate 
                for candidate in cluster_candidates
            ],
            'actual_error': actual_error.tolist(),
            'predicted_logicals': [logical.tolist() for logical in predicted_logicals],
            'all_corrections': [correction.tolist() for correction in all_corrections],
            'noise_level': noise_level,
            'is_zero_syndrome': is_zero_syndrome,
            'is_training_data': is_training_data
        }
        
        # 保存case数据
        case_path = os.path.join(case_dir, 'case_data.json')
        with open(case_path, 'w') as f:
            json.dump(case_data, f)
            
        self.data_count += 1
    
    def _check_cluster_correctness(self, cluster_state, actual_error):
        """检查簇是否正确
        Args:
            cluster_state: 簇的状态
            actual_error: 实际的错误
        Returns:
            bool: 是否正确
        """
        # TODO: 实现正确性检查逻辑
        # 这里需要根据具体的解码结果和实际错误来判断
        return False
    
    def process_data_for_training(self):
        """处理收集的数据，转换为训练格式"""
        if self._is_empty:
            return
            
        processed_dir = os.path.join(self.save_dir, 'processed_data')
        os.makedirs(processed_dir, exist_ok=True)
        
        # 收集所有实验数据
        all_data = []
        raw_data_dir = os.path.join(self.save_dir, 'raw_data')
        
        # 遍历所有实验目录
        for experiment in os.listdir(raw_data_dir):
            experiment_dir = os.path.join(raw_data_dir, experiment)
            
            # 读取code_structure
            with open(os.path.join(experiment_dir, 'code_structure.json'), 'r') as f:
                code_structure = json.load(f)
            
            # 遍历所有case
            for case in os.listdir(experiment_dir):
                if not case.startswith('case_'):
                    continue
                    
                case_dir = os.path.join(experiment_dir, case)
                case_file = os.path.join(case_dir, 'case_data.json')
                
                try:
                    with open(case_file, 'r') as f:
                        case_data = json.load(f)
                    
                    # 处理每个候选解
                    for cluster in case_data['cluster_candidates']:
                        # 构建特征向量
                        features = self._extract_features(cluster, case_data['syndrome'])
                        label = 1 if cluster['is_correct'] else 0
                        
                        all_data.append({
                            'features': features,
                            'label': label,
                            'experiment': experiment,
                            'case': case,
                            'code_structure': code_structure,
                            'noise_level': case_data['noise_level']  # 添加noise_level
                        })
                except Exception as e:
                    print(f"Warning: Error processing case {case}: {str(e)}")
                    continue
        
        # 保存处理后的数据
        if all_data:
            torch.save(all_data, os.path.join(processed_dir, 'processed_data.pt'))
            print(f"Successfully processed {len(all_data)} data points")
        else:
            print("Warning: No data to process")
    
    def _extract_features(self, cluster, syndrome):
        """从cluster数据中提取特征
        Args:
            cluster: 簇的信息
            syndrome: syndrome数据
        Returns:
            list: 特征向量
        """
        features = []
        
        # 1. 基本拓扑特征
        # 簇的大小（边数）
        num_edges = len(cluster['erasure'])
        features.append(num_edges)
        
        # 边界点集合
        boundary_points = set()
        for edge in cluster['erasure']:
            boundary_points.add(edge[0])
            boundary_points.add(edge[1])
        num_boundary_points = len(boundary_points)
        features.append(num_boundary_points)
        
        # 2. 连通性特征
        # 构建邻接图
        adjacency = defaultdict(set)
        for edge in cluster['erasure']:
            adjacency[edge[0]].add(edge[1])
            adjacency[edge[1]].add(edge[0])
        
        # 计算每个点的度数
        degrees = [len(adjacency[p]) for p in boundary_points]
        features.append(np.mean(degrees))  # 平均度数
        features.append(np.std(degrees))   # 度数的标准差
        
        # 3. 拓扑不变量
        # 欧拉示性数 = 顶点数 - 边数
        euler_characteristic = num_boundary_points - num_edges
        features.append(euler_characteristic)
        
        # 4. Syndrome相关特征
        syndrome_points = set(i for i, v in enumerate(syndrome) if v == 1)
        matched_points = syndrome_points.intersection(boundary_points)
        
        # 计算syndrome匹配的拓扑特性
        if syndrome_points:
            # syndrome匹配比例
            match_ratio = len(matched_points) / len(syndrome_points)
            features.append(match_ratio)
            
            # syndrome点的分布特征
            if matched_points:
                # 计算syndrome点之间的平均距离
                distances = []
                matched_list = list(matched_points)
                for i in range(len(matched_list)):
                    for j in range(i+1, len(matched_list)):
                        p1, p2 = matched_list[i], matched_list[j]
                        # 使用曼哈顿距离
                        dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
                        distances.append(dist)
                if distances:
                    features.append(np.mean(distances))
                    features.append(np.std(distances))
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
        
        # 5. 簇的几何特征（相对位置）
        if boundary_points:
            # 计算相对位置特征
            points_array = np.array(list(boundary_points))
            # 计算中心点
            center = np.mean(points_array, axis=0)
            # 计算相对位置
            relative_positions = points_array - center
            # 计算相对位置的统计特征
            features.append(np.mean(np.abs(relative_positions)))  # 平均相对距离
            features.append(np.std(np.abs(relative_positions)))   # 相对距离的标准差
        else:
            features.extend([0, 0])
        
        return features

class ClusterDataset(Dataset):
    """用于PyTorch训练的Dataset类"""
    def __init__(self, data_path):
        """初始化数据集
        Args:
            data_path: 处理后的数据文件路径
        """
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'features': torch.tensor(item['features'], dtype=torch.float32),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'experiment': item['experiment'],
            'case': item['case'],
            'code_structure': item['code_structure'],
            'noise_level': item['noise_level']
        } 