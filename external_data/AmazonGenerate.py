import pandas as pd
#import pyarrow.parquet as pq
import json
import random
import os
import glob
from typing import Dict, List, Any

class ESCIDataProcessor:
    # 改路径
    def __init__(self, data_path: str = "/root/autodl-tmp/external_data"):
        """
        初始化数据处理器
        Args:
            data_path: 数据文件所在路径
        """
        self.data_path = data_path
        self.queries_data = None
        self.products_data = None
        self.sources_data = None
        
    def load_parquet_files(self, pattern: str) -> pd.DataFrame:
        """
        加载多个parquet文件并合并
        Args:
            pattern: 文件匹配模式，如 'queries/*.parquet'
        Returns:
            合并后的DataFrame
        """
        file_path = os.path.join(self.data_path, pattern)
        files = glob.glob(file_path)
        
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {file_path}")
        
        dataframes = []
        for file in files:
            print(f"Loading {file}...")
            df = pd.read_parquet(file)
            dataframes.append(df)
        
        return pd.concat(dataframes, ignore_index=True)
    
    def load_datasets(self):
        """加载Parquet格式的数据集"""
        print("Loading parquet datasets...")
        
        try:
            # 尝试不同的文件路径模式
            patterns = [
                # 在子文件夹中
                "queries/*.parquet",
                "products/*.parquet",
                "sources/*.parquet",
                # 带train/test的模式
                "queries/train*.parquet",
                "products/train*.parquet",
                "sources/train*.parquet"
            ]
            
            # 加载queries数据
            queries_files = []
            for pattern in ["queries*.parquet", "queries/*.parquet", "queries/train*.parquet", "queries/test*.parquet"]:
                files = glob.glob(os.path.join(self.data_path, pattern))
                queries_files.extend(files)
            
            if queries_files:
                queries_dfs = [pd.read_parquet(f) for f in queries_files]
                self.queries_data = pd.concat(queries_dfs, ignore_index=True)
                print(f"Loaded {len(self.queries_data)} query-product pairs from {len(queries_files)} files")
            else:
                raise FileNotFoundError("No queries parquet files found")
            
            # 加载products数据
            products_files = []
            for pattern in ["products*.parquet", "products/*.parquet", "products/train*.parquet", "products/test*.parquet"]:
                files = glob.glob(os.path.join(self.data_path, pattern))
                products_files.extend(files)
            
            if products_files:
                products_dfs = [pd.read_parquet(f) for f in products_files]
                self.products_data = pd.concat(products_dfs, ignore_index=True)
                print(f"Loaded {len(self.products_data)} unique products from {len(products_files)} files")
            else:
                raise FileNotFoundError("No products parquet files found")
            
            # 加载sources数据（可选）
            sources_files = []
            for pattern in ["sources*.parquet", "sources/*.parquet", "sources/train*.parquet", "sources/test*.parquet"]:
                files = glob.glob(os.path.join(self.data_path, pattern))
                sources_files.extend(files)
            
            if sources_files:
                sources_dfs = [pd.read_parquet(f) for f in sources_files]
                self.sources_data = pd.concat(sources_dfs, ignore_index=True)
                print(f"Loaded {len(self.sources_data)} query sources from {len(sources_files)} files")
            else:
                print("No sources parquet files found (optional)")
                
        except Exception as e:
            print(f"Error loading datasets: {e}")
            print("Please make sure parquet files are in the correct location")
            print("Expected file patterns: queries*.parquet, products*.parquet, sources*.parquet")
            raise
        
    def build_query_product_pairs(self) -> pd.DataFrame:
        """构建query-商品信息对"""
        print("Building query-product pairs...")
        
        # 合并queries和products数据
        merged_data = self.queries_data.merge(
            self.products_data, 
            on='product_id', 
            how='left'
        )
        
        # 选择需要的列
        result_columns = [
            'query_id', 'query', 'product_id', 'product_title', 
            'product_description', 'product_bullet_point', 'product_brand', 
            'product_color', 'esci_label', 'product_locale_x'
        ]
        
        query_product_pairs = merged_data[result_columns].copy()
        query_product_pairs.rename(columns={'product_locale_x': 'product_locale'}, inplace=True)
        
        print(f"Built {len(query_product_pairs)} query-product pairs")
        return query_product_pairs
    
    def get_esci_label_mapping(self) -> Dict[str, bool]:
        """获取ESCI标签到匹配状态的映射"""
        return {
            'E': True,   # Exact match - 完全匹配
            'S': True,  # Substitute - 替代品（不是完全匹配）
            'C': False,  # Complement - 互补品（不是完全匹配）
            'I': False   # Irrelevant - 不相关
        }
    
    def format_training_data(self, query_product_pairs: pd.DataFrame, 
                           sample_size: int = 1000) -> List[Dict[str, Any]]:
        """格式化为训练数据格式"""
        print(f"Formatting training data (sampling {sample_size} examples)...")
        
        esci_mapping = self.get_esci_label_mapping()
        formatted_data = []
        
        # 如果样本数量超过要求，进行采样
        if len(query_product_pairs) > sample_size:
            sampled_data = query_product_pairs.sample(n=sample_size, random_state=42)
        else:
            sampled_data = query_product_pairs
        
        for _, row in sampled_data.iterrows():
            # 构建产品信息
            product_info = row['product_title']
            if pd.notna(row['product_description']) and row['product_description'].strip():
                product_info += f" - {row['product_description']}"
            if pd.notna(row['product_brand']) and row['product_brand'].strip():
                product_info = f"{row['product_brand']} {product_info}"
            
            # 获取匹配状态
            match_status = esci_mapping.get(row['esci_label'], False)
            
            formatted_item = {
                "instruction": "Determine whether a product matches the user's query intent.\nThe product must completely satisfy the user's search query in all aspects (including product type, brand, model, attributes, etc.).\nIf any aspect is irrelevant or incorrect, return `False`.\nOtherwise, return `True`.\n\nPlease reason and then give your response.\n\nQuery: {}\nProduct: {}".format(
                    row['query'], 
                    product_info
                ),
                "input": "",
                "output": str(match_status)
            }
            
            formatted_data.append(formatted_item)
        
        print(f"Formatted {len(formatted_data)} training examples")
        return formatted_data
    
    def analyze_dataset(self, query_product_pairs: pd.DataFrame):
        """分析数据集统计信息"""
        print("\n=== Dataset Analysis ===")
        print(f"Total query-product pairs: {len(query_product_pairs)}")
        print(f"Unique queries: {query_product_pairs['query_id'].nunique()}")
        print(f"Unique products: {query_product_pairs['product_id'].nunique()}")
        
        print("\nESCI Label Distribution:")
        esci_counts = query_product_pairs['esci_label'].value_counts()
        for label, count in esci_counts.items():
            label_name = {'E': 'Exact', 'S': 'Substitute', 'C': 'Complement', 'I': 'Irrelevant'}
            print(f"  {label} ({label_name.get(label, 'Unknown')}): {count} ({count/len(query_product_pairs)*100:.1f}%)")
        
        print("\nLanguage Distribution:")
        locale_counts = query_product_pairs['product_locale'].value_counts()
        for locale, count in locale_counts.items():
            print(f"  {locale}: {count}")
    
    def save_training_data(self, formatted_data: List[Dict], filename: str = "/root/autodl-tmp/external_data/esci_training_data2.json"):
        """保存训练数据到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        print(f"Training data saved to {filename}")
    
    def run_full_pipeline(self, sample_size: int = 1000, save_to_file: bool = True):
        """运行完整的数据处理流水线"""
        print("Starting ESCI data processing pipeline...")
        
        # 1. 加载数据集
        self.load_datasets()
        
        # 2. 构建query-商品信息对
        query_product_pairs = self.build_query_product_pairs()
        
        # 3. 分析数据集
        self.analyze_dataset(query_product_pairs)
        
        # 4. 格式化训练数据
        formatted_data = self.format_training_data(query_product_pairs, sample_size)
        
        # 5. 保存数据（可选）
        if save_to_file:
            self.save_training_data(formatted_data)
        
        # 6. 显示示例
        print("\n=== Sample Training Examples ===")
        for i, example in enumerate(formatted_data[:3]):
            print(f"\nExample {i+1}:")
            print(f"Instruction: {example['instruction'][:200]}...")
            print(f"Output: {example['output']}")
        
        return formatted_data, query_product_pairs

# 使用示例
def main():
    # 创建数据处理器实例
    processor = ESCIDataProcessor()
    
    # 运行完整流水线
    # sample_size: 控制输出的训练样本数量
    # save_to_file: 是否保存到文件
    formatted_data, query_product_pairs = processor.run_full_pipeline(
        # 最大就是500000了（）如果显存不够可以减少预训练数据规模
        sample_size=500000,
        save_to_file=True
    )
    
    # 你可以进一步处理formatted_data
    print(f"\nGenerated {len(formatted_data)} training examples")
    
    return formatted_data, query_product_pairs

if __name__ == "__main__":
    # 运行主函数
    formatted_data, query_product_pairs = main()