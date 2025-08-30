import json
import argparse
from typing import List, Dict, Any

def process_dataset(input_file: str, output_file: str, n_negative_samples: int = 5):
    """
    处理数据集,将JSONL格式转换为指定的JSON格式
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSON文件路径
        n_negative_samples: 每个query要生成的负样本数量
    """
    
    # 指令模板
    instruction_template = """Determine whether a product matches the user's query intent.
The product must completely satisfy the user's search query in all aspects (including product type, brand, model, attributes, etc.).
If any aspect is irrelevant or incorrect, return `False`.
Otherwise, return `True`.

Please reason and then give your response.

Query: {query}
Product: {product}"""
    
    results = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # 解析每一行JSON数据
                    data = json.loads(line.strip())
                    
                    # 检查必要字段是否存在
                    if not all(key in data for key in ['query', 'pos', 'neg']):
                        print(f"警告: 第{line_num}行缺少必要字段,跳过")
                        continue
                    
                    query = data['query']
                    pos_items = data['pos']
                    neg_items = data['neg']
                    
                    # 检查pos是否为空
                    if not pos_items:
                        print(f"警告: 第{line_num}行pos为空,跳过")
                        continue
                    
                    # 生成正样本（使用pos的第一项）
                    positive_sample = {
                        "instruction": instruction_template.format(
                            query=query,
                            product=pos_items[0]
                        ),
                        "input": "",
                        "output": "True"
                    }
                    results.append(positive_sample)
                    
                    # 生成负样本（使用neg的前n项）
                    neg_count = min(n_negative_samples, len(neg_items))
                    for i in range(neg_count):
                        negative_sample = {
                            "instruction": instruction_template.format(
                                query=query,
                                product=neg_items[i]
                            ),
                            "input": "",
                            "output": "False"
                        }
                        results.append(negative_sample)
                    
                    if line_num % 100 == 0:
                        print(f"已处理 {line_num} 行数据...")
                        
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON格式错误,跳过: {e}")
                    continue
                except Exception as e:
                    print(f"警告: 第{line_num}行处理出错,跳过: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        return
    except Exception as e:
        print(f"错误: 读取文件时出错: {e}")
        return
    
    # 写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成!")
        print(f"总共生成 {len(results)} 个样本")
        
        # 统计正负样本数量
        positive_count = sum(1 for item in results if item['output'] == 'True')
        negative_count = sum(1 for item in results if item['output'] == 'False')
        
        print(f"正样本数量: {positive_count}")
        print(f"负样本数量: {negative_count}")
        print(f"输出文件: {output_file}")
        
    except Exception as e:
        print(f"错误: 写入输出文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='将JSONL数据集转换为训练格式')
    parser.add_argument('--input_file', default='/root/autodl-tmp/external_data/bge/multi_cpr_ecom.jsonl', help='输入的JSONL文件路径')
    parser.add_argument('--output_file', default='/root/autodl-tmp/external_data/bge/multi_cpr_ecom.json', help='输出的JSON文件路径')
    parser.add_argument('-n', '--negative-samples', type=int, default=5, 
                       help='每个query生成的负样本数量 (默认: 5)')
    
    args = parser.parse_args()
    
    # 验证负样本数量
    if args.negative_samples <= 0:
        print("错误: 负样本数量必须大于0")
        return
    
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    print(f"负样本数量: {args.negative_samples}")
    print("开始处理...")
    
    process_dataset(args.input_file, args.output_file, args.negative_samples)

if __name__ == "__main__":
    main()