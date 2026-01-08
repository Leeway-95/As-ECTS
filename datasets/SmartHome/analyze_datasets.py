import pandas as pd
import numpy as np

# 读取数据集
illuminance_df = pd.read_csv('time_series_illuminance&bulb.csv')
temp_humidity_df = pd.read_csv('time_series_temperature&humidity.csv')

print("=== 光照&灯泡数据集分析 ===")
print(f"数据集形状: {illuminance_df.shape}")
print(f"列名: {illuminance_df.columns.tolist()}")
print("\n数值统计:")
print(illuminance_df.describe())

print("\n=== 温度&湿度数据集分析 ===")
print(f"数据集形状: {temp_humidity_df.shape}")
print(f"列名: {temp_humidity_df.columns.tolist()}")
print("\n数值统计:")
print(temp_humidity_df.describe())

# 分析光照值的分布
print("\n=== 光照值分布分析 ===")
illuminance_values = illuminance_df['illuminance'].values
print(f"光照值范围: {illuminance_values.min():.6f} - {illuminance_values.max():.6f}")
print(f"光照值均值: {illuminance_values.mean():.6f}")
print(f"光照值中位数: {np.median(illuminance_values):.6f}")

# 分析灯泡值的分布
bulb_values = illuminance_df['bulb'].values
print(f"\n灯泡值范围: {bulb_values.min():.6f} - {bulb_values.max():.6f}")
print(f"灯泡值均值: {bulb_values.mean():.6f}")
print(f"灯泡值中位数: {np.median(bulb_values):.6f}")

# 分析温度值的分布
print("\n=== 温度值分布分析 ===")
temperature_values = temp_humidity_df['temperature'].values
print(f"温度值范围: {temperature_values.min():.6f} - {temperature_values.max():.6f}")
print(f"温度值均值: {temperature_values.mean():.6f}")
print(f"温度值中位数: {np.median(temperature_values):.6f}")

# 分析湿度值的分布
humidity_values = temp_humidity_df['humidity'].values
print(f"\n湿度值范围: {humidity_values.min():.6f} - {humidity_values.max():.6f}")
print(f"湿度值均值: {humidity_values.mean():.6f}")
print(f"湿度值中位数: {np.median(humidity_values):.6f}")

# 检查时间序列特征
print("\n=== 时间序列特征 ===")
print(f"光照数据时间范围: {illuminance_df['Time'].min():.6f} - {illuminance_df['Time'].max():.6f}")
print(f"温湿度数据时间范围: {temp_humidity_df['time'].min():.6f} - {temp_humidity_df['time'].max():.6f}")

# 检查现有标注
print(f"\n光照数据现有标注数量: {illuminance_df['illuminance_label'].notna().sum()}")
print(f"灯泡数据现有标注数量: {illuminance_df['bulb_label'].notna().sum()}")
print(f"温度数据现有标注数量: {temp_humidity_df['temperature_label'].notna().sum()}")
print(f"湿度数据现有标注数量: {temp_humidity_df['humidity_label'].notna().sum()}")