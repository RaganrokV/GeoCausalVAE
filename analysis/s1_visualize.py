"""Descriptive GIS maps; preserves original plotting sections with audited data handling."""
import argparse
import json
import re
from release_utils import INPUT_ROOT, output_path, read_map, clip_map, save_figure, MapInputUnavailable

#%%
import os
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import matplotlib as mpl
import numpy as np
from shapely.geometry import box
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC', 'STSong']
# 绘制边界
bounds = [120.8, 30.67, 122, 31.9]
bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])

def plot_roads():
    #%% 读取道路和交叉口图层

    gpkg_path =  str(INPUT_ROOT / "mapdata/shanghai_road_network.gpkg")

    # 
    edges = read_map(gpkg_path, layer='edges')
    # Unused node layer is not required.
    fig, ax = plt.subplots(figsize=(15, 15))




    # 定义道路等级颜色映射
    highway_colors = {
        'motorway': '#e41a1c',    # 红色 - 高速公路
        'trunk': '#377eb8',        # 蓝色 - 主干道
        'primary': '#4daf4a',      # 绿色 - 主要道路
        'secondary': '#984ea3',    # 紫色 - 次要道路
        'tertiary': '#ff7f00',     # 橙色 - 三级道路
        'residential': '#a65628',  # 棕色 - 住宅区道路
        'service': '#999999',      # 灰色 - 服务道路
        'unclassified': '#cccccc'  # 浅灰 - 未分类道路
    }

    if 'highway' not in edges.columns:
        edges['highway'] = 'unclassified'

    # 为每条道路分配颜色
    edges['color'] = edges['highway'].apply(
        lambda x: highway_colors.get(x[0] if isinstance(x, list) else x, '#cccccc')
    )

    # 2. 裁剪数据到指定范围
    # 创建边界框多边形


    # 裁剪道路数据
    clipped_edges = clip_map(edges, bbox)

    # 3. 绘制道路
    clipped_edges.plot(ax=ax, linewidth=0.8, color=clipped_edges['color'], alpha=0.7)

    # 4. 创建图例元素
    legend_elements = []
    for hw_type, color in highway_colors.items():
        # 计算该等级道路数量（在裁剪区域内）
        count = len(clipped_edges[clipped_edges['highway'].apply(
            lambda x: hw_type in x if isinstance(x, list) else hw_type == x
        )])
        legend_elements.append(
            mpl.lines.Line2D(
                [0], [0], 
                color=color, 
                lw=3, 
                label=f"{hw_type.capitalize()} ({count}条)"
            )
        )

    ax.legend(
        handles=legend_elements, 
        loc='upper left', 
        title="道路等级",
        fontsize=10,
        title_fontsize=12
    )

    # 5. 添加标题和比例尺
    ax.set_title("上海市道路网络分布 (限定范围)", fontsize=20, pad=20)
    ax.text(0.02, 0.02, "数据来源: 用户提供的GIS图层", 
            transform=ax.transAxes, fontsize=10, color='gray')

    # 6. 添加比例尺（使用限定范围计算）
    minx, miny, maxx, maxy = bounds
    scale_length = (maxx - minx) * 0.1  # 比例尺长度为地图宽度的10%

    # 计算公里数（考虑纬度因素）
    # 1度纬度 ≈ 111公里，1度经度 ≈ 111公里 * cos(纬度)
    center_lat = (miny + maxy) / 2
    km_per_degree = 111 * np.cos(np.radians(center_lat))
    scale_km = scale_length * km_per_degree

    ax.plot([minx + 0.05*(maxx-minx), minx + 0.05*(maxx-minx) + scale_length], 
            [miny + 0.05*(maxy-miny), miny + 0.05*(maxy-miny)], 
            color='black', linewidth=2)
    ax.text(minx + 0.05*(maxx-minx) + scale_length/2, 
            miny + 0.06*(maxy-miny), 
            f"{scale_km:.1f} km", 
            ha='center', fontsize=12)

    # 7. 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))

    # 8. 设置坐标轴范围
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # 9. 添加网格
    ax.grid(True, linestyle='--', alpha=0.5)

    # 10. 显示坐标刻度
    ax.tick_params(axis='both', which='major', labelsize=10)

    save_figure(fig, 'roads.png')

def plot_buildings():
    # %% 按高度着色建筑

    gpkg_path =  str(INPUT_ROOT / "mapdata/shanghai_buildings.gpkg")

    buildings = read_map(gpkg_path)

    # 3. 创建图形
    fig, ax = plt.subplots(figsize=(15, 15))


    # 5. 裁剪数据到指定范围
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    clipped_buildings = clip_map(buildings, bbox)

    # 6. 按高度着色建筑
    if 'height' in clipped_buildings.columns:
        print("建筑高度数据可用，按高度着色...")
    
        # 处理高度数据
        def parse_height(height):
            if pd.isna(height):
                return np.nan
            text = str(height).strip().lower()
            match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*(m|metres|meters|ft|feet)?', text)
            if not match:
                return np.nan
            value = float(match.group(1))
            return value * 0.3048 if match.group(2) in ('ft','feet') else value

        clipped_buildings['height_num'] = clipped_buildings['height'].apply(parse_height)
    
        # 创建高度分类
        bins = [0, 10, 20, 30, 50, 100, 200, 1000]
        labels = ['0–10m', '10–20m', '20–30m', '30–50m', '50–100m', '100–200m', '200–1000m']
        clipped_buildings['height_cat'] = pd.cut(
            clipped_buildings['height_num'], 
            bins=bins, 
            labels=labels,
            include_lowest=True
        )
    
        # 创建颜色映射
        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.BoundaryNorm(bins, cmap.N)
    
        # 绘制建筑
        clipped_buildings.plot(
            ax=ax, 
            column='height_num', 
            cmap=cmap, 
            norm=norm, 
            alpha=0.7, 
            edgecolor='none',
            missing_kwds={'color':'lightgray','label':'Unknown height'}
        )
    
        # 添加颜色条
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
            ax=ax, 
            fraction=0.03, 
            pad=0.04
        )
        cbar.set_label('建筑高度 (米)', fontsize=12)
    
        # 添加图例
        legend_elements = []
        for i, cat in enumerate(labels):
            legend_elements.append(
                mpl.patches.Patch(
                    color=cmap(norm(bins[i])), 
                    label=f"{cat} ({len(clipped_buildings[clipped_buildings['height_cat'] == cat])}栋)"
                )
            )
        legend_elements.append(mpl.patches.Patch(color='lightgray', label=f"Unknown height ({int(clipped_buildings['height_num'].isna().sum())})"))
        ax.legend(
            handles=legend_elements, 
            title="高度分类", 
            loc='upper left', 
            fontsize=10,
            title_fontsize=12
        )
    
    else:
        print("警告：建筑数据中未找到高度信息，使用单一颜色绘制")
        clipped_buildings.plot(
            ax=ax, 
            color='#2c3e50', 
            alpha=0.6, 
            edgecolor='none'
        )

    # 8. 添加标题和比例尺
    ax.set_title("上海市建筑分布 (限定范围)", fontsize=20, pad=20)
    ax.text(0.02, 0.02, "数据来源: 用户提供的GIS图层", 
            transform=ax.transAxes, fontsize=10, color='gray')

    # 9. 添加比例尺（使用限定范围计算）
    minx, miny, maxx, maxy = bounds
    scale_length = (maxx - minx) * 0.1  # 比例尺长度为地图宽度的10%

    # 计算公里数（考虑纬度因素）
    center_lat = (miny + maxy) / 2
    km_per_degree = 111 * np.cos(np.radians(center_lat))
    scale_km = scale_length * km_per_degree

    ax.plot([minx + 0.05*(maxx-minx), minx + 0.05*(maxx-minx) + scale_length], 
            [miny + 0.05*(maxy-miny), miny + 0.05*(maxy-miny)], 
            color='black', linewidth=2)
    ax.text(minx + 0.05*(maxx-minx) + scale_length/2, 
            miny + 0.06*(maxy-miny), 
            f"{scale_km:.1f} km", 
            ha='center', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 10. 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))

    # 11. 设置坐标轴范围
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # 12. 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)

    # 13. 显示坐标刻度
    ax.tick_params(axis='both', which='major', labelsize=10)

    # 14. 添加建筑数量信息
    num_buildings = len(clipped_buildings)
    ax.text(0.98, 0.02, f"建筑总数: {num_buildings:,}", 
            transform=ax.transAxes, fontsize=12, 
            ha='right', bbox=dict(facecolor='white', alpha=0.7))


    save_figure(fig, 'buildings.png')
    # %%公交站点数据
    #

def plot_bus_stops():
    gpkg_path = str(INPUT_ROOT / "mapdata/shanghai_bus_stops.gpkg")
    bus_stops = read_map(gpkg_path)

    # 2. 设置可视化范围 (上海市完整范围)
    bounds = [120.8, 30.67, 122.0, 31.9]  # [minx, miny, maxx, maxy]

    # 3. 创建图形
    fig, ax = plt.subplots(figsize=(15, 15))

    # 4. 确认坐标系并转换
    print(f"原始坐标系: {bus_stops.crs}")
    if bus_stops.crs != 'EPSG:4326':
        print("正在转换坐标系到EPSG:4326...")
        bus_stops = bus_stops.to_crs(epsg=4326)
        print(f"转换后坐标系: {bus_stops.crs}")

    # 5. 裁剪数据到指定范围
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    clipped_stops = clip_map(bus_stops, bbox, points=True)

    # 6. 分析站点类型分布
    if 'public_transport' in clipped_stops.columns:
        # 统计不同类型站点数量
        stop_types = clipped_stops['public_transport'].value_counts()
        print("公交站点类型分布:")
        print(stop_types)
    
        # 创建类型颜色映射
        type_colors = {
            'stop_position': '#e41a1c',    # 红色 - 停靠点
            'platform': '#377eb8',         # 蓝色 - 站台
            'station': '#4daf4a',          # 绿色 - 车站
            'stop_area': '#984ea3',        # 紫色 - 停靠区域
            'other': '#ff7f00'             # 橙色 - 其他类型
        }
    
        # 为每个站点分配颜色
        clipped_stops['color'] = clipped_stops['public_transport'].apply(
            lambda x: type_colors.get(x, type_colors['other'])
        )
    
        # 绘制站点
        clipped_stops.plot(
            ax=ax, 
            color=clipped_stops['color'], 
            markersize=15, 
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
    
        # 创建图例
        legend_elements = []
        for stop_type, color in type_colors.items():
            count = len(clipped_stops[clipped_stops['public_transport'] == stop_type])
            legend_elements.append(
                mpl.lines.Line2D(
                    [0], [0], 
                    marker='o', 
                    color='w', 
                    markerfacecolor=color,
                    markersize=10,
                    label=f"{stop_type.capitalize()} ({count}个)"
                )
            )
    
        ax.legend(
            handles=legend_elements, 
            title="站点类型", 
            loc='upper left', 
            fontsize=10,
            title_fontsize=12
        )
    
    else:
        print("警告：未找到站点类型信息，使用单一颜色绘制")
        clipped_stops.plot(
            ax=ax, 
            color='#e41a1c', 
            markersize=15, 
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )


    # 8. 添加道路网络作为参考
    try:
        # 尝试加载道路数据
        road_gpkg = str(INPUT_ROOT / "mapdata/shanghai_road_network.gpkg")
        roads = read_map(road_gpkg, layer='edges')
    
        if roads.crs != 'EPSG:4326':
            roads = roads.to_crs(epsg=4326)
    
        # 裁剪道路数据
        clipped_roads = clip_map(roads, bbox)
    
        # 简化道路绘制
        clipped_roads.plot(
            ax=ax, 
            linewidth=0.5, 
            color='#555555', 
            alpha=0.3
        )
    
        print("已添加道路网络作为参考")
    except Exception as e:
        print(f"无法加载道路数据: {e}")

    # 9. 添加标题和比例尺
    ax.set_title("上海市公交站点分布 (限定范围)", fontsize=20, pad=20)
    ax.text(0.02, 0.02, "数据来源: 用户提供的GIS图层", 
            transform=ax.transAxes, fontsize=10, color='gray')

    # 10. 添加比例尺（使用限定范围计算）
    minx, miny, maxx, maxy = bounds
    scale_length = (maxx - minx) * 0.1  # 比例尺长度为地图宽度的10%

    # 计算公里数（考虑纬度因素）
    center_lat = (miny + maxy) / 2
    km_per_degree = 111 * np.cos(np.radians(center_lat))
    scale_km = scale_length * km_per_degree

    ax.plot([minx + 0.05*(maxx-minx), minx + 0.05*(maxx-minx) + scale_length], 
            [miny + 0.05*(maxy-miny), miny + 0.05*(maxy-miny)], 
            color='black', linewidth=2)
    ax.text(minx + 0.05*(maxx-minx) + scale_length/2, 
            miny + 0.06*(maxy-miny), 
            f"{scale_km:.1f} km", 
            ha='center', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 11. 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))

    # 12. 设置坐标轴范围
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # 13. 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)

    # 14. 显示坐标刻度
    ax.tick_params(axis='both', which='major', labelsize=10)



    # 16. 添加站点数量信息
    num_stops = len(clipped_stops)
    ax.text(0.98, 0.02, f"公交站点总数: {num_stops:,}", 
            transform=ax.transAxes, fontsize=12, 
            ha='right', bbox=dict(facecolor='white', alpha=0.7))


    save_figure(fig, 'bus_stops.png')

def plot_charging_stations():
    # %% 充电站数据
    # 1. 读取
    gpkg_path = str(INPUT_ROOT / "mapdata/shanghai_charging_stations.gpkg")
    charging_stations = read_map(gpkg_path)

    # 2. 设置可视化范围 (上海市完整范围)
    bounds = [120.8, 30.67, 122.0, 31.9]  # [minx, miny, maxx, maxy]

    # 3. 创建图形
    fig, ax = plt.subplots(figsize=(15, 15))

    # 4. 确认坐标系并转换
    print(f"原始坐标系: {charging_stations.crs}")
    if charging_stations.crs != 'EPSG:4326':
        print("正在转换坐标系到EPSG:4326...")
        charging_stations = charging_stations.to_crs(epsg=4326)
        print(f"转换后坐标系: {charging_stations.crs}")

    # 5. 裁剪数据到指定范围
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    clipped_stations = clip_map(charging_stations, bbox, points=True)

    # 6. 分析充电站属性
    # 创建自定义颜色映射（绿色到红色渐变）
    ev_colors = LinearSegmentedColormap.from_list('ev_colors', ['#2ecc71', '#f1c40f', '#e74c3c'])

    # 7. 按充电能力可视化
    # Coordinates are observed inputs; capacity/socket fields do not establish power in kW.
    clipped_stations.plot(ax=ax, color='#2ecc71', markersize=50, alpha=0.8,
                          edgecolor='white', linewidth=0.8)
    print('Charging station locations shown; no power inferred from capacity/socket tags.')

    # 8. 添加道路网络作为参考
    try:
        # 尝试加载道路数据
        road_gpkg = str(INPUT_ROOT / "mapdata/shanghai_road_network.gpkg")
        roads = read_map(road_gpkg, layer='edges')
    
        if roads.crs != 'EPSG:4326':
            roads = roads.to_crs(epsg=4326)
    
        # 裁剪道路数据
        clipped_roads = clip_map(roads, bbox)
    
        # 简化道路绘制
        clipped_roads.plot(
            ax=ax, 
            linewidth=0.5, 
            color='#7f8c8d', 
            alpha=0.3
        )
    
        print("已添加道路网络作为参考")
    except Exception as e:
        print(f"无法加载道路数据: {e}")

    # 9. 添加标题和比例尺
    ax.set_title("上海市电动汽车充电站分布 (限定范围)", fontsize=20, pad=20)
    ax.text(0.02, 0.02, "数据来源: 用户提供的GIS图层", 
            transform=ax.transAxes, fontsize=10, color='gray')

    # 10. 添加比例尺（使用限定范围计算）
    minx, miny, maxx, maxy = bounds
    scale_length = (maxx - minx) * 0.1  # 比例尺长度为地图宽度的10%

    # 计算公里数（考虑纬度因素）
    center_lat = (miny + maxy) / 2
    km_per_degree = 111 * np.cos(np.radians(center_lat))
    scale_km = scale_length * km_per_degree

    ax.plot([minx + 0.05*(maxx-minx), minx + 0.05*(maxx-minx) + scale_length], 
            [miny + 0.05*(maxy-miny), miny + 0.05*(maxy-miny)], 
            color='black', linewidth=2)
    ax.text(minx + 0.05*(maxx-minx) + scale_length/2, 
            miny + 0.06*(maxy-miny), 
            f"{scale_km:.1f} km", 
            ha='center', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 11. 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))

    # 12. 设置坐标轴范围
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # 13. 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)

    # 14. 显示坐标刻度
    ax.tick_params(axis='both', which='major', labelsize=10)

    # 15. 添加充电站密度分析
    # 创建热力图显示充电站密度
    try:
        if len(clipped_stations) > 100:
            print("添加充电站密度热力图...")
            from scipy.stats import gaussian_kde
    
            # 获取坐标点
            points = np.vstack([clipped_stations.geometry.x, clipped_stations.geometry.y])
    
            # 计算核密度估计
            kde = gaussian_kde(points)
    
            # 创建网格
            xgrid = np.linspace(bounds[0], bounds[2], 100)
            ygrid = np.linspace(bounds[1], bounds[3], 100)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            grid_coords = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
    
            # 计算密度
            Z = kde(grid_coords)
            Z = Z.reshape(Xgrid.shape)
    
            # 绘制密度图
            density = ax.contourf(
                Xgrid, Ygrid, Z, 
                levels=20, 
                cmap='Greens', 
                alpha=0.2,
                antialiased=True
            )
    
            # 添加等高线
            contour = ax.contour(
                Xgrid, Ygrid, Z, 
                levels=10, 
                colors='green', 
                linewidths=0.5,
                alpha=0.5
            )
    
            # 添加密度标签
            ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')


    except (np.linalg.LinAlgError, ValueError) as exc:
        print(f'Charging point density overlay skipped: {exc}')

    # 17. 添加充电站数量信息
    num_stations = len(clipped_stations)
    ax.text(0.98, 0.02, f"充电站总数: {num_stations:,}", 
            transform=ax.transAxes, fontsize=12, 
            ha='right', bbox=dict(facecolor='white', alpha=0.7))

    save_figure(fig, 'charging_stations.png')

def plot_landuse():
    # %% 土地利用

    gpkg_path = str(INPUT_ROOT / "mapdata/shanghai_landuse.gpkg")
    landuse = read_map(gpkg_path)

    # 2. 设置可视化范围 (上海市完整范围)
    bounds = [120.8, 30.67, 122.0, 31.9]  # [minx, miny, maxx, maxy]

    # 3. 创建图形
    fig, ax = plt.subplots(figsize=(15, 15))

    # 4. 确认坐标系并转换
    print(f"原始坐标系: {landuse.crs}")
    if landuse.crs != 'EPSG:4326':
        print("正在转换坐标系到EPSG:4326...")
        landuse = landuse.to_crs(epsg=4326)
        print(f"转换后坐标系: {landuse.crs}")

    # 5. 裁剪数据到指定范围
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    clipped_landuse = clip_map(landuse, bbox)

    # 6. 分析土地利用类型
    # 定义土地利用分类系统及颜色映射
    landuse_categories = {
        'residential': ('住宅区', '#FF9999'),
        'commercial': ('商业区', '#FFCC00'),
        'industrial': ('工业区', '#FF6600'),
        'retail': ('零售区', '#FF3333'),
        'farmland': ('农田', '#99CC66'),
        'forest': ('林地', '#339933'),
        'meadow': ('草地', '#99FF99'),
        'recreation_ground': ('休闲用地', '#66CCCC'),
        'allotments': ('分配地', '#CC9966'),
        'grass': ('草坪', '#CCFFCC'),
        'greenfield': ('绿地', '#99FF66'),
        'construction': ('建设区', '#999999'),
        'quarry': ('采石场', '#996633'),
        'railway': ('铁路用地', '#CCCCCC'),
        'water': ('水域', '#66CCFF'),
        'wetland': ('湿地', '#6699CC'),
        'other': ('其他', '#CCCC99')
    }

    # 7. 标准化土地利用类型
    # 创建土地利用类型列
    if 'fclass' in clipped_landuse.columns:
        clipped_landuse['landuse_type'] = clipped_landuse['fclass'].str.lower()
    elif 'landuse' in clipped_landuse.columns:
        clipped_landuse['landuse_type'] = clipped_landuse['landuse'].str.lower()
    else:
        # 尝试从其他列推断
        clipped_landuse['landuse_type'] = 'other'
        print("警告：未找到标准土地利用分类列，使用默认分类")

    # 将分类映射到标准类别
    def map_to_category(landuse_str):
        landuse_str = str(landuse_str).lower()
        for key in landuse_categories:
            if key in landuse_str:
                return key
        return 'other'

    clipped_landuse['category'] = clipped_landuse['landuse_type'].apply(map_to_category)

    # 8. 统计各类土地利用面积
    # 计算面积（平方米）
    clipped_landuse['area_m2'] = clipped_landuse.to_crs('+proj=laea +lat_0=31.2 +lon_0=121.5 +datum=WGS84 +units=m +no_defs').geometry.area

    # 按类别汇总
    landuse_stats = clipped_landuse.groupby('category').agg(
        count=('category', 'size'),
        total_area=('area_m2', 'sum')
    ).reset_index()

    # 添加中文名称和颜色
    landuse_stats['name_cn'] = landuse_stats['category'].apply(lambda x: landuse_categories[x][0])
    landuse_stats['color'] = landuse_stats['category'].apply(lambda x: landuse_categories[x][1])

    # 计算百分比
    total_area = landuse_stats['total_area'].sum()
    if total_area <= 0:
        raise MapInputUnavailable('No positive polygon area to summarize.')
    landuse_stats['percentage'] = (landuse_stats['total_area'] / total_area) * 100

    # 打印统计信息
    print("\n上海市土地利用统计:")
    print(landuse_stats[['name_cn', 'count', 'total_area', 'percentage']].sort_values('total_area', ascending=False))

    # 9. 创建图例
    legend_patches = []
    for _, row in landuse_stats.iterrows():
        patch = mpatches.Patch(
            color=row['color'],
            label=f"{row['name_cn']} ({row['percentage']:.1f}%)"
        )
        legend_patches.append(patch)

    # 10. 绘制土地利用图
    # 创建分类颜色映射
    cmap = ListedColormap([landuse_categories[cat][1] for cat in landuse_categories])
    norm = colors.BoundaryNorm(
        boundaries=np.arange(len(landuse_categories)+1),
        ncolors=len(landuse_categories)
    )

    # 绘制土地利用多边形
    clipped_landuse.plot(
        ax=ax,
        color=clipped_landuse['category'].map(lambda value: landuse_categories[value][1]),
        edgecolor='none',
        alpha=0.8
    )


    # 12. 添加道路网络和水系作为参考
    try:
        # 添加道路网络
        road_gpkg = str(INPUT_ROOT / "mapdata/shanghai_road_network.gpkg")
        roads = read_map(road_gpkg, layer='edges')
    
        if roads.crs != 'EPSG:4326':
            roads = roads.to_crs(epsg=4326)
    
        # 裁剪道路数据
        clipped_roads = clip_map(roads, bbox)
    
        # 按道路等级绘制
        if 'highway' in clipped_roads.columns:
            # 道路等级颜色映射
            road_colors = {
                'motorway': '#FF0000',
                'trunk': '#FF6600',
                'primary': '#FFCC00',
                'secondary': '#FFFF00',
                'tertiary': '#CCCCCC',
                'residential': '#999999',
                'service': '#666666',
                'other': '#333333'
            }
        
            # 道路等级宽度映射
            road_widths = {
                'motorway': 1.5,
                'trunk': 1.2,
                'primary': 1.0,
                'secondary': 0.8,
                'tertiary': 0.6,
                'residential': 0.4,
                'service': 0.3,
                'other': 0.2
            }
        
            # 绘制道路
            for hw_type in road_colors:
                if hw_type in clipped_roads['highway'].unique():
                    subset = clipped_roads[clipped_roads['highway'] == hw_type]
                    subset.plot(
                        ax=ax,
                        color=road_colors[hw_type],
                        linewidth=road_widths[hw_type],
                        alpha=0.5
                    )
        else:
            # 简化绘制所有道路
            clipped_roads.plot(
                ax=ax,
                color='#333333',
                linewidth=0.3,
                alpha=0.4
            )
    
        print("已添加道路网络作为参考")
    except Exception as e:
        print(f"无法加载道路数据: {e}")

    try:
        # 添加水系
        water_gpkg = str(INPUT_ROOT / "mapdata/shanghai_water.gpkg")
        water = read_map(water_gpkg)
    
        if water.crs != 'EPSG:4326':
            water = water.to_crs(epsg=4326)
    
        # 裁剪水系数据
        clipped_water = clip_map(water, bbox)
    
        # 绘制水系
        clipped_water.plot(
            ax=ax,
            color='#66CCFF',
            edgecolor='#3399CC',
            linewidth=0.5,
            alpha=0.6
        )
    
        print("已添加水系作为参考")
    except Exception as e:
        print(f"无法加载水系数据: {e}")

    # 13. 添加标题和比例尺
    ax.set_title("上海市土地利用分布图", fontsize=20, pad=20)
    ax.text(0.02, 0.02, "数据来源: 用户提供的GIS图层", 
            transform=ax.transAxes, fontsize=10, color='gray')


    # 15. 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))

    # 16. 设置坐标轴范围
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # 17. 添加图例
    ax.legend(
        handles=legend_patches,
        title="土地利用类型 (面积占比)",
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.8,
        fontsize=10,
        title_fontsize=12
    )

    # 18. 添加土地利用统计饼图
    # 在右上角添加饼图
    pie_ax = fig.add_axes([0.75, 0.75, 0.2, 0.2])  # [x, y, width, height]

    # 过滤小面积类别（小于1%）
    significant_stats = landuse_stats[landuse_stats['percentage'] > 1]
    other_percentage = 100 - significant_stats['percentage'].sum()

    # 添加"其他"类别
    if other_percentage > 0:
        other_row = pd.DataFrame({
            'name_cn': ['其他'],
            'percentage': [other_percentage],
            'color': ['#CCCC99']
        })
        pie_data = pd.concat([significant_stats, other_row])
    else:
        pie_data = significant_stats

    # 绘制饼图
    pie_ax.pie(
        pie_data['percentage'],
        labels=pie_data['name_cn'],
        colors=pie_data['color'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 0.5},
        textprops={'fontsize': 8}
    )
    pie_ax.set_title('土地利用面积占比', fontsize=10)

    # 19. 添加城市功能区分析
    # 识别主要城市功能区
    if 'name' in clipped_landuse.columns:
        # 找出面积最大的10个区域
        top_areas = clipped_landuse.nlargest(10, 'area_m2')
    
        # 标注主要功能区
        for idx, row in top_areas.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(
                text=row['name'] if pd.notna(row['name']) else landuse_categories[row['category']][0],
                xy=(centroid.x, centroid.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )

    # 20. 添加行政区划边界
    try:
        # 添加行政区划
        district_gpkg = str(INPUT_ROOT / "mapdata/shanghai_districts.gpkg")
        districts = read_map(district_gpkg)
    
        if districts.crs != 'EPSG:4326':
            districts = districts.to_crs(epsg=4326)
    
        # 裁剪行政区划数据
        clipped_districts = clip_map(districts, bbox)
    
        # 绘制行政区划边界
        clipped_districts.plot(
            ax=ax,
            facecolor='none',
            edgecolor='#333333',
            linewidth=1.0,
            alpha=0.7
        )
    
        # 标注行政区名称
        for idx, row in clipped_districts.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(
                text=row['name'],
                xy=(centroid.x, centroid.y),
                fontsize=12,
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
            )
    
        print("已添加行政区划边界")
    except Exception as e:
        print(f"无法加载行政区划数据: {e}")

    # 21. 添加总土地面积信息
    total_area_km2 = total_area / 1e6
    ax.text(0.98, 0.06, f"输入多边形面积之和: {total_area_km2:,.0f} km²", 
            transform=ax.transAxes, fontsize=12, 
            ha='right', bbox=dict(facecolor='white', alpha=0.7))

    # 22. 添加图例说明
    ax.text(0.98, 0.02, "注: 输入多边形面积占比；重叠未消除", 
            transform=ax.transAxes, fontsize=9, 
            ha='right', color='gray', alpha=0.7)

    save_figure(fig, 'landuse.png')

def plot_pois():
    # %%POI数据
    # 1. 读取



    gpkg_path = str(INPUT_ROOT / "mapdata/shanghai_pois.gpkg")
    pois = read_map(gpkg_path)

    # 2. 设置可视化范围 (上海市完整范围)
    bounds = [120.8, 30.67, 122.0, 31.9]  # [minx, miny, maxx, maxy]

    # 3. 创建图形
    fig, ax = plt.subplots(figsize=(15, 15))

    # 4. 确认坐标系并转换
    print(f"原始坐标系: {pois.crs}")
    if pois.crs != 'EPSG:4326':
        print("正在转换坐标系到EPSG:4326...")
        pois = pois.to_crs(epsg=4326)
        print(f"转换后坐标系: {pois.crs}")

    # 5. 裁剪数据到指定范围
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    clipped_pois = clip_map(pois, bbox, points=True)

    # 6. 分析POI类型
    # 定义POI分类系统及颜色映射
    poi_categories = {
        '餐饮': ('餐饮', '#FF5252', 'restaurant'),
        '购物': ('购物', '#FF9800', 'shopping'),
        '交通': ('交通', '#4CAF50', 'transport'),
        '教育': ('教育', '#2196F3', 'education'),
        '医疗': ('医疗', '#9C27B0', 'healthcare'),
        '金融': ('金融', '#FFEB3B', 'finance'),
        '旅游': ('旅游', '#00BCD4', 'tourism'),
        '政府': ('政府', '#795548', 'government'),
        '公司': ('公司', '#607D8B', 'company'),
        '住宅': ('住宅', '#FFC107', 'residential'),
        '娱乐': ('娱乐', '#E91E63', 'entertainment'),
        '体育': ('体育', '#8BC34A', 'sports'),
        '文化': ('文化', '#3F51B5', 'culture'),
        '酒店': ('酒店', '#009688', 'hotel'),
        '其他': ('其他', '#9E9E9E', 'other')
    }

    # 7. 标准化POI类型
    # 创建POI类型列
    if 'fclass' in clipped_pois.columns:
        clipped_pois['poi_type'] = clipped_pois['fclass']
    elif 'amenity' in clipped_pois.columns:
        clipped_pois['poi_type'] = clipped_pois['amenity']
    elif 'shop' in clipped_pois.columns:
        clipped_pois['poi_type'] = clipped_pois['shop']
    else:
        clipped_pois['poi_type'] = 'other'
        print("警告：未找到标准POI分类列，使用默认分类")

    # 将POI映射到标准类别
    def map_to_category(poi_str):
        if not isinstance(poi_str, str):
            return '其他'
    
        poi_str = poi_str.lower()
    
        # 餐饮相关
        if any(key in poi_str for key in ['restaurant', 'cafe', 'bar', 'food', '快餐', '餐厅', '咖啡']):
            return '餐饮'
    
        # 购物相关
        if any(key in poi_str for key in ['shop', 'store', 'mall', 'market', '购物', '商场', '超市']):
            return '购物'
    
        # 交通相关
        if any(key in poi_str for key in ['bus', 'metro', 'station', 'transport', '交通', '地铁', '公交']):
            return '交通'
    
        # 教育相关
        if any(key in poi_str for key in ['school', 'university', 'college', 'education', '教育', '学校', '大学']):
            return '教育'
    
        # 医疗相关
        if any(key in poi_str for key in ['hospital', 'clinic', 'pharmacy', 'health', '医疗', '医院', '诊所']):
            return '医疗'
    
        # 金融相关
        if any(key in poi_str for key in ['bank', 'atm', 'finance', '金融', '银行', '证券']):
            return '金融'
    
        # 旅游相关
        if any(key in poi_str for key in ['attraction', 'tourist', 'tourism', '旅游', '景点', '公园']):
            return '旅游'
    
        # 政府相关
        if any(key in poi_str for key in ['government', 'office', 'post', '政府', '机关', '邮局']):
            return '政府'
    
        # 公司相关
        if any(key in poi_str for key in ['company', 'office', 'business', '公司', '企业', '写字楼']):
            return '公司'
    
        # 住宅相关
        if any(key in poi_str for key in ['residential', 'apartment', 'housing', '住宅', '小区', '公寓']):
            return '住宅'
    
        # 娱乐相关
        if any(key in poi_str for key in ['entertainment', 'cinema', 'karaoke', '娱乐', '电影院', 'KTV']):
            return '娱乐'
    
        # 体育相关
        if any(key in poi_str for key in ['sports', 'gym', 'stadium', '体育', '健身', '体育馆']):
            return '体育'
    
        # 文化相关
        if any(key in poi_str for key in ['culture', 'museum', 'library', '文化', '博物馆', '图书馆']):
            return '文化'
    
        # 酒店相关
        if any(key in poi_str for key in ['hotel', 'motel', 'inn', '酒店', '宾馆', '旅馆']):
            return '酒店'
    
        return '其他'

    clipped_pois['category'] = clipped_pois['poi_type'].apply(map_to_category)

    # 8. 统计各类POI数量
    poi_stats = clipped_pois['category'].value_counts().reset_index()
    poi_stats.columns = ['category', 'count']
    poi_stats['percentage'] = (poi_stats['count'] / poi_stats['count'].sum()) * 100

    # 添加中文名称和颜色
    poi_stats['name_cn'] = poi_stats['category'].apply(lambda x: poi_categories[x][0])
    poi_stats['color'] = poi_stats['category'].apply(lambda x: poi_categories[x][1])

    # 打印统计信息
    print("\n上海市POI统计:")
    print(poi_stats.sort_values('count', ascending=False))

    # 9. 创建图例
    legend_patches = []
    for _, row in poi_stats.iterrows():
        patch = mpatches.Patch(
            color=row['color'],
            label=f"{row['name_cn']} ({row['count']:,}个, {row['percentage']:.1f}%)"
        )
        legend_patches.append(patch)

    # 10. 绘制POI图
    # 为每个POI分配颜色
    clipped_pois['color'] = clipped_pois['category'].apply(lambda x: poi_categories[x][1])

    # 绘制POI点
    clipped_pois.plot(
        ax=ax,
        color=clipped_pois['color'],
        markersize=8,
        alpha=0.7,
        edgecolor='white',
        linewidth=0.3
    )


    # 12. 添加道路网络作为参考
    try:
        # 添加道路网络
        road_gpkg = str(INPUT_ROOT / "mapdata/shanghai_road_network.gpkg")
        roads = read_map(road_gpkg, layer='edges')
    
        if roads.crs != 'EPSG:4326':
            roads = roads.to_crs(epsg=4326)
    
        # 裁剪道路数据
        clipped_roads = clip_map(roads, bbox)
    
        # 按道路等级绘制
        if 'highway' in clipped_roads.columns:
            # 道路等级颜色映射
            road_colors = {
                'motorway': '#FF0000',
                'trunk': '#FF6600',
                'primary': '#FFCC00',
                'secondary': '#FFFF00',
                'tertiary': '#CCCCCC',
                'residential': '#999999',
                'service': '#666666',
                'other': '#333333'
            }
        
            # 道路等级宽度映射
            road_widths = {
                'motorway': 1.2,
                'trunk': 1.0,
                'primary': 0.8,
                'secondary': 0.6,
                'tertiary': 0.4,
                'residential': 0.3,
                'service': 0.2,
                'other': 0.1
            }
        
            # 绘制道路
            for hw_type in road_colors:
                if hw_type in clipped_roads['highway'].unique():
                    subset = clipped_roads[clipped_roads['highway'] == hw_type]
                    subset.plot(
                        ax=ax,
                        color=road_colors[hw_type],
                        linewidth=road_widths[hw_type],
                        alpha=0.4
                    )
        else:
            # 简化绘制所有道路
            clipped_roads.plot(
                ax=ax,
                color='#333333',
                linewidth=0.3,
                alpha=0.3
            )
    
        print("已添加道路网络作为参考")
    except Exception as e:
        print(f"无法加载道路数据: {e}")

    # 13. 添加标题和比例尺
    ax.set_title("上海市兴趣点(POI)分布图", fontsize=20, pad=20)
    ax.text(0.02, 0.02, "数据来源: 用户提供的GIS图层", 
            transform=ax.transAxes, fontsize=10, color='gray')


    # 15. 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))

    # 16. 设置坐标轴范围
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # 17. 添加图例
    ax.legend(
        handles=legend_patches,
        title="POI类别 (数量及占比)",
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.8,
        fontsize=9,
        title_fontsize=12
    )


    # 20. 添加POI总数信息
    num_pois = len(clipped_pois)
    ax.text(0.98, 0.06, f"POI总数: {num_pois:,}", 
            transform=ax.transAxes, fontsize=12, 
            ha='right', bbox=dict(facecolor='white', alpha=0.7))

    # Unverified hand-positioned annotations omitted.
    # 22. 添加POI多样性分析
    print("分析POI多样性...")
    # 创建网格
    grid_size = 0.05  # angular degrees; not fixed-area 5 km cells
    x_bins = np.arange(bounds[0], bounds[2] + grid_size, grid_size)
    y_bins = np.arange(bounds[1], bounds[3] + grid_size, grid_size)

    # 计算每个网格的POI类型数量
    diversity_grid = []
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            cell = box(x_bins[i], y_bins[j], x_bins[i+1], y_bins[j+1])
            cell_pois = clipped_pois[(clipped_pois.geometry.x >= x_bins[i]) & (clipped_pois.geometry.x < x_bins[i+1]) & (clipped_pois.geometry.y >= y_bins[j]) & (clipped_pois.geometry.y < y_bins[j+1])]
        
            if len(cell_pois) > 0:
                num_categories = cell_pois['category'].nunique()
                center_x = (x_bins[i] + x_bins[i+1]) / 2
                center_y = (y_bins[j] + y_bins[j+1]) / 2
                diversity_grid.append([center_x, center_y, num_categories])

    diversity_df = pd.DataFrame(diversity_grid, columns=['x', 'y', 'diversity'])

    # 绘制多样性热图
    if not diversity_df.empty:
        diversity_scatter = ax.scatter(
            diversity_df['x'], 
            diversity_df['y'], 
            c=diversity_df['diversity'], 
            cmap='viridis',
            s=200,
            alpha=0.5,
            edgecolors='none'
        )
    
        # 添加多样性颜色条
        cbar = fig.colorbar(diversity_scatter, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label('POI类别数（非Shannon熵）', fontsize=12)
    
        # 标注多样性最高的区域
        max_diversity = diversity_df.loc[diversity_df['diversity'].idxmax()]
        ax.annotate(
            text=f"最多类别: {int(max_diversity['diversity'])}类",
            xy=(max_diversity['x'], max_diversity['y']),
            xytext=(0, -20),
            textcoords="offset points",
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="none", alpha=0.7)
        )

    # Unverified hand-positioned annotations omitted.
    save_figure(fig, 'pois.png')

def plot_traffic_lights():
    # %%交通信号灯数据
    # 1. 读取交通信号灯数据
    traffic_lights_path = str(INPUT_ROOT / "mapdata/shanghai_traffic_lights.gpkg")
    traffic_lights = read_map(traffic_lights_path)

    # 2. 设置可视化范围 (上海市中心区域)
    bounds = [121.3, 31.1, 121.6, 31.3]  # [minx, miny, maxx, maxy]

    # 3. 创建图形
    fig, ax = plt.subplots(figsize=(15, 15))

    # 4. 确认坐标系并转换
    print(f"原始坐标系: {traffic_lights.crs}")
    if traffic_lights.crs != 'EPSG:4326':
        print("正在转换坐标系到EPSG:4326...")
        traffic_lights = traffic_lights.to_crs(epsg=4326)
        print(f"转换后坐标系: {traffic_lights.crs}")

    # 5. 裁剪数据到指定范围
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    clipped_lights = clip_map(traffic_lights, bbox, points=True)

    # 6. 统计信号灯总数
    num_lights = len(clipped_lights)
    print(f"上海市中心区域交通信号灯总数: {num_lights}")

    # 7. 绘制信号灯位置
    # 使用统一的红色表示所有信号灯
    clipped_lights.plot(
        ax=ax,
        color='red',
        markersize=8,
        alpha=0.7,
        edgecolor='white',
        linewidth=0.5,
        label=f'交通信号灯 ({num_lights}个)'
    )

    # 9. 添加标题和比例尺
    ax.set_title("上海市交通信号灯位置分布图", fontsize=20, pad=20)
    ax.text(0.02, 0.02, "数据来源: 用户提供的GIS图层", 
            transform=ax.transAxes, fontsize=10, color='gray')



    # 11. 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', 
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))

    # 12. 设置坐标轴范围
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    # 13. 添加图例
    ax.legend(loc='upper right', frameon=True, framealpha=0.8)

    # 14. 添加信号灯密度热力图
    print("添加信号灯密度热力图...")
    # 获取坐标点
    points = clipped_lights.geometry.apply(lambda p: (p.x, p.y)).tolist()
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # 创建热力图
    heatmap = ax.hexbin(
        x, y, 
        gridsize=80, 
        cmap='Reds', 
        alpha=0.5,
        mincnt=1
    )

    # 添加热力图颜色条
    cbar = fig.colorbar(heatmap, ax=ax, label='每个六边形内的要素数')

    # Unverified hand-positioned annotations omitted.
    # 18. 添加信号灯总数信息
    ax.text(0.98, 0.06, f"信号灯总数: {num_lights:,}", 
            transform=ax.transAxes, fontsize=14, 
            ha='right', bbox=dict(facecolor='white', alpha=0.8))

    # 19. 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)


    save_figure(fig, 'traffic_lights.png')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', nargs='+', choices=list(MAPS), default=list(MAPS))
    args = parser.parse_args()
    report = {}
    for name in args.maps:
        try:
            MAPS[name]()
            report[name] = {'status':'saved'}
        except (FileNotFoundError, MapInputUnavailable) as exc:
            plt.close('all')
            report[name] = {'status':'skipped', 'reason':str(exc)}
            print(f'{name}: SKIPPED — {exc}')
    output_path('map_run_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

MAPS = {'roads': plot_roads, 'buildings': plot_buildings, 'bus_stops': plot_bus_stops, 'charging_stations': plot_charging_stations, 'landuse': plot_landuse, 'pois': plot_pois, 'traffic_lights': plot_traffic_lights}

if __name__ == "__main__":
    main()
