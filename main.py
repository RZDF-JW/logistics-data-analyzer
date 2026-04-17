import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from openpyxl.styles import Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# ====================== 环境兼容配置 ======================
def setup_environment():
    try:
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
            os.environ['MATPLOTLIBRC'] = os.path.join(base_path, 'matplotlib')
    except Exception:
        pass
setup_environment()

# ====================== 基础配置 ======================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
DPI = 300
st.set_page_config(page_title="物流数据分析工具", layout="wide")

# ====================== 输出目录配置 ======================
def get_output_dir():
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.expanduser("~"), "Desktop", "物流分析报告")
    else:
        return "物流分析报告"
OUTPUT_DIR = get_output_dir()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 数据加载模块（支持多文件合并） ======================
@st.cache_data
def load_data(uploaded_files):
    dfs = []
    for uploaded_file in uploaded_files:
        try:
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext == '.csv':
                try:
                    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                except:
                    df = pd.read_csv(uploaded_file, encoding="gbk")
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"❌ 不支持的文件格式：{uploaded_file.name}")
                continue
            df.columns = [str(col).strip().replace(" ", "") for col in df.columns]
            dfs.append(df)
        except Exception as e:
            st.error(f"❌ 读取文件 {uploaded_file.name} 失败：{str(e)}")
    if not dfs:
        return None
    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined

# ====================== 数据清洗模块（含异常值处理） ======================
@st.cache_data
def clean_data(df, scene):
    try:
        # 自动匹配列名
        standard_cols = {
            "到港日期": ["到港日期","入库日期","日期"],
            "离港日期": ["离港日期","出库日期"],
            "货物品类": ["货物品类","物料品类","品类","货物名称"],
            "重量（吨）": ["重量（吨）","库存数量","数量","重量"],
            "仓储费用（元）": ["仓储费用（元）","仓储费","保管费","仓租"],
            "运输费用（元）": ["运输费用（元）","运输费","运费","物流费"]
        }
        rename_map = {}
        for std, alias_list in standard_cols.items():
            for alias in alias_list:
                if alias in df.columns:
                    rename_map[alias] = std
        df.rename(columns=rename_map, inplace=True)

        required_cols = ["到港日期","离港日期","货物品类"]
        if not all(col in df.columns for col in required_cols):
            st.error("❌ 缺少核心字段：至少需要【日期】【品类】字段")
            return None

        # 日期清洗
        df["到港日期"] = pd.to_datetime(df["到港日期"], errors="coerce")
        df["离港日期"] = pd.to_datetime(df["离港日期"], errors="coerce")
        df = df.dropna(subset=["到港日期","离港日期"])
        
        df["季度"] = df["到港日期"].dt.quarter
        df["季度名称"] = df["季度"].map({1:"第一季度",2:"第二季度",3:"第三季度",4:"第四季度"})

        # 数值清洗 + 异常值拦截
        df["重量（吨）"] = pd.to_numeric(df.get("重量（吨）", 0), errors="coerce").fillna(0)
        df["仓储费用（元）"] = pd.to_numeric(df.get("仓储费用（元）", 0), errors="coerce").fillna(0)
        df["运输费用（元）"] = pd.to_numeric(df.get("运输费用（元）", 0), errors="coerce").fillna(0)
        
        # 剔除异常值
        before_count = len(df)
        df = df[(df["重量（吨）"] >= 0) & (df["仓储费用（元）"] >= 0) & (df["运输费用（元）"] >= 0)]
        df["周转天数"] = (df["离港日期"] - df["到港日期"]).dt.days
        df = df[df["周转天数"] >= 0]
        after_count = len(df)
        if before_count > after_count:
            st.warning(f"⚠️ 已自动剔除 {before_count - after_count} 条异常数据（负数/负天数）")

        df["总费用（元）"] = df["仓储费用（元）"] + df["运输费用（元）"]
        df["单吨成本"] = (df["总费用（元）"] / df["重量（吨）"].replace(0, 1)).round(2)
        
        return df
    except Exception as e:
        st.error(f"❌ 数据清洗失败：{str(e)}")
        return None

# ====================== 品类分析模块 ======================
@st.cache_data
def analyze_category(df):
    result = df.groupby("货物品类", dropna=False).agg(
        总重量=("重量（吨）","sum"),
        总仓储费=("仓储费用（元）","sum"),
        总运输费=("运输费用（元）","sum"),
        总费用=("总费用（元）","sum"),
        平均周转天数=("周转天数","mean")
    ).reset_index()
    result["平均周转天数"] = result["平均周转天数"].round(1)
    return result

# ====================== 风险预警模块 ======================
def risk_warning(df, threshold):
    try:
        mean_days = df.groupby("货物品类")["周转天数"].mean().reset_index()
        mean_days.columns = ["货物品类", "品类平均周转天数"]
        df = df.merge(mean_days, on="货物品类")

        df["风险等级"] = "低风险"
        df.loc[df["周转天数"] > df["品类平均周转天数"] * threshold, "风险等级"] = "中风险"
        df.loc[df["周转天数"] > df["品类平均周转天数"] * threshold * 1.5, "风险等级"] = "高风险"
        
        return df, df[df["风险等级"] != "低风险"]
    except Exception as e:
        st.error(f"❌ 风险分析失败：{str(e)}")
        return df, pd.DataFrame()

# ====================== 季度分析模块 ======================
def season_analysis(df):
    abnormal_data = df[df["风险等级"] != "低风险"]
    if abnormal_data.empty:
        return pd.DataFrame(columns=["季度名称", "异常数量"]), abnormal_data
    season_result = abnormal_data.groupby("季度名称").agg(异常数量=("货物品类","count")).reset_index()
    return season_result, abnormal_data

# ====================== 可视化模块（专业配色+标签） ======================
def visualize_all(df, cat_result, season_result):
    # 1. 品类占比饼图
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    total_weight = cat_result["总重量"].sum()
    if total_weight > 0:
        cat_result["占比"] = cat_result["总重量"] / total_weight
        main_data = cat_result[cat_result["占比"] >= 0.01]
        other_sum = cat_result[cat_result["占比"] < 0.01]["总重量"].sum()
        labels = list(main_data["货物品类"]) + ["其他"]
        sizes = list(main_data["总重量"]) + [other_sum]
        ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Set2"))
    ax1.set_title("各品类货物重量/数量占比", fontsize=14)
    st.pyplot(fig1)
    plt.close(fig1)

    # 2. 费用对比柱状图
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x_pos = range(len(cat_result))
    width = 0.3
    ax2.bar([i - width/2 for i in x_pos], cat_result["总仓储费"], width=width, label="仓储费用", color="#1f77b4")
    ax2.bar([i + width/2 for i in x_pos], cat_result["总运输费"], width=width, label="运输费用", color="#ff7f0e")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cat_result["货物品类"], rotation=30)
    ax2.legend()
    ax2.set_title("各品类费用构成对比", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # 3. 周转天数分布直方图
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["周转天数"], bins=10, kde=True, color="#2ca02c")
    ax3.set_title("周转天数分布情况", fontsize=14)
    st.pyplot(fig3)
    plt.close(fig3)

    # 4. 季度异常统计柱状图
    if not season_result.empty:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.bar(season_result["季度名称"], season_result["异常数量"], color="#d62728")
        ax4.set_title("各季度异常数量统计", fontsize=14)
        st.pyplot(fig4)
        plt.close(fig4)

# ====================== Excel导出模块（格式美化+双路径） ======================
def export_excel(df, cat_result, risk_list, season_result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"物流分析报告_{timestamp}.xlsx")

    summary = pd.DataFrame({
        "统计指标": ["总记录数", "总重量/数量", "总费用(元)", "异常数据量"],
        "数值": [len(df), df["重量（吨）"].sum(), df["总费用（元）"].sum(), len(risk_list)]
    })

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        # 写入数据
        summary.to_excel(writer, sheet_name="分析汇总", index=False)
        df.to_excel(writer, sheet_name="清洗后原始数据", index=False)
        cat_result.to_excel(writer, sheet_name="品类分析汇总", index=False)
        risk_list.to_excel(writer, sheet_name="风险预警清单", index=False)
        if not season_result.empty:
            season_result.to_excel(writer, sheet_name="季度异常分析", index=False)

        # 美化格式
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            # 表头加粗
            for cell in ws[1]:
                cell.font = Font(bold=True)
            # 自动调整列宽
            for column_cells in ws.columns:
                max_length = 0
                column = column_cells[0].column_letter
                for cell in column_cells:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column].width = adjusted_width
    return save_path

# ====================== 风险等级高亮样式 ======================
def highlight_risk(row):
    if row["风险等级"] == "高风险":
        return ["background-color: #ffcccc"] * len(row)
    elif row["风险等级"] == "中风险":
        return ["background-color: #fff2cc"] * len(row)
    else:
        return [""] * len(row)

# ====================== 主程序（含交互式筛选+数据概览） ======================
def main():
    st.title("🚢 物流港口 & 企业仓储 通用数据分析工具")
    st.sidebar.header("⚙️ 核心参数设置")
    
    scene = st.sidebar.radio("选择业务场景", ["港口物流", "企业仓储"])
    threshold = st.sidebar.slider("风险预警阈值（倍数）", 1.0, 3.0, 1.5, 0.1)

    # 支持多文件上传
    uploaded_files = st.file_uploader("📁 上传数据文件（可多选）", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("💡 请上传数据文件开始分析")
        return

    df = load_data(uploaded_files)
    if df is None:
        st.stop()

    # --- 轻量优化1：数据概览面板 ---
    st.divider()
    st.subheader("📊 数据概览")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总记录数", f"{len(df):,}")
    with col2:
        if "到港日期" in df.columns:
            min_date = pd.to_datetime(df["到港日期"]).min().strftime("%Y-%m-%d")
            max_date = pd.to_datetime(df["到港日期"]).max().strftime("%Y-%m-%d")
            st.metric("时间跨度", f"{min_date} ~ {max_date}")
    with col3:
        st.metric("缺失值行数", f"{df.isnull().any(axis=1).sum():,}")
    with col4:
        if "货物品类" in df.columns:
            st.metric("货品种类数", f"{df['货物品类'].nunique()}")

    with st.expander("🔍 查看原始数据预览"):
        st.dataframe(df.head(), use_container_width=True)

    df_clean = clean_data(df, scene)
    if df_clean is None or df_clean.empty:
        st.stop()

    # --- 进阶优化2：交互式筛选 ---
    st.sidebar.subheader("🔍 数据筛选")
    selected_seasons = st.sidebar.multiselect("按季度筛选", df_clean["季度名称"].unique(), default=df_clean["季度名称"].unique())
    selected_categories = st.sidebar.multiselect("按货物品类筛选", df_clean["货物品类"].unique(), default=df_clean["货物品类"].unique())
    
    df_filtered = df_clean[(df_clean["季度名称"].isin(selected_seasons)) & (df_clean["货物品类"].isin(selected_categories))]
    if df_filtered.empty:
        st.warning("⚠️ 当前筛选条件下无数据，请调整筛选")
        return

    # 核心分析
    df_risk, risk_data = risk_warning(df_filtered, threshold)
    category_data = analyze_category(df_filtered)
    season_data, _ = season_analysis(df_risk)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 品类分析结果")
        st.dataframe(category_data, use_container_width=True)
    with col2:
        risk_title = "滞港风险" if scene == "港口物流" else "库存积压风险"
        st.subheader(f"⚠️ {risk_title}预警")
        if not risk_data.empty:
            show_cols = ["货物品类", "周转天数", "品类平均周转天数", "风险等级"]
            # --- 轻量优化5：风险高亮 ---
            styled_risk = risk_data[show_cols].style.apply(highlight_risk, axis=1)
            st.dataframe(styled_risk, use_container_width=True)
        else:
            st.success("✅ 暂无风险数据，运营正常！")

    st.divider()
    st.subheader("🌍 季度异常数据分析")
    if not season_data.empty:
        st.dataframe(season_data, use_container_width=True)
    else:
        st.info("ℹ️ 暂无异常数据，无需季度分析")

    st.divider()
    st.subheader("📈 数据可视化图表")
    visualize_all(df_filtered, category_data, season_data)

    # --- 轻量优化6：双路径导出 ---
    export_path = export_excel(df_filtered, category_data, risk_data, season_data)
    st.success(f"✅ 分析完成！报告已自动保存至：\n{export_path}")
    
    # 网页下载按钮
    with open(export_path, "rb") as f:
        st.download_button(
            label="📥 点击下载Excel报告",
            data=f,
            file_name=os.path.basename(export_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()