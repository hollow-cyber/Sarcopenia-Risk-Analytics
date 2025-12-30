import os
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# 使用 Streamlit 缓存，资源只在启动时加载一次
@st.cache_resource
def load_model_assets(method_name="Cox"):
	# 获取当前文件的绝对路径
	# __file__ 指向当前这个 .py 文件
	current_file = Path(__file__).resolve()
	
	# 获取项目根目录 (假设你的文件在 src/ 下，父目录就是根目录)
	project_root = current_file.parent.parent
	base_path = project_root / 'models' / method_name
	
	# 路径安全检查
	if not base_path.exists():
		st.error(f"Model path not found: {base_path}")
		st.stop()
	
	# 加载特征名
	with open(base_path / 'final_model_features.txt', 'r', encoding='utf-8') as f:
		features = f.read().strip().split('\t')
	
	assets = {
		"features": features,
		"preprocessors": joblib.load(base_path / 'final_feature_preprocessors.joblib'),
		"models": joblib.load(base_path / 'final_models.joblib')
	}
	return assets


def cal_single_person_surv_func(personal_data_dict, assets):
	"""
	先通过已训练好的数据处理preprocessors对用户传入的数据进行标准化，
	接着再使用训练好的生存分析模型、所用参数计算该用户的生存函数和相对风险。

	Args:
		personal_data_dict (dict): 用户传入的数据。
		method_name (str): 使用模型的名称。

	Returns:
		(pd.Series, float): 用户的生存函数和相对风险。
	"""
	
	# ================= 1. 加载资源 =================
	features = assets["features"]
	preprocessors = assets["preprocessors"]
	models = assets["models"]
	
	# ================= 2. 数据处理 =================
	# 将用户数据字典转为 DataFrame并确保初始列顺序对齐
	person_data_df = pd.DataFrame([personal_data_dict])[features]
	
	# 存放所有生存函数的列表
	all_survival_funcs = []
	# 存放所有相对风险的列表
	risk_scores = []
	
	# ================= 3. 循环预测 =================
	for model, preprocessor in zip(models, preprocessors):
		# 使用冻结的参数进行数据标准化
		X_processed = preprocessor.transform(person_data_df)
		# 注意：preprocessor里的 ColumnTransformer 会改变列的顺序
		# 必须重新构建 DataFrame 并按照模型需要的顺序重排
		cols = preprocessor.get_feature_names_out()
		X_processed = pd.DataFrame(X_processed, columns=cols)
		
		# Cox 模型对列的顺序很敏感
		# ColumnTransformer 输出的列顺序是按 transformers 列表的顺序来的 (先连续，后分类)
		# 但你的 Cox 模型可能训练时顺序是混着的 (比如 Lasso 选出来的顺序)，所以必须强制重排一下列顺序
		try:
			X_final = X_processed[features]
		except KeyError as e:
			st.error(f"""
			程序发现列名不匹配！标准化处理后的列: {X_processed.columns.tolist()}\n
			生存分析模型需要的列: {features}
			""")
			raise e
		
		# 预测生存函数
		all_survival_funcs.append(model.predict_survival_function(X_final))
		
		# 计算 Partial Hazard (即相对风险)
		# lifelines 的 predict_partial_hazard 返回的是 exp(beta * (x - mean))
		# 因为数据已经标准化了，mean=0，所以这就是 exp(beta * x)
		risk_scores.append(model.predict_partial_hazard(X_final).item())
	
	# ================= 4. 结果集成 =================
	# 对所有生存函数求平均
	# 索引 (Index)：代表 时间轴，值 (Values)：代表 生存概率 S(t)
	avg_survival_func = pd.concat(all_survival_funcs, axis=1).mean(axis=1)
	
	return avg_survival_func, np.mean(risk_scores)


def cal_probability_at_time(survival_func, time):
	"""
	安全地从生存曲线中计算特定时间点的健康和患病概率。

	Args:
		survival_func (pd.Series): 用户的生存函数。
		time (int | float): 需要评估概率的时间点。

	Returns:
		(float, float): 用户在特定时间点的健康和患病概率。
	"""
	
	# 获取生存概率，患病概率就是 1 - 生存概率
	# asof(year) 会找 <= time 的最近时间点的概率
	# 如果 time 小于所有时间点（比如预测第0天），则默认生存率是 1.0
	prob_surv = survival_func.asof(time)
	if pd.isna(prob_surv):
		prob_surv = 1.0
	
	return prob_surv, 1 - prob_surv


def ensure_survival_func_0_time(survival_func):
	"""
	如果生存函数数据中不包含0时刻，则补全起点 (t=0, p=1.0)，因为生存分析的逻辑起点是100%。

	Args:
		survival_func (pd.Series): 用户的生存函数。

	Returns:
		pd.Series: 补全起点后的生存函数，
	"""
	
	if 0 not in survival_func.index:
		survival_func = pd.concat([
			pd.Series([1.0], index=[0.0]),
			survival_func
		]).sort_index()
	
	return survival_func


def plot_survival_curve(survival_func, method_name="Cox", line_style='step', highlight_times=None):
	"""
	根据用户的生存函数绘制其生存曲线。

	Args:
		survival_func (pd.Series): 索引为时间，值为生存率的生存函数。
		method_name (str): 所用生存分析方法名称。
		line_style (str): 'step' (阶梯状，推荐) 或 'smooth' (平滑折线)
		highlight_times (list | None): 需要特别标注的时间点，如 [1, 3, 5, 7]
	"""
	
	# 数据预处理：补全起点 (t=0, p=1.0)
	curve_plot = ensure_survival_func_0_time(survival_func)
	
	# 设置全局字体
	plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
	
	# 创建画布
	fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
	
	# 颜色定义
	line_color = '#2E86C1'  # 稳重的蓝色
	fill_color = '#D6EAF8'  # 浅蓝填充
	dot_color = '#C0392B'  # 醒目的红色标注
	
	# 绘制主曲线，根据参数选择绘图风格
	if line_style == 'step':
		# 阶梯图 (Step-post): 严谨的生存分析画法
		ax.step(curve_plot.index, curve_plot.values, where='post',
		        color=line_color, linewidth=3, label='Survival Probability')
		# 填充曲线下方区域
		ax.fill_between(curve_plot.index, curve_plot.values, step='post',
		                alpha=0.2, color=fill_color)
	else:
		# 平滑折线图: 视觉上更流畅
		ax.plot(curve_plot.index, curve_plot.values,
		        color=line_color, linewidth=3, label='Survival Probability', marker='o', markersize=4)
		ax.fill_between(curve_plot.index, curve_plot.values,
		                alpha=0.2, color=fill_color)
	
	# 动态设置坐标轴范围 (增加留白)
	max_time = curve_plot.index.max()
	# X轴：从 0 开始，右侧多留 10% 的空间
	ax.set_xlim(0, max_time * 1.1)
	# Y轴：从 0 到 1.05 (留一点头顶空间)
	ax.set_ylim(0, 1.05)
	
	# 6. 标注关键时间点
	if highlight_times is not None:
		# 去除掉实际生存函数中不存在的时间点
		highlight_times = [t for t in highlight_times if t <= max_time]
		
		for t in highlight_times:
			prob_surv, _ = cal_probability_at_time(survival_func, t)
			# 画一个圆点
			ax.scatter(t, prob_surv, color=dot_color, s=80, zorder=5, edgecolors='#C0392B', linewidth=2)
			# 添加文字标注
			ax.annotate(f'{prob_surv:.2%}', xy=(t, prob_surv), xytext=(10, 10),
			            textcoords='offset points', fontsize=13, fontweight='bold', color=dot_color)
			# 画虚线引导(垂线)
			ax.vlines(t, 0, prob_surv, linestyles=':', colors='gray', alpha=0.6, linewidth=1.5)
			# 水平虚线
			ax.hlines(prob_surv, 0, t, linestyles=':', colors='gray', alpha=0.6, linewidth=1.5)
	
	# 6. 坐标轴美化
	ax.set_xlabel('Time (Years)', fontsize=14, fontweight='bold', labelpad=10)
	ax.set_ylabel('Survival Probability (No Sarcopenia)', fontsize=14, fontweight='bold', labelpad=10)
	
	# 刻度字体调整
	ax.tick_params(axis='both', which='major', labelsize=12)
	
	# 添加网格 (灰色虚线，不抢眼)
	# ax.grid(True, linestyle='--', alpha=0.5)
	
	# 移除顶部和右侧的边框 (Spines)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	# B. 设置左轴和底轴的粗细
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	
	# 消除 (0,0) 处的突出
	# 强制不显示 0 这个位置的刻度短横线（防止它突出）
	ax.xaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))
	ax.yaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))
	
	# 添加坐标轴箭头 (Arrow)
	# 这里的 transform=ax.transAxes 表示使用相对坐标系 (0~1)
	# (1, 0) 是 X 轴最右端，(0, 1) 是 Y 轴最顶端
	# clip_on=False 保证箭头画在框外时不被切掉
	# X轴箭头
	ax.plot(1, 0, ">", transform=ax.transAxes, clip_on=False,
	        markersize=8, color='black', markeredgewidth=0)
	# Y轴箭头
	ax.plot(0, 1, "^", transform=ax.transAxes, clip_on=False,
	        markersize=8, color='black', markeredgewidth=0)
	
	# 图像标题
	ax.set_title('Individualized Survival Prediction', fontsize=16, fontweight='bold', pad=20)
	
	# 自动调整子图、标签和标题的间距，避免元素重叠或显示不全
	plt.tight_layout()
	
	plt.savefig(rf'survival_results\{method_name}\Individualized Survival Curve.svg')
	plt.show()


def show_altair_survival_chart(survival_func, highlight_times=None):
	"""
	使用 Altair 在streamlit网页中画出带阴影的阶梯状生存曲线。

	Args:
		survival_func (pd.Series): 用户的生存函数。
		highlight_times (list | None): 需要特别标注的时间点，如 [1, 3, 5, 7]
	"""
	
	# 1. 数据准备
	# .reset_index()将生存函数的index放进列当中，变成DataFrame，
	# 原来的 Index 变成了名为 index 的一列，原来的 Values 变成了名为 0 的一列。
	# 后面设定列名后方便alt读取数据画图
	data = ensure_survival_func_0_time(survival_func).reset_index()
	data.columns = ['Time', 'Survival Probability']
	
	# 2. 定义图表
	# 创建一个基础层
	base = alt.Chart(data).encode(
		x=alt.X(shorthand='Time',  # 数据列的名称
		        title='Time (Years)',
		        # 强制刻度最小间隔为1，且格式化为整数
		        axis=alt.Axis(tickMinStep=1, format='d', grid=False)
		        ),
		y=alt.Y(shorthand='Survival Probability',
		        title='Survival Probability (No Sarcopenia)',
		        # Y轴范围固定，格式化为百分比(或.2f)
		        scale=alt.Scale(domain=[0, 1.05]),
		        axis=alt.Axis(format='.2f')
		        ),
		tooltip=[
			# 悬停时显示的第一行：标题是"Time"，数值取自'Time'列，格式为整数('d')
			alt.Tooltip(shorthand='Time', title='Time:', format='d'),
			# 悬停时显示的第二行：标题是"Survival Probability"，数值取自'Survival Probability'列，格式为百分比('.2%')
			alt.Tooltip(shorthand='Survival Probability', title='Survival Probability:', format='.2%')
		]
	)
	
	# 层1: 区域填充 (带阶梯)
	area = base.mark_area(
		opacity=0.4,
		color='#2E86C1',
		interpolate='step-after'  # 【关键】设置阶梯状填充
	)
	
	# 层2: 线条 (带阶梯)
	line = base.mark_line(
		color='#2E86C1',
		interpolate='step-after'  # 【关键】设置阶梯状线条
	)
	
	# 基础图层
	layers = [area, line]
	
	# 可选层3: 标出关键数据点
	if highlight_times is not None:
		# 去除掉实际生存函数中不存在的时间点
		max_time = data["Time"].max()
		highlight_times = [t for t in highlight_times if t <= max_time]
		
		# 使用过滤器保留关键点
		points = base.mark_circle(
			size=100,
			color='red',
			opacity=1  # 只要通过了过滤器的点，全部显示
		).transform_filter(
			# 使用 FieldOneOfPredicate 过滤数据，保留 Time 在指定列表中的数据行
			alt.FieldOneOfPredicate(field='Time', oneOf=highlight_times)
		)
		# 添加该层
		layers.append(points)
	
	# 组合图层
	chart = alt.layer(*layers).properties(
		title='个体化生存预测曲线 (Altair)',
		height=450
	).configure_axis(
		labelFontSize=12,
		titleFontSize=14,
		# labelFont='Times New Roman', # 坐标轴标签字体
		# titleFont='Times New Roman', # 坐标轴标题字体(推荐用中文避免乱码)
		grid=True,  # 开启网格
		gridDash=[2, 2],  # 虚线网格
		gridOpacity=0.3
	).interactive()  # 开启交互 (缩放平移)
	
	st.markdown("### 📈 动态生存轨迹")
	st.caption("💡 提示：将鼠标悬停在曲线上可查看精确数值，支持滚轮缩放和拖拽平移。")
	st.altair_chart(chart)


def get_user_input_sidebar():
	# ================= 第一部分：基本人口学信息 =================
	st.sidebar.subheader("👤 基本信息 (Demographics)")
	
	age = st.sidebar.number_input("年龄：", min_value=50, max_value=999, value=None, placeholder="请输入你的实际年龄")
	
	col1, col2 = st.sidebar.columns(2)
	with col1:
		sex_label = st.radio("性别：", ["男", "女"], index=None, horizontal=True)
		# 映射逻辑
		sex = 1 if sex_label == "男" else (2 if sex_label == "女" else None)
	
	with col2:
		smoker_label = st.radio("吸烟状况", ["是", "否"], index=None, horizontal=True, help="当前是否有吸烟的习惯")
		current_smoker = 1 if smoker_label == "是" else (0 if smoker_label == "否" else None)
	
	st.sidebar.divider()
	
	# ================= 第二部分：核心身体测量 =================
	st.sidebar.subheader("📏 身体测量 (Anthropometrics)")
	
	# 身高体重放在一行，显得紧凑
	c1, c2 = st.sidebar.columns(2)
	with c1:
		height = st.number_input(
			"身高 (cm)",
			min_value=1.0, max_value=999.0, step=0.01, value=None,
			format="%.2f"
		)
	with c2:
		weight = st.number_input(
			"体重 (kg)",
			min_value=1.00, max_value=999.00, step=0.01, value=None,
			format="%.2f"
		)
	
	# --- 实时计算 BMI 并展示 ---
	bmi = None
	if height is not None and weight is not None:
		# 注意：身高 cm 转 m
		bmi = weight / ((height / 100) ** 2)
		
		# 使用 info 或 metric 展示计算结果，给用户正反馈
		if 10 <= bmi <= 50:
			st.sidebar.info(f"📊 自动计算的 BMI: **{bmi:.2f}**")
		else:
			st.sidebar.warning(f"⚠️ 计算出的 BMI (**{bmi:.2f}**) 似乎异常，请检查身高体重。")
	else:
		st.sidebar.caption("👉 输入身高和体重后将自动计算 BMI")
	
	st.sidebar.divider()
	
	# ================= 第三部分：围度指标 =================
	st.sidebar.subheader("📐 围度指标 (Circumferences)")
	
	# 腰臀围可以并排展示
	c3, c4 = st.sidebar.columns(2)
	with c3:
		arm_circumference = st.number_input(
			"上臂围 (Arm Circ. cm)",
			min_value=1.0, max_value=999.0, step=0.1, value=None,
			format="%.2f",
			help="请测量优势手（常用手）上臂中段周长"
		)
		hip_circumference = st.number_input(
			"臀围 (cm)",
			min_value=1.0, max_value=999.0, step=0.01, value=None, format="%.2f",
			help="请测量臀部最粗处的周长"
		)
	with c4:
		waist_circumference = st.number_input(
			"腰围 (cm)",
			min_value=1.0, max_value=999.0, step=0.01, value=None, format="%.2f",
			help="请测量呼气后肚脐处的周长"
		)
		# 小腿围是肌少症最重要的指标之一，建议放显眼位置或加 Help
		calf_circumference = st.number_input(
			"小腿围 (Calf Circ. cm)",
			min_value=1.0, max_value=999.0, step=0.01, value=None,
			format="%.2f",
			help="请测量优势侧小腿最粗处的周长"
		)
	
	# ================= 数据打包 =================
	# 检查是否所有数据都已填好
	all_filled = all(v is not None for v in [
		age, sex, bmi, current_smoker,
		arm_circumference,
		waist_circumference, hip_circumference, calf_circumference,
	])
	
	# 返回字典
	user_data = {
		'age': age,
		'sex': sex,
		'bmi': bmi,
		'current_smoker': current_smoker,
		'arm_circumference': arm_circumference,
		'waist_circumference': waist_circumference,
		'hip_circumference': hip_circumference,
		'calf_circumference': calf_circumference,
	}
	
	return user_data, all_filled


def show_key_metrics(survival_func, eval_times):
	"""
	计算并显示用户在特定时间点的患病风险。

	Args:
		survival_func (pd.Series): 用户的生存函数。
		eval_times (list | None): 需要评估风险的时间点，如 [1, 3, 5, 7]
	"""
	
	# 去除掉实际生存函数中不存在的时间点
	max_time = survival_func.index.max()
	eval_times = [t for t in eval_times if t <= max_time]
	
	st.markdown("### 📊 肌少症患病风险评估")
	
	# 动态生成列：数量 = 列表长度
	cols = st.columns(len(eval_times))
	
	# 遍历列和元素，一一对应输出
	for col, t in zip(cols, eval_times):
		with col:
			_, prob_risk = cal_probability_at_time(survival_func, t)
			# 显示指标结果
			st.metric(label=f"{t}年内患病风险", value=f"{prob_risk:.2%}", delta="长期预测", delta_color="inverse")
	
	# 添加更详细的背书说明 (Badge)
	st.caption(f"""
	🛡️ **模型背书**：本预测基于多变量 Cox 比例风险模型。
	在外部验证集中，模型的区分度 (C-index) 达到 ****，
	校准度 (Brier Score) 表现优异，具有较高的临床参考价值。
	""")


def show_altair_survival_chart(survival_func, highlight_times=None):
	"""
	使用 Altair 在streamlit网页中画出带阴影的阶梯状生存曲线。

	Args:
		survival_func (pd.Series): 用户的生存函数。
		highlight_times (list | None): 需要特别标注的时间点，如 [1, 3, 5, 7]
	"""
	
	# 1. 数据准备
	# .reset_index()将生存函数的index放进列当中，变成DataFrame，
	# 原来的 Index 变成了名为 index 的一列，原来的 Values 变成了名为 0 的一列。
	# 后面设定列名后方便alt读取数据画图
	data = ensure_survival_func_0_time(survival_func).reset_index()
	data.columns = ['Time', 'Survival Probability']
	
	# 2. 定义图表
	# 创建一个基础层
	base = alt.Chart(data).encode(
		x=alt.X(shorthand='Time',  # 数据列的名称
		        title='Time (Years)',
		        # 强制刻度最小间隔为1，且格式化为整数
		        axis=alt.Axis(tickMinStep=1, format='d', grid=False)
		        ),
		y=alt.Y(shorthand='Survival Probability',
		        title='Survival Probability (No Sarcopenia)',
		        # Y轴范围固定，格式化为百分比(或.2f)
		        scale=alt.Scale(domain=[0, 1.05]),
		        axis=alt.Axis(format='.2f')
		        ),
		tooltip=[
			# 悬停时显示的第一行：标题是"Time"，数值取自'Time'列，格式为整数('d')
			alt.Tooltip(shorthand='Time', title='Time:', format='d'),
			# 悬停时显示的第二行：标题是"Survival Probability"，数值取自'Survival Probability'列，格式为百分比('.2%')
			alt.Tooltip(shorthand='Survival Probability', title='Survival Probability:', format='.2%')
		]
	)
	
	# 层1: 区域填充 (带阶梯)
	area = base.mark_area(
		opacity=0.4,
		color='#2E86C1',
		interpolate='step-after'  # 【关键】设置阶梯状填充
	)
	
	# 层2: 线条 (带阶梯)
	line = base.mark_line(
		color='#2E86C1',
		interpolate='step-after'  # 【关键】设置阶梯状线条
	)
	
	# 基础图层
	layers = [area, line]
	
	# 可选层3: 标出关键数据点
	if highlight_times is not None:
		# 去除掉实际生存函数中不存在的时间点
		max_time = data["Time"].max()
		highlight_times = [t for t in highlight_times if t <= max_time]
		
		# 使用过滤器保留关键点
		points = base.mark_circle(
			size=100,
			color='red',
			opacity=1  # 只要通过了过滤器的点，全部显示
		).transform_filter(
			# 使用 FieldOneOfPredicate 过滤数据，保留 Time 在指定列表中的数据行
			alt.FieldOneOfPredicate(field='Time', oneOf=highlight_times)
		)
		# 添加该层
		layers.append(points)
	
	# 组合图层
	chart = alt.layer(*layers).properties(
		title='个体化生存预测曲线 (Altair)',
		height=450
	).configure_axis(
		labelFontSize=12,
		titleFontSize=14,
		# labelFont='Times New Roman', # 坐标轴标签字体
		# titleFont='Times New Roman', # 坐标轴标题字体(推荐用中文避免乱码)
		grid=True,  # 开启网格
		gridDash=[2, 2],  # 虚线网格
		gridOpacity=0.3
	).interactive()  # 开启交互 (缩放平移)
	
	st.markdown("### 📈 动态生存轨迹")
	st.caption("💡 提示：将鼠标悬停在曲线上可查看精确数值，支持滚轮缩放和拖拽平移。")
	st.altair_chart(chart)
