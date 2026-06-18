from typing import Any

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def full_eda_report(df: pd.DataFrame) -> tuple[list[Any], list[Any]]:
    """
        Собирает базовую статистику по всему датафрейму одной функцией.

        Parameters
        ----------
        df : pandas

        Returns
        -------
       tuple (num_cols, cat_cols)

        Examples
        --------
        >>> num_cols, cat_cols = full_eda_report(df)
        """


    """Полный EDA отчёт в одной функции"""
    print("=" * 60)
    print(f"DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicates: {df.duplicated().sum():,} ({df.duplicated().mean()*100:.2f}%)")

    print("\n--- DATA TYPES ---")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
        print(missing_df.to_string())
    else:
        print("  No missing values!")

    print("\n--- NUMERIC COLUMNS ---")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  {len(num_cols)} columns: {num_cols}")

    print("\n--- CATEGORICAL COLUMNS ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"  {len(cat_cols)} columns: {cat_cols}")

    return num_cols, cat_cols



def analyze_numeric(df: pd.DataFrame, cols: list = None, target: str = None):
    """
            Детально анализирует числовые колонки.

            Parameters
            ----------
            df : pandas

            cols: list (список числовык фичей)

            target: str (Название колонки таргета)

            Returns
            -------
            -

            Examples
            --------
            >>> analyze_numeric(df, cols=num_cols, target='SalePrice')
            """


    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in cols:
            cols.remove(target)

    summary = []
    for col in cols:
        series = df[col].dropna()
        stat = {
            'column': col,
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'zeros_pct': (series == 0).mean() * 100,
            'negatives_pct': (series < 0).mean() * 100,
        }
        # Количество выбросов (метод IQR)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
        stat['outliers_count'] = outliers
        stat['outliers_pct'] = outliers / len(series) * 100

        # Тест на нормальность (для малых выборок)
        if len(series) <= 5000:
            _, pvalue = stats.shapiro(series.sample(min(len(series), 1000), random_state=42))
            stat['shapiro_p'] = pvalue
            stat['is_normal'] = pvalue > 0.05

        summary.append(stat)

    return pd.DataFrame(summary).set_index('column')


def plot_distributions(df: pd.DataFrame, cols: list = None, n_cols: int = 3, output_dir: str = None):
    """
                Визуализирует числовые фичи 2 графиками: Гистограмма + boxplot
                Сохраняте графики в одно изображение если указана дирректория

                Parameters
                ----------
                df : pandas

                cols : list Список названий числовых фичей

                n_cols: int Количество графиков в одно строке/количество столбцов

                output_dir: str Дирректория в которую сохранять изображение(если не указана, сохранения не будет)

                Returns
                -------
                -

                неявный return (Сохранение изображения)

                Examples
                --------
                >>> plot_distributions(df, cols=['age', 'salary', 'experience'])
                """


    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(6 * n_cols, 4 * n_rows * 2))
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

    for idx, col in enumerate(cols):
        row_hist = (idx // n_cols) * 2
        row_box = row_hist + 1
        col_idx = idx % n_cols

        data = df[col].dropna()

        # Гистограмма с KDE
        ax1 = axes[row_hist, col_idx] if n_rows > 1 else axes[0, col_idx]
        ax1.hist(data, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax1_twin = ax1.twinx()
        data.plot.kde(ax=ax1_twin, color='red', linewidth=2)
        ax1_twin.set_ylabel('')
        ax1.set_title(f'{col}\nSkew: {data.skew():.2f}', fontweight='bold')
        ax1.set_xlabel('')

        # Boxplot
        ax2 = axes[row_box, col_idx] if n_rows > 1 else axes[1, col_idx]
        ax2.boxplot(data, vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
        ax2.set_xlabel(col)

    # Скрываем лишние оси
    for idx in range(len(cols), n_rows * n_cols):
        for r in [0, 1]:
            axes[(idx // n_cols) * 2 + r, idx % n_cols].set_visible(False)

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f'{output_dir}/distributions.png', dpi=150, bbox_inches='tight')
    plt.show()



def plot_correlation_matrix(df: pd.DataFrame, target: str = None,
                             method: str = 'pearson', figsize=(14, 12)):
    """
                    Построение матрицы корреляции

                    Parameters
                    ----------
                    df : pandas

                    target : str Название целевой переменной

                    method: str метод анализа ('pearson', 'spearman', 'kendall')

                    figsize: tuple Размер в plt для построения матрицы

                    Returns
                    -------
                    -

                    Examples
                    --------
                    >>> corr_matrix = plot_correlation_matrix(df, target='SalePrice')
                    """


    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(method=method)

    # Маска для верхнего треугольника
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    # Топ корреляций с таргетом
    if target and target in corr:
        print(f"\nТоп корреляции с '{target}':")
        target_corr = corr[target].drop(target).abs().sort_values(ascending=False)
        for feat, val in target_corr.head(10).items():
            direction = "+" if corr[target][feat] > 0 else "-"
            print(f"  {feat}: {direction}{val:.3f}")

    return corr




def analyze_categorical(df: pd.DataFrame, target: str = None, max_categories: int = 20):
    """
                       Анализирует категориальные переменные:
                       values_count внутри фичи
                       Сравнение с таргетом

                       Parameters
                       ----------
                       df : pandas

                       target : str Название целевой переменной

                       max_categories: int Максимальное количество категорий внутри фичи для анализа

                       Returns
                       -------
                       -

                       Examples
                       --------
                       >>> analyze_categorical(df, target='price')
                       """

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cat_cols:
        n_unique = df[col].nunique()
        print(f"\n{'='*50}")
        print(f"Column: {col} | Unique: {n_unique} | Missing: {df[col].isnull().sum()}")

        if n_unique > max_categories:
            print(f"  Too many categories ({n_unique}), showing top 10:")
            print(df[col].value_counts().head(10).to_string())
        else:
            print(df[col].value_counts(normalize=True).mul(100).round(1).to_string())

        if target and target in df.columns:
            if df[target].dtype in [np.float64, np.int64]:
                # Числовой таргет — показываем среднее
                group_stats = df.groupby(col)[target].agg(['mean', 'median', 'count'])
                group_stats = group_stats.sort_values('mean', ascending=False)
                print(f"\n  Target '{target}' by {col}:")
                print(group_stats.to_string())




def detect_outliers(df: pd.DataFrame, cols: list = None, method: str = 'iqr'):
    """
                           Выявляет выбросы выбранным методом: IQR, Z-index

                           Parameters
                           ----------
                           df : pandas

                           cols : list Названия числовых фичей

                           method: str Метод для выявления выбросов ('iqr', 'zscore', 'percentile')

                           Returns
                           -------
                           -

                           Examples
                           --------
                           >>>  outliers = detect_outliers(df, method='iqr')
                           >>>  print(outliers[outliers.n_outliers > 0].to_string())
                           """

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    results = {}
    for col in cols:
        series = df[col].dropna()
        if method == 'iqr':
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (df[col] < lower) | (df[col] > upper)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outlier_mask = z_scores > 3
            lower = series.mean() - 3 * series.std()
            upper = series.mean() + 3 * series.std()
        elif method == 'percentile':
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            outlier_mask = (df[col] < lower) | (df[col] > upper)

        results[col] = {
            'n_outliers': outlier_mask.sum(),
            'pct_outliers': outlier_mask.mean() * 100,
            'lower_bound': lower,
            'upper_bound': upper,
            'outlier_indices': df.index[outlier_mask].tolist()[:10]  # первые 10
        }

    return pd.DataFrame(results).T

