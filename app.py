import pandas as pd
import streamlit as st
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    """Load and prepare the initial dataset."""
    try:
        df = pd.read_csv("heart disease para limpiar.csv",
                         sep=";", na_values=["NA", "Na"])
        df['depresion'] = df['depresion'].astype(
            str).str.replace(',', '.').astype(float)
        df['colesterol'] = pd.to_numeric(df['colesterol'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def encode_categorical(df, categorical_cols):
    """Encode categorical variables and return encoders."""
    label_encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        non_null_mask = df_encoded[col].notna()
        df_encoded.loc[non_null_mask, col] = le.fit_transform(
            df_encoded.loc[non_null_mask, col].astype(str))
        label_encoders[col] = le
    return df_encoded, label_encoders


def simple_imputation(df, strategy='mean'):
    """Perform simple imputation using mean, median, or mode."""
    df_imputed = df.copy()
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    imputer = SimpleImputer(strategy=strategy)
    df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_cols] = cat_imputer.fit_transform(
            df[categorical_cols])

    return df_imputed


def knn_imputation(df, n_neighbors=5):
    """Perform KNN imputation."""
    df_imputed = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed.iloc[:, :] = imputer.fit_transform(df)
    return df_imputed


def iterative_imputation(df, max_iter=10):
    """Perform iterative imputation (MICE)."""
    df_imputed = df.copy()
    imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    df_imputed.iloc[:, :] = imputer.fit_transform(df)
    return df_imputed


def regression_imputation(df, target_col):
    """Perform regression imputation for a specific column."""
    df_imputed = df.copy()

    predictor_cols = [col for col in df_imputed.columns if col != target_col]
    initial_imputer = SimpleImputer(strategy='mean')
    df_imputed[predictor_cols] = initial_imputer.fit_transform(
        df_imputed[predictor_cols])

    mask = df_imputed[target_col].isna()
    if not mask.all() and sum(mask) > 0:
        X = df_imputed[predictor_cols]
        y = df_imputed[target_col]

        reg = LinearRegression()
        reg.fit(X[~mask], y[~mask])

        df_imputed.loc[mask, target_col] = reg.predict(X[mask])

    return df_imputed


def plot_missing_values(df):
    """Create a bar plot of missing values."""
    missing_values = df.isnull().sum()
    missing_df = pd.DataFrame(
        {'Column': missing_values.index, 'Missing Values': missing_values.values})
    fig = px.bar(missing_df, x='Column', y='Missing Values',
                 title='Missing Values by Column')
    return fig


def plot_distribution(df, column):
    """Plot distribution of values in a column before and after imputation."""
    fig = px.histogram(df, x=column, title=f'Distribution of {column}')
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical columns."""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numerical_cols].corr()

    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=numerical_cols,
                    y=numerical_cols,
                    color_continuous_scale="RdBu")

    fig.update_layout(title="Correlation Heatmap")
    return fig


def plot_missing_pattern(df):
    """Create missing value pattern visualization."""
    missing = df.isnull()
    fig = px.imshow(missing.T,
                    labels=dict(color="Missing"),
                    aspect="auto",
                    color_continuous_scale=["#ffffff", "#4a90e2"])

    fig.update_layout(title="Missing Value Patterns",
                      xaxis_title="Data Points",
                      yaxis_title="Variables")
    return fig


def plot_distribution_comparison(original_df, imputed_df, column):
    """Plot distribution comparison before and after imputation."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Original Distribution', 'Imputed Distribution'))

    fig.add_trace(
        go.Histogram(x=original_df[column], name="Original"),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(x=imputed_df[column], name="Imputed"),
        row=1, col=2
    )

    fig.update_layout(title_text=f"Distribution Comparison for {column}")
    return fig


def get_method_description(method):
    """Return expanded description for each imputation method."""
    descriptions = {
        "Simple (Mean)": {
            "title": "Imputación por Media",
            "description": """
            ### ¿Qué es?
            La imputación por media reemplaza los valores faltantes con el promedio de los valores existentes en cada columna.

            ### Proceso Detallado
            1. **Cálculo**: Para cada columna numérica:
               - Se suman todos los valores no faltantes.
               - Se divide entre el número total de valores no faltantes.
               - Se reemplazan los valores faltantes con este promedio.
            
            ### Consideraciones Estadísticas
            - **Preservación de la media**: La media de la variable se mantiene igual.
            - **Efecto en la varianza**: Reduce la varianza original.
            - **Efecto en correlaciones**: Puede debilitar las relaciones entre variables.
            
            ### Casos de Uso Ideales
            - Variables con distribución normal.
            - Datos faltantes completamente al azar (MCAR).
            - Pequeña proporción de valores faltantes (<5%).
            
            ### Limitaciones
            1. No considera la estructura de los datos.
            2. Puede distorsionar la distribución.
            3. No apropiado para datos categóricos.
            """,
            "code_example": """
            from sklearn.impute import SimpleImputer
            
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            """
        },
        "Simple (Median)": {
            "title": "Imputación por Mediana",
            "description": """
            ### ¿Qué es?
            La imputación por mediana reemplaza los valores faltantes con la mediana de los valores existentes.

            ### Proceso Detallado
            1. **Cálculo**: Se ordenan los valores no faltantes y se obtiene la mediana.
            2. **Sustitución**: Se reemplazan los valores faltantes con la mediana calculada.

            ### Consideraciones Estadísticas
            - **Más robusta ante valores atípicos** que la media.
            - **No reduce la varianza tanto como la media**.
            
            ### Casos de Uso Ideales
            - Datos con valores atípicos significativos.
            - Variables con distribución sesgada.
            - Datos MCAR o MAR (Missing At Random).

            ### Limitaciones
            1. No conserva la distribución original completamente.
            2. Puede no ser representativa en datos bimodales o multimodales.
            """,
            "code_example": """
            from sklearn.impute import SimpleImputer
            
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            """
        },
        "Simple (Mode)": {
            "title": "Imputación por Moda",
            "description": """
            ### ¿Qué es?
            La imputación por moda reemplaza los valores faltantes con el valor más frecuente en la columna.

            ### Proceso Detallado
            1. **Cálculo**: Se identifica el valor más frecuente en la columna.
            2. **Sustitución**: Se reemplazan los valores faltantes con este valor.

            ### Consideraciones Estadísticas
            - **Funciona bien con datos categóricos**.
            - **Puede sesgar la distribución si la moda no es representativa**.

            ### Casos de Uso Ideales
            - Datos categóricos con valores repetidos.
            - Variables con pocas categorías.

            ### Limitaciones
            1. No adecuado para datos numéricos continuos.
            2. Puede introducir sesgo en la distribución si la moda es dominante.
            """,
            "code_example": """
            from sklearn.impute import SimpleImputer
            
            imputer = SimpleImputer(strategy='most_frequent')
            X_imputed = imputer.fit_transform(X)
            """
        },
        "KNN": {
            "title": "Imputación con K-Nearest Neighbors",
            "description": """
            ### ¿Qué es?
            Estima los valores faltantes en función de los valores más cercanos en el espacio de características.

            ### Proceso Detallado
            1. **Selección de vecinos**: Encuentra los k valores más cercanos en las filas sin valores faltantes.
            2. **Cálculo del promedio o interpolación**: Usa los valores de los vecinos para imputar.

            ### Consideraciones Estadísticas
            - **Captura patrones en los datos**.
            - **Puede ser costoso computacionalmente**.
            - **Sensible a la cantidad de vecinos elegidos**.

            ### Casos de Uso Ideales
            - Datos con patrones de proximidad evidentes.
            - Datos no completamente al azar.
            - Datos con una cantidad moderada de valores faltantes (5%-20%).

            ### Limitaciones
            1. Computacionalmente exigente.
            2. Sensible a la cantidad de vecinos elegidos.
            3. Puede no ser óptimo si los datos tienen alta dimensionalidad.
            """,
            "code_example": """
            from sklearn.impute import KNNImputer
            
            imputer = KNNImputer(n_neighbors=5)
            X_imputed = imputer.fit_transform(X)
            """
        },
        "MICE": {
            "title": "Imputación con MICE",
            "description": """
            ### ¿Qué es?
            La imputación con MICE (Multiple Imputation by Chained Equations) usa modelos estadísticos iterativos para estimar valores faltantes.

            ### Proceso Detallado
            1. **Inicialización**: Se imputan los valores faltantes con una estimación inicial.
            2. **Regresión Iterativa**: Cada variable con valores faltantes se predice en función de las demás.
            3. **Ajuste del Modelo**: Se repite el proceso varias veces hasta la convergencia.

            ### Consideraciones Estadísticas
            - **Captura relaciones entre variables**.
            - **Mejor para datos con valores faltantes MAR (Missing At Random)**.

            ### Casos de Uso Ideales
            - Datos con múltiples valores faltantes.
            - Cuando la correlación entre variables es importante.

            ### Limitaciones
            1. Computacionalmente costoso.
            2. Puede sobreajustar si el número de iteraciones es muy alto.
            """,
            "code_example": """
            from sklearn.impute import IterativeImputer
            
            imputer = IterativeImputer(max_iter=10, random_state=42)
            X_imputed = imputer.fit_transform(X)
            """
        },
        "Regression": {
            "title": "Imputación con Regresión",
            "description": """
            ### ¿Qué es?
            Predice valores faltantes usando regresión lineal basada en otras variables del dataset.

            ### Proceso Detallado
            1. **Selección de variables predictoras**: Se eligen variables sin valores faltantes como predictores.
            2. **Entrenamiento del modelo**: Se entrena una regresión lineal para predecir la variable con valores faltantes.
            3. **Predicción e imputación**: Se predicen los valores faltantes y se reemplazan.

            ### Consideraciones Estadísticas
            - **Mantiene correlaciones dentro del dataset**.
            - **Puede mejorar la imputación en datos con relaciones lineales claras**.

            ### Casos de Uso Ideales
            - Cuando la variable a imputar tiene una relación fuerte con otras variables.
            - Datos con valores faltantes MAR o MCAR.

            ### Limitaciones
            1. Puede introducir sesgo si la relación entre variables no es fuerte.
            2. No es óptimo para relaciones no lineales.
            """,
            "code_example": """
            from sklearn.linear_model import LinearRegression
            
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            """
        }
    }
    return descriptions.get(method, {"title": "Método no encontrado", "description": "Descripción no disponible"})


def plot_missing_values(df):
    """Create enhanced missing values visualization."""
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': missing_percentages.values
    })

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Missing Values Count',
                                        'Missing Values Percentage'),
                        specs=[[{"type": "bar"}, {"type": "bar"}]])

    fig.add_trace(
        go.Bar(x=missing_df['Column'],
               y=missing_df['Missing Values'], name="Count"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=missing_df['Column'],
               y=missing_df['Percentage'], name="Percentage"),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    return fig


def create_sidebar():
    """Create an enhanced sidebar with professional and personal information."""
    # Estilo personalizado para el sidebar
    st.markdown("""
        <style>
        .sidebar-text {
            font-size: 14px;
            color: #fff;
            margin-bottom: 20px;
        }
        .sidebar-header {
            font-size: 18px;
            font-weight: bold;
            color: #fff;
            margin-top: 20px;
        }
        .sidebar-link {
            color: #4a90e2;
            text-decoration: none;
            font-size: 14px;
        }
        .sidebar-divider {
            margin: 20px 0;
            border-top: 1px solid #ddd;
        }
        </style>
    """, unsafe_allow_html=True)

    # st.sidebar.image("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png",
    #                  use_column_width=True)
    st.sidebar.image("logo_generadoporIA.jpg",
                     use_column_width=True)

    # # Información del Proyecto
    # st.sidebar.title("🛠️ Imputacion de Datos")

    # # Separador
    # st.sidebar.markdown(
    #     '<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.sidebar.markdown(
        '<p class="sidebar-header">👨‍💻 Desarrollado por:</p>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p class="sidebar-text">Marco Fidel Mayta Quispe</p>', unsafe_allow_html=True)

    st.sidebar.markdown(
        '<p class="sidebar-header">📊 Data Scientist & ML Engineer</p>', unsafe_allow_html=True)
    st.sidebar.markdown("""
        <p class="sidebar-text">
        Especializado en Machine Learning y Análisis de Datos, 
        con enfoque en técnicas de preprocesamiento y 
        modelado predictivo.
        </p>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(
        '<p class="sidebar-header">🔗 Enlaces Profesionales</p>', unsafe_allow_html=True)

    cols = st.sidebar.columns(4)
    with cols[0]:
        st.markdown("""
            <a href="https://github.com/Maqui2404" target="_blank">
                <img src="https://raw.githubusercontent.com/gauravghongde/social-icons/master/PNG/Black/Github_black.png" width="30">
            </a>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
            <a href="https://linkedin.com/in/marco-mayta-835781170" target="_blank">
                <img src="https://raw.githubusercontent.com/gauravghongde/social-icons/master/PNG/Black/LinkedIN_black.png" width="30">
            </a>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            """
            <a href="https://maqui2404.github.io/PortafolioMarco.github.io/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="30">
            </a>
            """,
            unsafe_allow_html=True
        )

    with cols[3]:
        st.markdown("""
            <a href="https://orcid.org/0009-0009-6019-2925">
                <svg class="w-8 h-8" fill="none" viewBox="0 0 24 24" width="30" height="30">
              <circle cx="12" cy="12" r="10" fill="#A6CE39" />
              <text
                x="7.5"
                y="16"
                font-size="10"
                fill="white"
                font-family="Arial, sans-serif"
                font-weight="bold"
              >
                iD
              </text>
            </svg>
            </a>
        """, unsafe_allow_html=True)

    st.sidebar.markdown(
        '<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.sidebar.markdown(
        '<p class="sidebar-header">📚 Recursos</p>', unsafe_allow_html=True)

    st.sidebar.markdown("""
        ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
        ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
        ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
        ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
        ![Matplotlib](https://img.shields.io/badge/Matplotlib-%2300599C.svg?style=for-the-badge&logo=Matplotlib&logoColor=white)
        ![Seaborn](https://img.shields.io/badge/Seaborn-%23151A1E.svg?style=for-the-badge&logo=seaborn&logoColor=white)
        ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
    """)

    st.sidebar.markdown("""
        * [Documentación Scikit-learn](https://scikit-learn.org/stable/modules/impute.html)
        * [Pandas Documentation](https://pandas.pydata.org/docs/)
        * [NumPy Documentation](https://numpy.org/doc/stable/)
        * [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
        * [Seaborn Documentation](https://seaborn.pydata.org/)
        * [Plotly Documentation](https://plotly.com/python/)
    """)

    st.sidebar.markdown(
        '<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.sidebar.markdown(
        '<p class="sidebar-header">📌 Información del Proyecto</p>', unsafe_allow_html=True)
    st.sidebar.markdown("""
        <p class="sidebar-text">
        Versión: 1.0.0<br>
        Última actualización: Febrero 2024
        </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("### 💻 Tech Stack")
        st.markdown("Python 3.8+")
        st.markdown("Streamlit")
    with col2:
        st.markdown("### 📈 Status")
        st.markdown("Active")
        st.markdown("v1.0.0")


def main():
    st.set_page_config(layout="wide")

    st.markdown("""
        <style>
        .main {
            background-color: #333333;
        }
        .stButton>button {
            width: 100%;
        }
        .stSelectbox {
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🔍 Explorador Dinámico de Métodos de Imputación")

    create_sidebar()

    tab1, tab2, tab3 = st.tabs(
        ["📊 Exploración", "🛠 Imputación", "📘 Aprendizaje"])

    original_df = load_data()
    if original_df is None:
        return

    with tab1:
        st.header("Exploración de Datos")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", len(original_df))
        with col2:
            st.metric("Valores Faltantes", original_df.isnull().sum().sum())
        with col3:
            st.metric("Columnas", len(original_df.columns))

        st.subheader("Vista Previa de Datos")
        st.dataframe(original_df.head())

        st.subheader("Análisis de Valores Faltantes")
        st.plotly_chart(plot_missing_values(
            original_df), use_container_width=True)

        st.subheader("Patrón de Valores Faltantes")
        st.plotly_chart(plot_missing_pattern(
            original_df), use_container_width=True)

        if st.checkbox("Mostrar Matriz de Correlación"):
            st.plotly_chart(create_correlation_heatmap(
                original_df), use_container_width=True)

    with tab2:
        st.header("Imputación de Datos")

        imputation_method = st.selectbox(
            "Selecciona el Método de Imputación",
            ["Simple (Mean)", "Simple (Median)", "Simple (Mode)",
             "KNN", "MICE", "Regression"]
        )

        col1, col2 = st.columns(2)
        with col1:
            if imputation_method == "KNN":
                n_neighbors = st.slider("Número de Vecinos (k)", 1, 20, 5)
            elif imputation_method == "MICE":
                max_iter = st.slider("Máximo de Iteraciones", 1, 50, 10)
            elif imputation_method == "Regression":
                target_column = st.selectbox(
                    "Columna Objetivo", original_df.columns)

        if st.button("Aplicar Imputación", key="impute_button"):
            with st.spinner("Realizando imputación..."):
                try:
                    categorical_cols = [
                        'sex', 'azucar_ayunas', 'enferm_cardiaca']
                    df_encoded, encoders = encode_categorical(
                        original_df, categorical_cols)

                    if imputation_method == "Simple (Mean)":
                        result_df = simple_imputation(df_encoded, 'mean')
                    elif imputation_method == "Simple (Median)":
                        result_df = simple_imputation(df_encoded, 'median')
                    elif imputation_method == "Simple (Mode)":
                        result_df = simple_imputation(
                            df_encoded, 'most_frequent')
                    elif imputation_method == "KNN":
                        result_df = knn_imputation(df_encoded, n_neighbors)
                    elif imputation_method == "MICE":
                        result_df = iterative_imputation(df_encoded, max_iter)
                    else:
                        result_df = regression_imputation(
                            df_encoded, target_column)

                    st.success("¡Imputación completada!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Datos Originales")
                        st.dataframe(original_df.head())
                    with col2:
                        st.subheader("Datos Imputados")
                        st.dataframe(result_df.head())

                    if imputation_method == "Regression":
                        st.plotly_chart(
                            plot_distribution_comparison(
                                original_df, result_df, target_column),
                            use_container_width=True
                        )

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar Datos Imputados",
                        data=csv,
                        file_name="datos_imputados.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error durante la imputación: {str(e)}")

    with tab3:
        st.header("Recursos de Aprendizaje")

        method_info = get_method_description(imputation_method)

        st.subheader(method_info["title"])
        st.markdown(method_info["description"])

        if "code_example" in method_info:
            st.code(method_info["code_example"], language="python")

        st.subheader("Referencias y Recursos")
        st.markdown("""
        📚 **Referencias**:
        * [Documentación Scikit-learn sobre imputación](https://scikit-learn.org/stable/modules/impute.html)
        * [Tutorial sobre manejo de datos faltantes](https://pandas.pydata.org/docs/user_guide/missing_data.html)
        
        🔗 **Enlaces útiles**:
        * [Artículo sobre MICE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/)
        * [Comparación de métodos de imputación](https://www.jstatsoft.org/article/view/v045i03)
        """)


if __name__ == "__main__":
    main()
